#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

import torch
import time 
import numpy as np
from typing import List, Callable
import torch.nn as nn
import torch.distributed as dist
from torch.multiprocessing import Process, Queue
from pyhessian.utils import group_product, group_add, normalization, get_params_grad, DataPartitioner,\
    orthnormal, init_process


def hv_product(rank: int, size: int, model: nn.Module, data: List[torch.Tensor],
               criterion: Callable, v: List[torch.Tensor]):
    # group of processes
    group = dist.new_group(list(range(size)))
    # current process device
    device = torch.device("cuda:{}".format(rank))

    # count the number of datum points in the dataloader
    num_data = 0
    # to accumulate the hessian vector product
    temp_hv = []

    for inputs, targets in data:
        model.zero_grad()
        tmp_num_data = inputs.size(0)
            
        if rank == 0: 
            for_start = time.time()

        outputs = model(inputs.to(device))
        loss = criterion(outputs, targets.to(device))
        
        if rank == 0:
            for_end = time.time()
            print("Time to forward Hv: %f" % (for_end - for_start))    

        if rank == 0: 
            back_start = time.time()

        loss.backward(create_graph=True)
        
        
        if rank == 0:
            back_end = time.time()
            print("Time to back Hv: %f" % (back_end - back_start))    
        
        
        if rank == 0: 
            back2_start = time.time()
        
        params, gradsH = get_params_grad(model)
        # model.zero_grad()
        Hv = torch.autograd.grad(gradsH,
                                 params,
                                 grad_outputs=v,
                                 only_inputs=True,
                                 retain_graph=False)
        
        if rank == 0:
            back2_end = time.time()
            print("Time to back2 Hv: %f" % (back2_end - back2_start))   
            
        # accumulating the eigenvector
        if len(temp_hv) == 0:
            temp_hv = [
                Hv1.contiguous() * float(tmp_num_data) + 0.
                for Hv1 in Hv
            ]
        else:
            temp_hv = [
                temp_hv1 + Hv1 * float(tmp_num_data) + 0.
                for temp_hv1, Hv1 in zip(temp_hv, Hv)
            ]
        num_data += float(tmp_num_data)

    # reducing the temp_hv after every i
    if rank == 0: 
        reduce_start = time.time()
        
    for t in range(len(temp_hv)):
        dist.all_reduce(temp_hv[t], op=dist.ReduceOp.SUM, group=group)
    
    if rank == 0:
        reduce_end = time.time()
        print("Time to reduce Hv: %f" % (reduce_end - reduce_start))        
        
    temp_hv = [temp_hv1 / float(size * num_data) for temp_hv1 in temp_hv]
    eigenvalue = group_product(temp_hv, v).cpu().item()

    return eigenvalue, temp_hv


def eigenvalue_(rank: int, size: int, model: nn.Module, data: List[torch.Tensor],
                criterion: Callable, queue: Queue, max_iter: int = 100, tol: float = 1e-3, top_n: int = 1):
    # group of processes
    group = dist.new_group(list(range(size)))
    # current process device
    device = torch.device("cuda:{}".format(rank))

    assert top_n >= 1

    eigenvalues = []
    eigenvectors = []

    computed_dim = 0

    # moving model to respective GPU
    model = model.to(device)
    # setting model to eval mode
    model.eval()
    # getting model params and gradients
    params, gradsH = get_params_grad(model)

    while computed_dim < top_n:
        eigenvalue = None
        # generate random vector
        v = [torch.randn(p.size()).to(device) for p in params]
        v = normalization(v)

        # since we have manually set the seed for cuda
        # same v should be initialized across all GPUs
        # but this step is necessary if the seed is
        # not set
        if rank == 0: 
            bcast_start = time.time()
        for t in v:
            dist.broadcast(t, src=0, group=group)
    
        if rank == 0:
            bcast_end = time.time()
            print("Time to bcast Hv: %f" % (bcast_end - bcast_start))  
            
            
        if rank == 0: 
            total_start = time.time()
        for i in range(max_iter):
            print (i)
            v = orthnormal(v, eigenvectors)

            tmp_eigenvalue, Hv = hv_product(rank, size, model, data, criterion, v)
            v = normalization(Hv)

            if eigenvalue is None:
                eigenvalue = tmp_eigenvalue
            else:
                if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) +
                                                       1e-6) < tol:
                    break
                else:
                    eigenvalue = tmp_eigenvalue
    
        if rank == 0:
            total_end = time.time()
            print("Time to total Hv: %f" % (total_end - total_start))  

        eigenvalues.append(eigenvalue)
        computed_dim += 1

    # communicating eigenvalues and eigenvectors to parent process
    if rank == 0:
        queue.put(eigenvalues)


def eigenvalue(size: int, model: nn.Module, data_partitions: DataPartitioner,
                criterion: Callable, ip: str):
    processes = []
    queue = Queue()
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, model, data_partitions.use(rank),
                                               criterion, queue, eigenvalue_, ip))
        p.start()
        processes.append(p)

    eigenvalues = queue.get()

    for p in processes:
        p.join()

    return eigenvalues


def trace_(rank: int, size: int, model: nn.Module, data: List[torch.Tensor], criterion: Callable,
           queue: Queue, max_iter: int = 100, tol: float = 1e-3):
    # group of processes
    group = dist.new_group(list(range(size)))
    # current process device
    device = torch.device("cuda:{}".format(rank))

    trace_vhv = []
    trace = 0.
            
    if rank == 0: 
        total_start = time.time()
        
    # moving model to respective GPU
    model = model.to(device)
    # setting model to eval mode
    model.eval()
    # getting model params and gradients
    params, gradsH = get_params_grad(model)

    for i in range(max_iter):
        print (i)
        # generate random vector
        v = [torch.randint_like(p, high=2, device=device) for p in params]

        # generate Rademacher random variables
        for v_i in v:
            v_i[v_i == 0] = -1

        # since we have manually set the seed for cuda
        # same v should be initialized across all GPUs
        # but this step is necessary if the seed is
        # not set
        
        if rank == 0: 
            bcast_start = time.time()
            
        for t in v:
            dist.broadcast(t, src=0, group=group)


        if rank == 0:
            bcast_end = time.time()
            print("Time to bcast Hv: %f" % (bcast_end - bcast_start))  
            
        _, Hv = hv_product(rank, size, model, data, criterion, v)

        prod = group_product(Hv, v)
        trace_vhv.append(prod.item())

        if abs(np.mean(trace_vhv) - trace) / (trace + 1e-6) < tol:
            break
        else:
            trace = np.mean(trace_vhv)

    if rank == 0:
        total_end = time.time()
        print("Time to total Hv: %f" % (total_end - total_start))  
    # communicating eigenvalues and eigenvectors to parent process
    if rank == 0:
        queue.put(np.mean(trace_vhv))


def trace(size: int, model: nn.Module, data_partitions: DataPartitioner, criterion: Callable, ip: str):
    processes = []
    queue = Queue()
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, model, data_partitions.use(rank),
                                               criterion, queue, trace_, ip))
        p.start()
        processes.append(p)

    trace = queue.get()

    for p in processes:
        p.join()

    return trace


def density_(rank: int, size: int, model: nn.Module, data: List[torch.Tensor],
             criterion: Callable, queue: Queue, iter=100, n_v=1):
    """
    compute estimated eigenvalue density using stochastic lanczos algorithm (SLQ)
    iter: number of iterations used to compute trace
    n_v: number of SLQ runs
    """
    # group of processes
    group = dist.new_group(list(range(size)))
    # current process device
    device = torch.device("cuda:{}".format(rank))

    eigen_list_full = []
    weight_list_full = []

    # moving model to respective GPU
    model = model.to(device)
    # setting model to eval mode
    model.eval()
    # getting model params and gradients
    params, gradsH = get_params_grad(model)

    for k in range(n_v):
        v = [torch.randint_like(p, high=2, device=device) for p in params]

        # generate Rademacher random variables
        for v_i in v:
            v_i[v_i == 0] = -1
        v = normalization(v)

        # since we have manually set the seed for cuda
        # same v should be initialized across all GPUs
        # but this step is necessary if the seed is
        # not set
        for t in v:
            dist.broadcast(t, src=0, group=group)

        # standard lanczos algorithm initlization
        v_list = [v]
        w_list = []
        alpha_list = []
        beta_list = []

        for i in range(iter):
            print (i)
            if i == 0:
                _, w_prime = hv_product(rank, size, model, data, criterion, v)
                alpha = group_product(w_prime, v)
                alpha_list.append(alpha.cpu().item())
                w = group_add(w_prime, v, alpha=-alpha)
                w_list.append(w)
            else:
                beta = torch.sqrt(group_product(w, w))
                beta_list.append(beta.cpu().item())
                if beta_list[-1] != 0.:
                    # We should re-orth it
                    v = orthnormal(w, v_list)
                    v_list.append(v)
                else:
                    # generate a new vector
                    w = [torch.randn(p.size()).to(device) for p in params]
                    for t in w:
                        dist.broadcast(t, src=0, group=group)
                    v = orthnormal(w, v_list)
                    v_list.append(v)
                _, w_prime = hv_product(rank, size, model, data, criterion, v)
                alpha = group_product(w_prime, v)
                alpha_list.append(alpha.cpu().item())
                w_tmp = group_add(w_prime, v, alpha=-alpha)
                w = group_add(w_tmp, v_list[-2], alpha=-beta)
            torch.cuda.synchronize(device)

    if rank == 0:
        T = torch.zeros(iter, iter).to(device).contiguous()
        for i in range(len(alpha_list)):
            T[i, i] = alpha_list[i]
            if i < len(alpha_list) - 1:
                T[i + 1, i] = beta_list[i]
                T[i, i + 1] = beta_list[i]
        a_, b_ = torch.eig(T, eigenvectors=True)
        print ("Done cal eig of T")
        eigen_list = a_[:, 0]
        weight_list = b_[0, :]**2

        eigen_list_full.append(list(eigen_list.cpu().numpy()))
        weight_list_full.append(list(weight_list.cpu().numpy()))
        queue.put(eigen_list_full)
        queue.put(weight_list_full)


def density(size: int, model: nn.Module, data_partitions: DataPartitioner, criterion: Callable, ip: str):
    processes = []
    queue = Queue()
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, model, data_partitions.use(rank), criterion, queue,
                                               density_, ip))
        p.start()
        processes.append(p)

    eigen_list_full = queue.get()
    weight_list_full = queue.get()

    for p in processes:
        p.join()

    return eigen_list_full, weight_list_full
