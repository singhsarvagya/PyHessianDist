"""run.py:"""
#!/usr/bin/env python

from __future__ import print_function

import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.distributed as dist
from torch.multiprocessing import Process


from utils import *
from density_plot import get_esd_plot
from models.resnet import resnet

from pyhessian.utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal
from typing import List, Callable


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



# Settings
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--mini-hessian-batch-size', type=int, default=200,
                    help='input batch size for mini-hessian batch (default: 200)')
parser.add_argument('--hessian-batch-size', type=int, default=200, help='input batch size for hessian (default: 200)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--batch-norm', action='store_false', help='do we need batch norm or not')
parser.add_argument('--residual', action='store_false', help='do we need residual connect or not')
parser.add_argument('--cuda', action='store_false', help='do we use gpu or not')
parser.add_argument('--resume', type=str, default='', help='get the checkpoint')

# eigen info
parser.add_argument('--eigenvalue', dest='eigenvalue', action='store_true')  # default false
parser.add_argument('--trace', dest='trace', action='store_true')  # default false
parser.add_argument('--density', dest='density', action='store_true')  # default false
parser.set_defaults(eigenvalue=False)
parser.set_defaults(trace=False)
parser.set_defaults(density=False)

# for parallel computing
parser.add_argument('--ip', type=str, required=True, help='ip address of the machine for distributed computing')
parser.add_argument('--device_count', type=int, required=True, help='number of available devices')

args = parser.parse_args()

# set random seed to reproduce the work
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

for arg in vars(args):
    print(arg, getattr(args, arg))

# # get dataset
train_loader, test_loader = getData(name='cifar10_without_data_augmentation',
                                    train_bs=args.mini_hessian_batch_size,
                                    test_bs=1)
##############
# Get the hessian data
##############
assert (args.hessian_batch_size % args.mini_hessian_batch_size == 0)
batch_num = args.hessian_batch_size // args.mini_hessian_batch_size
# getting the dataset batches
hessian_dataloader = []
for i, (inputs, labels) in enumerate(train_loader):
    hessian_dataloader.append((inputs, labels))
    if i == batch_num - 1:
        break

# dividing dataset into partitions
assert len(hessian_dataloader) % args.device_count == 0
# partitioning data into number of GPUs available
data_partitions = DataPartitioner(hessian_dataloader,
                                  [len(hessian_dataloader) // args.device_count] * args.device_count)

# get model
model = resnet(num_classes=10,
               depth=20,
               residual_not=args.residual,
               batch_norm_not=args.batch_norm)

# label loss
criterion = nn.CrossEntropyLoss()

###################
# Get model checkpoint, get saving folder
###################
if args.resume == '':
    raise Exception("please choose the trained model")
# loading the state dictionary into the model
state_dict = torch.load(args.resume, map_location=torch.device('cpu'))
# since model was trained using DataParallel, have to remove 'module.'
# from state dictionary keys
state_dict_ = OrderedDict()
for key in state_dict.keys():
    new_key = key[7:]
    state_dict_[new_key] = state_dict[key]
model.load_state_dict(state_dict_)


def hv_product(rank: int, size: int, model: nn.Module, data: List[torch.Tensor], v: List[torch.Tensor]):
    # group of processes
    group = dist.new_group(list(range(size)))
    # current process device
    device = torch.device("cuda:{}".format(rank))

    # count the number of datum points in the dataloader
    num_data = 0
    # to accumulate the hessian vector product
    THv = []

    for inputs, targets in data:
        model.zero_grad()
        tmp_num_data = inputs.size(0)
        outputs = model(inputs.to(device))
        loss = criterion(outputs, targets.to(device))
        loss.backward(create_graph=True)
        params, gradsH = get_params_grad(model)
        model.zero_grad()
        Hv = torch.autograd.grad(gradsH,
                                 params,
                                 grad_outputs=v,
                                 only_inputs=True,
                                 retain_graph=False)
        # accumulating the eigenvector
        if len(THv) == 0:
            THv = [
                Hv1.contiguous() * float(tmp_num_data) + 0.
                for Hv1 in Hv
            ]
        else:
            THv = [
                THv1 + Hv1 * float(tmp_num_data) + 0.
                for THv1, Hv1 in zip(THv, Hv)
            ]
        num_data += float(tmp_num_data)

    # reducing the THv after every i
    for t in range(len(THv)):
        dist.all_reduce(THv[t], op=dist.ReduceOp.SUM, group=group)

    THv = [THv1 / float(size * num_data) for THv1 in THv]
    eigenvalue = group_product(THv, v).cpu().item()
    return eigenvalue, THv


def eigenvalue(rank: int, size: int, model: nn.Module, data: List[torch.Tensor],
        max_iter: int = 100, tol: float = 1e-3, top_n: int = 1):
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
        for t in v:
            dist.broadcast(t, src=0, group=group)

        for i in range(max_iter):
            v = orthnormal(v, eigenvectors)

            tmp_eigenvalue, Hv = hv_product(rank, size, model, data, v)
            v = normalization(Hv)

            if eigenvalue is None:
                eigenvalue = tmp_eigenvalue
            else:
                if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) +
                                                       1e-6) < tol:
                    break
                else:
                    eigenvalue = tmp_eigenvalue

            # TODO remove this later
            if rank == 0:
                print(eigenvalue)

        eigenvalues.append(eigenvalue)
        eigenvectors.append(v)
        computed_dim += 1

    # TODO remove this later
    if rank == 0:
        print(eigenvalues)


def init_process(rank: int, size: int, model: nn.Module, data: List[torch.Tensor],
                 fn: Callable, ip: str, backend: str = 'nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = ip
    os.environ['MASTER_PORT'] = '29512'  # random port for now
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, model, data)


if __name__ == "__main__":
    processes = []
    start = time.time()
    for rank in range(args.device_count):
        p = Process(target=init_process, args=(rank, args.device_count, model, data_partitions.use(rank),
                                               eigenvalue, args.ip))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    end = time.time()
    print("Time to compute top eigenvalue: %f" % (end - start))