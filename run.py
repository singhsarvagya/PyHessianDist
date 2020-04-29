"""run.py:"""
#!/usr/bin/env python

from __future__ import print_function

import os
import time
from collections import OrderedDict

import argparse
import torch.distributed as dist
from torch.multiprocessing import Process, Queue


from utils import *
from density_plot import get_esd_plot
from models.resnet import resnet

from pyhessian.utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal
from typing import List, Callable


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
    temp_hv = []

    for inputs, targets in data:
        model.zero_grad()
        tmp_num_data = inputs.size(0)
        outputs = model(inputs.to(device))
        loss = criterion(outputs, targets.to(device))
        loss.backward(create_graph=True)
        params, gradsH = get_params_grad(model)
        # model.zero_grad()
        Hv = torch.autograd.grad(gradsH,
                                 params,
                                 grad_outputs=v,
                                 only_inputs=True,
                                 retain_graph=False)
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
    for t in range(len(temp_hv)):
        dist.all_reduce(temp_hv[t], op=dist.ReduceOp.SUM, group=group)

    temp_hv = [temp_hv1 / float(size * num_data) for temp_hv1 in temp_hv]
    eigenvalue = group_product(temp_hv, v).cpu().item()
    return eigenvalue, temp_hv


def eigenvalue_(rank: int, size: int, model: nn.Module, data: List[torch.Tensor],
                queue: Queue, max_iter: int = 100, tol: float = 1e-3, top_n: int = 1):
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

        eigenvalues.append(eigenvalue)
        computed_dim += 1

    # communicating eigenvalues and eigenvectors to parent process
    if rank == 0:
        queue.put(eigenvalues)



def eigenvalue(size: int, model: nn.Module, data_partitions: DataPartitioner, ip: str):
    processes = []
    queue = Queue()
    for rank in range(args.device_count):
        p = Process(target=init_process, args=(rank, size, model, data_partitions.use(rank), queue,
                                               eigenvalue_, ip))
        p.start()
        processes.append(p)

    eigenvalues = queue.get()

    for p in processes:
        p.join()

    return eigenvalues


def trace_(rank: int, size: int, model: nn.Module, data: List[torch.Tensor], queue: Queue,
           max_iter: int = 100, tol: float = 1e-3):
    # group of processes
    group = dist.new_group(list(range(size)))
    # current process device
    device = torch.device("cuda:{}".format(rank))

    trace_vhv = []
    trace = 0.

    # moving model to respective GPU
    model = model.to(device)
    # setting model to eval mode
    model.eval()
    # getting model params and gradients
    params, gradsH = get_params_grad(model)

    for i in range(max_iter):
        # generate random vector
        v = [torch.randint_like(p, high=2, device=device) for p in params]

        # generate Rademacher random variables
        for v_i in v:
            v_i[v_i == 0] = -1

        # since we have manually set the seed for cuda
        # same v should be initialized across all GPUs
        # but this step is necessary if the seed is
        # not set
        for t in v:
            dist.broadcast(t, src=0, group=group)

        _, Hv = hv_product(rank, size, model, data, v)

        prod = group_product(Hv, v)
        trace_vhv.append(prod.item())

        if abs(np.mean(trace_vhv) - trace) / (trace + 1e-6) < tol:
            break
        else:
            trace = np.mean(trace_vhv)

    # communicating eigenvalues and eigenvectors to parent process
    if rank == 0:
        queue.put(np.mean(trace_vhv))


def trace(size: int, model: nn.Module, data_partitions: DataPartitioner, ip: str):
    processes = []
    queue = Queue()
    for rank in range(args.device_count):
        p = Process(target=init_process, args=(rank, size, model, data_partitions.use(rank), queue,
                                               trace_, ip))
        p.start()
        processes.append(p)

    trace = queue.get()

    for p in processes:
        p.join()

    return trace


def density_(rank: int, size: int, model: nn.Module, data: List[torch.Tensor], queue: Queue, iter=100, n_v=1):
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

    print("Nv: %d", n_v)
    for k in range(n_v):
        print("k %d", k)
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
            print(i)
            if i == 0:
                _, w_prime = hv_product(rank, size, model, data, v)
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
                _, w_prime = hv_product(rank, size, model, data, v)
                alpha = group_product(w_prime, v)
                alpha_list.append(alpha.cpu().item())
                w_tmp = group_add(w_prime, v, alpha=-alpha)
                w = group_add(w_tmp, v_list[-2], alpha=-beta)
            torch.cuda.synchronize(device)

    if rank == 0:
        T = torch.zeros(iter, iter).to(device).contiguous()
        for i in range(len(alpha_list)):
            print("heren")
            T[i, i] = alpha_list[i]
            if i < len(alpha_list) - 1:
                T[i + 1, i] = beta_list[i]
                T[i, i + 1] = beta_list[i]
        print("here3")
        a_, b_ = torch.eig(T, eigenvectors=True)
        print("here4")
        eigen_list = a_[:, 0]
        weight_list = b_[0, :]**2

        eigen_list_full.append(list(eigen_list.cpu().numpy()))
        weight_list_full.append(list(weight_list.cpu().numpy()))
        print("here5")
        queue.put(eigen_list_full)
        queue.put(weight_list_full)
        # print(eigen_list_full)
        # print(weight_list_full)


def density(size: int, model: nn.Module, data_partitions: DataPartitioner, ip: str):
    processes = []
    queue = Queue()
    for rank in range(args.device_count):
        p = Process(target=init_process, args=(rank, size, model, data_partitions.use(rank), queue,
                                               density_, ip))
        p.start()
        processes.append(p)

    eigen_list_full = queue.get()
    weight_list_full = queue.get()

    for p in processes:
        p.join()

    return eigen_list_full, weight_list_full


def init_process(rank: int, size: int, model: nn.Module, data: List[torch.Tensor], queue: Queue,
                 fn: Callable, ip: str, backend: str = 'gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = ip
    os.environ['MASTER_PORT'] = '29520'  # random port for now
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, model, data, queue)


if __name__ == "__main__":
    # start = time.time()
    # eigenvalues = eigenvalue(args.device_count, model, data_partitions, args.ip)
    # print(eigenvalues)
    # end = time.time()
    # print("Time to compute top eigenvalue: %f" % (end - start))
    # start = time.time()
    # trace = trace(args.device_count, model, data_partitions, args.ip)
    # print (trace)
    # end = time.time()
    # print("Time to compute trace: %f" % (end - start))
    start = time.time()
    eigen_list, weight_list = density(args.device_count, model, data_partitions, args.ip)
    print (eigen_list)
    print (weight_list)
    end = time.time()
    print("Time to compute trace: %f" % (end - start))
