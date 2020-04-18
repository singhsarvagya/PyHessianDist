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
# from pyhessian import hessian
# from multiprocessing import set_start_method
# try:
#     set_start_method('spawn')
# except RuntimeError:
#     pass



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

args = parser.parse_args()

# set random seed to reproduce the work
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

for arg in vars(args):
    print(arg, getattr(args, arg))

# total available devices
device_count = 2

# # get dataset
train_loader, test_loader = getData(name='cifar10_without_data_augmentation',
                                    train_bs=args.mini_hessian_batch_size,
                                    test_bs=1)
##############
# Get the hessian data
##############
assert (args.hessian_batch_size % args.mini_hessian_batch_size == 0)
batch_num = args.hessian_batch_size // args.mini_hessian_batch_size
#
# # getting the dataset batches
hessian_dataloader = []
for i, (inputs, labels) in enumerate(train_loader):
    hessian_dataloader.append((inputs, labels))
    if i == batch_num - 1:
        break

# dividing dataset into partitions
print(len(hessian_dataloader))
assert len(hessian_dataloader) % device_count == 0
# partitioning data into number of GPUs available
data_partitions = DataPartitioner(hessian_dataloader, [len(hessian_dataloader) // device_count] * device_count)


# get model
model = resnet(num_classes=10,
               depth=20,
               residual_not=args.residual,
               batch_norm_not=args.batch_norm)


criterion = nn.CrossEntropyLoss()  # label loss

###################
# Get model checkpoint, get saving folder
###################
if args.resume == '':
    raise Exception("please choose the trained model")

state_dict = torch.load(args.resume, map_location=torch.device('cpu'))
state_dict_ = OrderedDict()
for key in state_dict.keys():
    new_key = key[7:]
    state_dict_[new_key] = state_dict[key]
model.load_state_dict(state_dict_)

# test(model, test_loader, cuda=False)

"""Non-blocking point-to-point communication."""


def dataloader_hv_product(v, model, data, device, size):
    num_data = 0  # count the number of datum points in the dataloader

    THv = []

    for inputs, targets in data:
        # print(inputs.shape)
        # print(inputs[0, 0, 0, 7])
        model.zero_grad()
        tmp_num_data = inputs.size(0)
        outputs = model(inputs.to(device))
        loss = criterion(outputs, targets.to(device))
        # print(loss)
        loss.backward(create_graph=True)
        params, gradsH = get_params_grad(model)
        model.zero_grad()
        # print(gradsH[4][13])
        Hv = torch.autograd.grad(gradsH,
                                 params,
                                 grad_outputs=v,
                                 only_inputs=True,
                                 retain_graph=False)
        # print(Hv[4][13])
        # exit(0)
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

    # TODO reduce Hv
    group = dist.new_group(range(size))
    for t in range(len(THv)):
        dist.all_reduce(THv[t], op=dist.ReduceOp.SUM, group=group)
    # print (THv[0][0, 0])
    # exit(0)
    THv = [THv1 / float(size * num_data) for THv1 in THv]
    eigenvalue = group_product(THv, v).cpu().item()
    return eigenvalue, THv


""" All-Reduce example."""
def run(rank, size, model, data):
    """ Simple point-to-point communication. """
    group = dist.new_group(range(size))
    device = torch.device("cuda:{}".format(rank))
    # tensor = torch.ones(1).to(device)
    # dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    # print('Rank ', rank, ' has data ', tensor[0])

    maxIter = 20
    tol = 1e-3
    top_n = 1

    assert top_n >= 1

    eigenvalues = []
    eigenvectors = []

    computed_dim = 0

    model = model.to(device)
    params, gradsH = get_params_grad(model)
    model.eval()

    if rank == 0:
        start = time.time()

    while computed_dim < top_n:
        eigenvalue = None
        v = [torch.randn(p.size()).to(device) for p in params]  # generate random vector
        v = normalization(v)

        for t in v:
            dist.broadcast(t, src=0, group=group)

        for i in range(maxIter):
            # print (i)
            v = orthnormal(v, eigenvectors)
            model.zero_grad()

            tmp_eigenvalue, Hv = dataloader_hv_product(v, model, data, device, size)
            # print (tmp_eigenvalue)
            # print (Hv[0][0, 0, 0])
            # exit(0)

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
                print (eigenvalue)

        eigenvalues.append(eigenvalue)
        eigenvectors.append(v)
        computed_dim += 1

    if rank == 0:
        print(eigenvalues)
        end = time.time()
        print("Time to compute top eigenvalue: %f" % (end - start))


def init_process(rank, size, model, data, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '128.32.35.224'
    os.environ['MASTER_PORT'] = '29512'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, model, data)


if __name__ == "__main__":
    size = device_count
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, model, data_partitions.use(rank), run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()