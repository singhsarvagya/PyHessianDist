"""run.py:"""
#!/usr/bin/env python

from __future__ import print_function

import time
from collections import OrderedDict
import argparse
from utils import *
from models.resnet import resnet
from pyhessian import *
import torch.nn as nn


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

if __name__ == "__main__":
    if args.eigenvalue:
        start = time.time()
        eigenvalues = eigenvalue(args.device_count, model, data_partitions, criterion, args.ip)
        print(eigenvalues)
        end = time.time()
        print("Time to compute top eigenvalue: %f" % (end - start))
    if args.trace:
        start = time.time()
        trace = trace(args.device_count, model, data_partitions, criterion, args.ip)
        print(trace)
        end = time.time()
        print("Time to compute trace: %f" % (end - start))
    if args.density:
        start = time.time()
        eigen_list, weight_list = density(args.device_count, model, data_partitions, criterion, args.ip)
        print(eigen_list)
        print(weight_list)
        end = time.time()
        print("Time to compute trace: %f" % (end - start))
