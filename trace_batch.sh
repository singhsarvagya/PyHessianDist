#!/bin/bash 

python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.3' --hessian-batch-size 8000 --mini-hessian-batch-size 100 --trace --device_count 8
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.3' --hessian-batch-size 8000 --mini-hessian-batch-size 200 --trace --device_count 8
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.3' --hessian-batch-size 8000 --mini-hessian-batch-size 250 --trace --device_count 8
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.3' --hessian-batch-size 8000 --mini-hessian-batch-size 400 --trace --device_count 8
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.3' --hessian-batch-size 8000 --mini-hessian-batch-size 500 --trace --device_count 8
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.3' --hessian-batch-size 8000 --mini-hessian-batch-size 800 --trace --device_count 8
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.3' --hessian-batch-size 8000 --mini-hessian-batch-size 1000 --trace --device_count 8