#!/bin/bash 

python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.3' --hessian-batch-size 8000 --trace --device_count 1
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.3' --hessian-batch-size 8000 --trace --device_count 2
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.3' --hessian-batch-size 8000 --trace --device_count 3
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.3' --hessian-batch-size 8000 --trace --device_count 4
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.3' --hessian-batch-size 8000 --trace --device_count 5
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.3' --hessian-batch-size 8000 --trace --device_count 6
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.3' --hessian-batch-size 8000 --trace --device_count 7
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.3' --hessian-batch-size 8000 --trace --device_count 8