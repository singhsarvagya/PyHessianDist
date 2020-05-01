#!/bin/bash 

python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.3' --hessian-batch-size 1600 --eigenvalue --device_count 8
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.3' --hessian-batch-size 3200 --eigenvalue --device_count 8
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.3' --hessian-batch-size 4800 --eigenvalue --device_count 8
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.3' --hessian-batch-size 6400 --eigenvalue --device_count 8
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.3' --hessian-batch-size 8000 --eigenvalue --device_count 8
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.3' --hessian-batch-size 9600 --eigenvalue --device_count 8
