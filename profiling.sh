#!/bin/bash 

python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.2' --hessian-batch-size 8000 --eigenvalue --device_count 1 > eigen_1.out 
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.2' --hessian-batch-size 8000 --eigenvalue --device_count 2 > eigen_2.out 
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.2' --hessian-batch-size 8000 --eigenvalue --device_count 3 > eigen_3.out 
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.2' --hessian-batch-size 8000 --eigenvalue --device_count 4 > eigen_4.out 
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.2' --hessian-batch-size 8000 --eigenvalue --device_count 5 > eigen_5.out 
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.2' --hessian-batch-size 8000 --eigenvalue --device_count 6 > eigen_6.out 
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.2' --hessian-batch-size 8000 --eigenvalue --device_count 7 > eigen_7.out 
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.2' --hessian-batch-size 8000 --eigenvalue --device_count 8 > eigen_8.out 

python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.2' --hessian-batch-size 8000 --trace --device_count 1 > trace_1.out
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.2' --hessian-batch-size 8000 --trace --device_count 2 > trace_2.out
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.2' --hessian-batch-size 8000 --trace --device_count 3 > trace_3.out
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.2' --hessian-batch-size 8000 --trace --device_count 4 > trace_4.out
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.2' --hessian-batch-size 8000 --trace --device_count 5 > trace_5.out
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.2' --hessian-batch-size 8000 --trace --device_count 6 > trace_6.out
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.2' --hessian-batch-size 8000 --trace --device_count 7 > trace_7.out
python3 example_pyhessian_analysis.py --resume checkpoints/net.pkl --ip '172.17.0.2' --hessian-batch-size 8000 --trace --device_count 8 > trace_8.out
