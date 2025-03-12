#!/bin/bash

validation_num=TEST_RBM5

mkdir /data6/sobhan/RLLM/results/validation/${validation_num}
eval "$(conda shell.bash hook)"

conda activate rllm
CUDA_VISIBLE_DEVICES=2 python main.py --proteins rbm5 --runmode evaluate --eval-dir /data6/sobhan/RLLM/results/validation/${validation_num}