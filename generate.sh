#!/bin/bash

validation_num=RBM5

mkdir /data6/sobhan/RLLM/results/validation/${validation_num}
eval "$(conda shell.bash hook)"

conda activate rllm
CUDA_VISIBLE_DEVICES=3 python main.py --runmode generate_single --result-eval-dir /data6/sobhan/RLLM/results/validation/${validation_num}