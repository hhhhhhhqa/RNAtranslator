#!/bin/bash

validation_num=TEST

mkdir /data6/sobhan/RLLM/results/validation/${validation_num}
eval "$(conda shell.bash hook)"

conda activate rllm
CUDA_VISIBLE_DEVICES=2 python main.py --runmode evaluate --result-eval-dir /data6/sobhan/RLLM/results/validation/${validation_num}