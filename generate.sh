#!/bin/bash

validation_num=ELAVL1

mkdir /data6/sobhan/RLLM/results/validation/${validation_num}
eval "$(conda shell.bash hook)"

conda activate rllm
CUDA_VISIBLE_DEVICES=2 python main.py --runmode generate --proteins elavl1 --eval-dir /data6/sobhan/RLLM/results/validation/${validation_num}