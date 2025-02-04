#!/bin/bash

validation_num=ELAVL1
protein=elavl1

mkdir /data6/sobhan/RLLM/results/validation/${validation_num}
eval "$(conda shell.bash hook)"

# conda activate rllm
# CUDA_VISIBLE_DEVICES=2 python generate.py --runmode create_pool --proteins ${protein} --eval-dir /data6/sobhan/RLLM/results/validation/${validation_num}

conda activate /data1/sobhan/deepclip/deep-env
CUDA_VISIBLE_DEVICES=2 python generate.py --runmode filter --proteins ${protein} --eval-dir /data6/sobhan/RLLM/results/validation/${validation_num}