#!/bin/bash

validation_num=RBM5v4

mkdir /data6/sobhan/RLLM/results/validation/${validation_num}

eval "$(conda shell.bash hook)"

conda activate rllm
CUDA_VISIBLE_DEVICES=7 python main.py --runmode generate_single --result-eval-dir /data6/sobhan/RLLM/results/validation/${validation_num}


# conda activate /data1/sobhan/deepclip/deep-env
# CUDA_VISIBLE_DEVICES=4 python /data1/sobhan/deepclip/DeepCLIP.py --runmode predict --network_file /data6/sobhan/dataset/DeepCLIP/RBM5_ENCODE-HepG2.pkl --sequences /data6/sobhan/RLLM/results/validation/${validation_num}/rnas.fasta --predict_output_file /data6/sobhan/RLLM/results/validation/${validation_num}/RBM5.json
# CUDA_VISIBLE_DEVICES=4 python /data1/sobhan/deepclip/DeepCLIP.py --runmode predict --network_file /data6/sobhan/dataset/DeepCLIP/RBM5_ENCODE-HepG2.pkl --sequences /data6/sobhan/RLLM/notebooks/sampled_rnas.fasta --predict_output_file /data6/sobhan/RLLM/notebooks/sampled_rnas.json


conda activate rllm
CUDA_VISIBLE_DEVICES=7 python main.py --runmode evaluate_single --input-compare-dir /data6/sobhan/RLLM/results/validation/${validation_num}/RBM5.json --result-eval-dir /data6/sobhan/RLLM/results/validation/${validation_num} --version 1
# CUDA_VISIBLE_DEVICES=7 python main.py --runmode evaluate_single --input-compare-dir /data6/sobhan/RNAGEN/validation/sampled_rnas.json --result-eval-dir /data6/sobhan/RNAGEN/validation --version 1
