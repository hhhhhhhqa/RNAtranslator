#!/bin/bash

validation_num=NoSeen
protein=4g82
pool_size=128
max_len=32

mkdir /data6/sobhan/RLLM/results/validation/${validation_num}
eval "$(conda shell.bash hook)"

conda activate /data6/sobhan/envs/rllm2
CUDA_VISIBLE_DEVICES=7 python generate.py --runmode create_pool --max_len ${max_len} --rna_num ${pool_size} --proteins ${protein} --eval-dir /data6/sobhan/RLLM/results/validation/${validation_num}

conda activate /data1/sobhan/deepclip/deep-env
CUDA_VISIBLE_DEVICES=7 python generate.py --ignore_clip true --runmode filter --max_len ${max_len} --proteins ${protein} --eval-dir /data6/sobhan/RLLM/results/validation/${validation_num}

conda activate /data6/sobhan/envs/rllm2
CUDA_VISIBLE_DEVICES=7 python generate.py --runmode aggregate --max_len ${max_len} --proteins ${protein} --eval-dir /data6/sobhan/RLLM/results/validation/${validation_num}

conda activate /data1/sobhan/deepclip/deep-env
CUDA_VISIBLE_DEVICES=2 python /data1/sobhan/deepclip/DeepCLIP.py --runmode predict --network_file /data6/sobhan/RLLM_OPT/deepclip_models/RBM5_ENCODE-HepG2.pkl --sequences /data6/sobhan/RLLM/results/validation/${validation_num}/rnas.fasta --predict_output_file /data6/sobhan/RLLM/results/validation/${validation_num}/${protein}.json

conda activate /data6/sobhan/envs/rllm2
CUDA_VISIBLE_DEVICES=7 python main.py --runmode evaluate --proteins ${protein} --eval-dir /data6/sobhan/RLLM/results/validation/${validation_num}

