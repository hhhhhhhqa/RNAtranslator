#!/bin/bash

validation_num=MOV10
protein=mov10

mkdir /data6/sobhan/RLLM/results/validation/${validation_num}

eval "$(conda shell.bash hook)"

conda activate rllm
CUDA_VISIBLE_DEVICES=2 python main.py --runmode generate --proteins ${protein} --eval-dir /data6/sobhan/RLLM/results/validation/${validation_num}


conda activate /data1/sobhan/deepclip/deep-env
CUDA_VISIBLE_DEVICES=2 python /data1/sobhan/deepclip/DeepCLIP.py --runmode predict --network_file /data6/sobhan/dataset/DeepCLIP/RBM5_ENCODE-HepG2.pkl --sequences /data6/sobhan/RLLM/results/validation/${validation_num}/rnas.fasta --predict_output_file /data6/sobhan/RLLM/results/validation/${validation_num}/${protein}.json


conda activate rllm
CUDA_VISIBLE_DEVICES=2 python main.py --runmode evaluate --proteins ${protein} --eval-dir /data6/sobhan/RLLM/results/validation/${validation_num}