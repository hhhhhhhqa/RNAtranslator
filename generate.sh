#!/bin/bash

validation_num=ELAVL1_Pool
protein=elavl1

mkdir /data6/sobhan/RLLM/results/validation/${validation_num}
eval "$(conda shell.bash hook)"

conda activate rllm
CUDA_VISIBLE_DEVICES=2 python generate.py --runmode create_pool --proteins ${protein} --eval-dir /data6/sobhan/RLLM/results/validation/${validation_num}

conda activate /data1/sobhan/deepclip/deep-env
CUDA_VISIBLE_DEVICES=2 python generate.py --runmode filter --proteins ${protein} --eval-dir /data6/sobhan/RLLM/results/validation/${validation_num}

conda activate rllm
CUDA_VISIBLE_DEVICES=2 python generate.py --runmode aggregate --proteins ${protein} --eval-dir /data6/sobhan/RLLM/results/validation/${validation_num}

conda activate /data1/sobhan/deepclip/deep-env
CUDA_VISIBLE_DEVICES=2 python /data1/sobhan/deepclip/DeepCLIP.py --runmode predict --network_file /data6/sobhan/dataset/DeepCLIP/models/graphprot/ELAVL1.pkl --sequences /data6/sobhan/RLLM/results/validation/${validation_num}/rnas.fasta --predict_output_file /data6/sobhan/RLLM/results/validation/${validation_num}/${protein}.json

conda activate rllm
CUDA_VISIBLE_DEVICES=2 python main.py --runmode evaluate --proteins ${protein} --eval-dir /data6/sobhan/RLLM/results/validation/${validation_num}

