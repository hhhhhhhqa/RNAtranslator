#!/bin/bash

validation_num=RBM5_Pool9
protein=rbm5
pool_size=128
max_len=37

mkdir /data6/sobhan/RLLM/results/validation/${validation_num}
eval "$(conda shell.bash hook)"

# conda activate /data6/sobhan/envs/rllm2
# CUDA_VISIBLE_DEVICES=2 python generate.py --runmode create_pool --max_len ${max_len} --pool_size ${pool_size} --proteins ${protein} --eval-dir /data6/sobhan/RLLM/results/validation/${validation_num}

# # conda activate /data1/sobhan/deepclip/deep-env
# # CUDA_VISIBLE_DEVICES=2 python generate.py --runmode filter --max_len ${max_len} --proteins ${protein} --eval-dir /data6/sobhan/RLLM/results/validation/${validation_num}

# conda activate /data6/sobhan/envs/rllm2
# CUDA_VISIBLE_DEVICES=2 python generate.py --runmode aggregate --max_len ${max_len} --proteins ${protein} --eval-dir /data6/sobhan/RLLM/results/validation/${validation_num}

# conda activate /data1/sobhan/deepclip/deep-env
# CUDA_VISIBLE_DEVICES=2 python /data1/sobhan/deepclip/DeepCLIP.py --runmode predict --network_file /data6/sobhan/RLLM_OPT/deepclip_models/RBM5_our.pkl --sequences /data6/sobhan/RLLM/results/validation/${validation_num}/rnas.fasta --predict_output_file /data6/sobhan/RLLM/results/validation/${validation_num}/${protein}.json

conda activate /data6/sobhan/envs/rllm2
CUDA_VISIBLE_DEVICES=2 python main.py --runmode evaluate --proteins ${protein} --eval-dir /data6/sobhan/RLLM/results/validation/${validation_num}


# CUDA_VISIBLE_DEVICES=6 python ./mainhigh.py --runmode predict --network_file ./models/RBM5.pt --sequences /data6/helya/dataset/CLIPdb_cluster/cd_hit_results_RBPs/identity_90/RBM5_rnas_cdhit_90.fa --predict_output_file ./models/RBM5.json
