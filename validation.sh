!/bin/bash

# validation_num=ELAVL1_32_BEAM
# protein=elavl1

# mkdir /data6/sobhan/RLLM/results/validation/${validation_num}

eval "$(conda shell.bash hook)"

# conda activate rllm
# CUDA_VISIBLE_DEVICES=2 python main.py --runmode generate --proteins ${protein} --eval-dir /data6/sobhan/RLLM/results/validation/${validation_num}


# conda activate /data1/sobhan/deepclip/deep-env
# CUDA_VISIBLE_DEVICES=2 python /data1/sobhan/deepclip/DeepCLIP.py --runmode predict --network_file /data6/sobhan/dataset/DeepCLIP/models/graphprot/ELAVL1.pkl --sequences /data6/sobhan/RLLM/results/validation/${validation_num}/rnas.fasta --predict_output_file /data6/sobhan/RLLM/results/validation/${validation_num}/${protein}.json


# conda activate rllm
# CUDA_VISIBLE_DEVICES=2 python main.py --runmode evaluate --proteins ${protein} --eval-dir /data6/sobhan/RLLM/results/validation/${validation_num}


# TEMP
conda activate /data1/sobhan/deepclip/deep-env
CUDA_VISIBLE_DEVICES=2 python /data1/sobhan/deepclip/DeepCLIP.py --runmode predict --network_file /data6/sobhan/dataset/DeepCLIP/models/ENCODE/ENCODE-HepG2/RBM5_ENCODE-HepG2.pkl --sequences /data6/sobhan/RLLM/results/validation/sampling/rbm5.fasta --predict_output_file /data6/sobhan/RLLM/results/validation/sampling/rbm5.json


# PROTEIN_DIR="/data6/sobhan/dataset/DeepCLIP/models/graphprot"  
# RESULTS_DIR="/data6/sobhan/RLLM/results/validation/proteins"
# GPU_NUM=3

# for file in ${PROTEIN_DIR}/*.pkl; do
#     protein=$(basename "$file" .pkl)
#     validation_num=${RESULTS_DIR}/${protein}

    
#     mkdir -p ${validation_num}

    
#     eval "$(conda shell.bash hook)"
#     conda activate rllm
#     CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py --runmode generate --proteins ${protein} --eval-dir ${validation_num}

    
#     conda activate /data1/sobhan/deepclip/deep-env
#     CUDA_VISIBLE_DEVICES=${GPU_NUM} python /data1/sobhan/deepclip/DeepCLIP.py --runmode predict --network_file ${file} --sequences ${validation_num}/rnas.fasta --predict_output_file ${validation_num}/${protein}.json

    
#     conda activate rllm
#     CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py --runmode evaluate --proteins ${protein} --eval-dir ${validation_num}
# done