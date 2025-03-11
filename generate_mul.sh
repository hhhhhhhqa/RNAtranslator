#!/bin/bash

# Parameters
pool_size=128
max_len=37
deepclip_model_dir="/data6/sobhan/RLLM_OPT/deepclip_models"
results_dir="/data6/sobhan/RLLM/results/validation/test374800-filtered"
# Define an array of proteins (adjust names to match your convention)
proteins=("RBM5" "FUS" "ZC3H7B" "HNRNPA1" "MOV10" "AGO2" "TARDBP" "ELAVL1" "U2AF2" "hnRPNC" "SRSF1")

# Initialize conda for this shell session
eval "$(conda shell.bash hook)"

# Loop over each protein
for protein in "${proteins[@]}"; do
    # Create a unique validation ID, for example prefixing with 'PDB'
    validation_num="PDB${protein}"
    
    # Create a directory to store the results for this protein
    mkdir -p "${results_dir}/${validation_num}"
    
    echo "Processing protein: ${protein}"
    echo "Results will be saved in: ${results_dir}/${validation_num}"
    
    # 1. Create RNA pool using the RLLM environment
    conda activate /data6/sobhan/envs/rllm2
    CUDA_VISIBLE_DEVICES=2 python generate.py \
        --runmode create_pool \
        --max_len ${max_len} \
        --rna_num ${pool_size} \
        --proteins ${protein} \
        --eval-dir "${results_dir}/${validation_num}"
    
    # 2. Filter using DeepCLIP's environment
    conda activate /data1/sobhan/deepclip/deep-env
    CUDA_VISIBLE_DEVICES=2 python generate.py \
        --runmode filter \
        --max_len ${max_len} \
        --proteins ${protein} \
        --eval-dir "${results_dir}/${validation_num}"
    
    # 3. Aggregate pool (back in the RLLM environment)
    conda activate /data6/sobhan/envs/rllm2
    CUDA_VISIBLE_DEVICES=2 python generate.py \
        --runmode aggregate \
        --max_len ${max_len} \
        --proteins ${protein} \
        --eval-dir "${results_dir}/${validation_num}"
    
    # 4. Automatically find the DeepCLIP model file corresponding to this protein
    deepclip_model_file=$(find "${deepclip_model_dir}" -type f -name "*${protein}*.pkl" | head -n 1)
    if [ -z "$deepclip_model_file" ]; then
        echo "DeepCLIP model not found for protein ${protein}. Skipping prediction..."
        continue
    else
        echo "Using DeepCLIP model: ${deepclip_model_file}"
    fi
    
    # 5. Run DeepCLIP prediction in its environment using the found model file
    conda activate /data1/sobhan/deepclip/deep-env
    CUDA_VISIBLE_DEVICES=2 python /data1/sobhan/deepclip/DeepCLIP.py \
        --runmode predict \
        --network_file "${deepclip_model_file}" \
        --sequences "${results_dir}/${validation_num}/rnas.fasta" \
        --predict_output_file "${results_dir}/${validation_num}/${protein}.json"
    
    # 6. Evaluate the results (back in the RLLM environment)
    conda activate /data6/sobhan/envs/rllm2
    CUDA_VISIBLE_DEVICES=2 python main.py \
        --runmode evaluate \
        --proteins ${protein} \
        --eval-dir "${results_dir}/${validation_num}"
    
    echo "Finished processing ${protein}"
done
