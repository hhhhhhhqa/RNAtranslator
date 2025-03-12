#!/bin/bash

protein=RBM5
base=/data6/sobhan/RLLM/dataset/deepclip

sequences=${base}/${protein}/test.fasta

epochs=50
early_stopping=5

mkdir ${base}/${protein}
eval "$(conda shell.bash hook)"

conda activate /data6/sobhan/envs/rllm
# CUDA_VISIBLE_DEVICES=6 python ./deepclip.py --runmode predict -n ${base}/${protein}/model.pt --sequences ${sequences} --predict_output_file ${base}/${protein}/${protein}.json

CUDA_VISIBLE_DEVICES=6 python ./deepclip.py --runmode test -n ${base}/${protein}/model.pt --test_positive ${base}/${protein}/data/test_positives.fasta --test_negative ${base}/${protein}/data/test_negatives.fasta --test_output_file ${base}/${protein}/test.json --plot_file ${base}/${protein}/test.png --predict_output_file ${base}/${protein}/${protein}.json
