#!/bin/bash

protein=U2AF2
base=/data6/sobhan/RLLM/dataset/deepclip

sequences=${base}/${protein}/data/train_positives.fasta
background_sequences=${base}/${protein}/data/train_negatives.fasta
epochs=100
early_stopping=10

mkdir ${base}/${protein}
eval "$(conda shell.bash hook)"

conda activate /data6/sobhan/envs/rllm
CUDA_VISIBLE_DEVICES=7 nohup python ./deepclip.py --runmode train --early_stopping ${early_stopping} -e ${epochs} -n ${base}/${protein}/model.pt --sequences ${sequences} --background_sequences ${background_sequences} --force_max_length 75 > ${base}/${protein}/log.txt &
# CUDA_VISIBLE_DEVICES=5 python ./deepclip.py --runmode train --early_stopping ${early_stopping} -e ${epochs} -n ${base}/${protein}/model.pt --sequences ${sequences} --background_sequences ${background_sequences} --force_max_length 75 
