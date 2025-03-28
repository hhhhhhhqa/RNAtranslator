#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main.py --runmode generate --protein-fasta /data6/sobhan/RLLM/examples/protein.fasta --rna_num 128 --max_len 75