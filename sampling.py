import argparse
import os
from datetime import datetime
import pytz
import json
from collections import Counter
import sys

import torch.nn.functional as F
import torch
from transformers import T5ForConditionalGeneration

from src.utils.helpers import * 
from src.utils.tokenizer import *

MAX_LEN = 32

def parse_opt():
    ################################################################ Arguments

    parser = argparse.ArgumentParser(description='Sampling Hyperparameters')

    # Training Configuration
    parser.add_argument('--train-hyp', default="/data6/sobhan/RLLM/hyps/train.yaml", type=str, help='Training hyperparameters')
    parser.add_argument('--model-hyp', default="/data6/sobhan/RLLM/hyps/t5.yaml", type=str, help='Model hyperparameters')

    # Generation Configurations
    parser.add_argument('--checkpoints', default='/data6/sobhan/rllm/results/train/t5/run3_20240822-152114/checkpoints/checkpoint-349800', type=str, help='Load Model')
    parser.add_argument('--eval-dir', default="/data6/sobhan/RLLM/results/validation/sampling", type=str, help='Output dir of the evaluation')
    parser.add_argument('--proteins', nargs='+', default=['hnrpnc', 'ago2', 'elavl1', 'rbm5', ] ,type=str, help='List of protein names or IDs')
    parser.add_argument('--rna_num', default=128, type=int, help='Number of RNAs to generate')
    parser.add_argument('--results-dir', default='./results', type=str, metavar='PATH', help='Path to cache (default: none)')

    # args = parser.parse_args()  # For command line execution
    args = parser.parse_args()    # For running in IPython or Jupyter

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Get the number of GPUs and print info
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} ID: cuda:{i}")
    return args

def gen_rna_batch(model, prot_ids, dec_tok, num_candidates, tolerance=5, max_token=MAX_LEN, 
                  strategy='beam_search', temperature=1.0, num_beams=5, top_k=None, top_p=None):
    """
    Generate a batch of candidate RNA sequences using a given sampling strategy and hyperparameters.
    """
    inputs = torch.tensor(prot_ids, dtype=torch.long).unsqueeze(0).to(model.device)
    
    candidate_rnas = []
    while len(candidate_rnas) < num_candidates:
        with torch.no_grad():
            gen_args = {
                'max_length': 15,
                'repetition_penalty': 1.5,
                'encoder_repetition_penalty': 1.3,
                'num_return_sequences': num_candidates,
            }
            if strategy == 'beam_search':
                # For beam search, ensure that num_return_sequences <= num_beams.
                effective_beams = max(num_beams, num_candidates)
                gen_args.update({
                    'do_sample': False,
                    'num_beams': effective_beams,
                    'num_return_sequences': effective_beams,
                })
            elif strategy == 'top_k':
                gen_args.update({
                    'do_sample': True,
                    'temperature': temperature,
                    'top_k': top_k if top_k is not None else 50,
                    'num_beams': num_beams,
                })
            elif strategy == 'top_p':
                gen_args.update({
                    'do_sample': True,
                    'temperature': temperature,
                    'top_p': top_p if top_p is not None else 0.92,
                    'num_beams': num_beams,
                })
            else:  # Simple sampling
                gen_args.update({
                    'do_sample': True,
                    'temperature': temperature,
                    'num_beams': num_beams,
                })
                
            seqs = model.generate(inputs, **gen_args)
            
        decoded_rnas = [
            postprocess_rna(dec_tok.decode(seq.cpu().numpy().tolist()))
            for seq in seqs
        ]
        new_candidates = [
            rna for rna in decoded_rnas
            if (max_token - tolerance) <= len(rna) <= (max_token + tolerance)
        ]
        candidate_rnas.extend(new_candidates)
        candidate_rnas = candidate_rnas[:num_candidates]
    return candidate_rnas

def grid_search_generation(args, model, source_tokenizer, rna_tokenizer):
    grid_config = {
        'beam_search': [
            {'num_beams': 1},
            {'num_beams': 5},
            {'num_beams': 25}
        ],
        'top_k': [
            {'top_k': 30, 'temperature': 0.7, 'num_beams': 1},
            {'top_k': 30, 'temperature': 1.0, 'num_beams': 1},
            {'top_k': 30, 'temperature': 1.5, 'num_beams': 1},
            {'top_k': 100, 'temperature': 0.7, 'num_beams': 1},
            {'top_k': 100, 'temperature': 1.0, 'num_beams': 1},
            {'top_k': 100, 'temperature': 1.5, 'num_beams': 1},
        ],
        'top_p': [
            {'top_p': 0.7, 'temperature': 0.7, 'num_beams': 1},
            {'top_p': 0.7, 'temperature': 1.0, 'num_beams': 1},
            {'top_p': 0.7, 'temperature': 1.5, 'num_beams': 1},
            {'top_p': 0.9, 'temperature': 0.7, 'num_beams': 1},
            {'top_p': 0.9, 'temperature': 1.0, 'num_beams': 1},
            {'top_p': 0.9, 'temperature': 1.5, 'num_beams': 1},
        ],
        'sample': [
            {'temperature': 0.7, 'num_beams': 1},
            {'temperature': 1.0, 'num_beams': 1},
            {'temperature': 1.5, 'num_beams': 1},
        ]
    }
    
    os.makedirs(args.eval_dir, exist_ok=True)
    
    for protein_name in args.proteins:
        protein_seq = read_protein_from_csv(protein_name, file_path="/data6/sobhan/dataset/proteins/protein_seqs.csv")
        if protein_seq is None:
            print(f"Warning: Protein {protein_name} not found.")
            continue
        print("Processing Protein:", protein_name)
        prot_ids = source_tokenizer.tokenize(protein_seq).ids
        
        for strategy, hyper_list in grid_config.items():
            for hyperparams in hyper_list:
                temperature = hyperparams.get('temperature', 1.0)
                num_beams = hyperparams.get('num_beams', 1)
                top_k = hyperparams.get('top_k', None)
                top_p = hyperparams.get('top_p', None)
                
                print(f"Generating for Protein: {protein_name}, Strategy: {strategy}, Hyperparameters: {hyperparams}")
                candidate_rnas = gen_rna_batch(
                    model,
                    prot_ids,
                    rna_tokenizer,
                    args.rna_num,
                    strategy=strategy,
                    temperature=temperature,
                    num_beams=num_beams,
                    top_k=top_k,
                    top_p=top_p
                )
                param_str = "_".join([f"{k}_{v}" for k, v in hyperparams.items()])
                output_filename = f"{protein_name}.fasta"
                output_path = os.path.join(args.eval_dir, output_filename)
                with open(output_path, "a") as f:
                    for idx, rna in enumerate(candidate_rnas):
                        f.write(f">RNA_{idx}_{strategy}_{param_str}\n{rna}\n")
                print(f"Saved results to {output_path}")

if __name__ == '__main__':
    args = parse_opt()
    args = set_hyps(args.train_hyp, args)
    args = set_hyps(args.model_hyp, args)

    # Load the pretrained model.
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoints).to(args.device)
    model.eval()

    source_tokenizer = get_tokenizer(
        tokenizer_name=args.tokenizer,
        vocab_size=args.vocab_size,
        seq_size=args.seq_size,
        tokenizer_path=args.source_tokenizer
    )
    rna_tokenizer = get_tokenizer(
        tokenizer_name=args.tokenizer,
        vocab_size=args.vocab_size,
        seq_size=args.seq_size,
        tokenizer_path=args.rna_tokenizer
    )
    grid_search_generation(args, model, source_tokenizer, rna_tokenizer)