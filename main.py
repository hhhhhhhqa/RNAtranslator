#!/usr/bin/env python
import argparse
import os
import time
import json
from datetime import datetime
import pytz
import torch
from accelerate import Accelerator
import wandb

from src.models import get_model
from src.utils.helpers import set_hyps
from src.utils.tokenizer import get_tokenizer
from src.data import get_datasets
from train import train
from generate import generate
from evaluate import evaluate

def parse_opt():
    parser = argparse.ArgumentParser(description='RNA-LLM Implementation')
    parser.add_argument('--runmode', default="generate", choices=["train", "generate", "evaluate"], type=str, help='Run mode')
    
    parser.add_argument('--train-hyp', default="./hyps/train.yaml", type=str, help='Training hyperparameters YAML')
    parser.add_argument('--model-hyp', default="./hyps/t5.yaml", type=str, help='Model hyperparameters YAML')
    
    # Generation configuration
    parser.add_argument('--checkpoints', default="/data6/sobhan/RLLM/finetune/checkpoint-423800", type=str, help='Path to model checkpoint')
    parser.add_argument('--protein-fasta', default=None, type=str, help='Protein fasta file for generation')
    parser.add_argument('--protein-seq', default=None, type=str, help='Protein sequence (if not using fasta)')
    parser.add_argument('--rna_num', type=int, default=500, help='Number of RNAs to generate')
    parser.add_argument('--max_len', type=int, default=75, help='Maximum length for RNA generation')
    
    parser.add_argument('--eval-dir', default="results/validation", type=str, help='Directory for evaluation outputs')
    parser.add_argument('--deepclip', default=False, type=str, help='Analyze the DeepCLIP output')
    parser.add_argument('--rnas_fasta', default=None, type=str, help='RNA fasta file for evaluation')

    
    parser.add_argument('--results-dir', default='./results', type=str, help='Results directory')
    
    args = parser.parse_args()
    
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    accelerator = Accelerator()
    args.device = str(accelerator.device)
    return args

def main():
    os.environ['WANDB_DISABLED'] = 'true'
    args = parse_opt()
    
    args = set_hyps(args.train_hyp, args)
    args = set_hyps(args.model_hyp, args)
    
    if args.runmode == "train":
        mode_folder = "train"
    elif args.runmode == "generate":
        mode_folder = "inference"
    else:
        mode_folder = "evaluation"
        
    model_name = getattr(args, "model", "default")
    args.results_dir = os.path.join(args.results_dir, mode_folder, model_name)
    os.makedirs(args.results_dir, exist_ok=True)
    args.results_dir = os.path.join(args.results_dir, "run" + str(len(os.listdir(os.path.join(args.results_dir)))) + "_" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(args.results_dir, exist_ok=True)
    
    start_time = datetime.now(pytz.timezone('UTC')).strftime("%Y-%m-%d %H:%M")
    print("Started", args.runmode, "at:", start_time)
    
    if args.runmode == "train":
        source_tokenizer = get_tokenizer(tokenizer_name=args.tokenizer, vocab_size=args.vocab_size,
                                         seq_size=args.seq_size, tokenizer_path=args.source_tokenizer)
        rna_tokenizer = get_tokenizer(tokenizer_name=args.tokenizer, vocab_size=args.vocab_size,
                                      seq_size=args.seq_size, tokenizer_path=args.rna_tokenizer)
        train_dataset, eval_dataset = get_datasets(args, source_tokenizer=source_tokenizer, rna_tokenizer=rna_tokenizer)
        model = get_model(args=args)
        print("Model Size:", sum(p.numel() for p in model.parameters()))
        with open(os.path.join(args.results_dir, 'Main_Config.json'), 'w') as json_file:
            json.dump(vars(args), json_file, indent=4)
        train(args, wandb, model, train_dataset, eval_dataset)
        
    elif args.runmode == "generate":
        generate(args)
        
    elif args.runmode == "evaluate":
        evaluate(args)
        
    end_time = datetime.now(pytz.timezone('UTC')).strftime("%Y-%m-%d %H:%M")
    print("Finished", args.runmode, "at:", end_time)
    wandb.finish()

if __name__=="__main__":
    main()
