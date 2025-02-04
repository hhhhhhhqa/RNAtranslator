import argparse
import os
import wandb
from datetime import datetime
import pytz
import time
import json

from src.models import get_model
from src.utils.helpers import set_hyps
from src.utils.tokenizer import get_tokenizer, BpeTokenizer
from src.data import get_datasets
from train import train
from generate import generate
from eval import evaluate


from datasets import load_dataset

from accelerate import Accelerator

import torch

def parse_opt():
    ################################################################ Arguments

    parser = argparse.ArgumentParser(description='Multilingual RNA Implementation')

    # action
    parser.add_argument('--runmode', default="generate", choices=["train", "evaluate", "generate"], type=str, help='Runmode')


    # Trainig Configuration
    parser.add_argument('--train-hyp', default="/data6/sobhan/RLLM/hyps/train.yaml", type=str, help='Training Arguments hyperprameters')
    parser.add_argument('--model-hyp', default="/data6/sobhan/RLLM/hyps/t5.yaml", type=str, help='Model hyperprameters')

    # Generation Configurations
    parser.add_argument('--checkpoints', default='/data6/sobhan/rllm/results/train/t5/run3_20240822-152114/checkpoints/checkpoint-349800', type=str, help='Load Model')

    parser.add_argument('--eval-dir', default="/data6/sobhan/RLLM/results/validation", type=str, help='Output dir of the evaluation')

    parser.add_argument('--proteins', nargs='+', type=str, help='List of protein sequences')
    parser.add_argument('--rna_num', default=128, type=int, help='Number of RNAs to generate')

    parser.add_argument('--results-dir', default='./results', type=str, metavar='PATH', help='path to cache (default: none)')

    # args = parser.parse_args()  # running in command line
    args = parser.parse_args()  # running in ipynb

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Get the number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Print device IDs for each GPU
    for i in range(num_gpus):
        print(f"GPU {i} is: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} ID: cuda:{i}")

    accelerator = Accelerator()
    device = accelerator.device
    print(f"Using device: {device}")

    args.device = str(device)
    print(args.device)
    return args


def main(args:object, wandb)->None:
    # track total training time
    start_time = datetime.now(pytz.timezone('Turkey')).strftime("%Y-%m-%d %H:%M")
    args.start_time = start_time
    if args.runmode == "train":
        print("============================================================================================")
        print("Started training at : ", start_time)
        print("============================================================================================")
    else:
        print("============================================================================================")
        print("Started generating at : ", start_time)
        print("============================================================================================")


    ############################################Results DIR########################################### 

    # Handle Training Arguments
    args = set_hyps(args.train_hyp, args)
    args = set_hyps(args.model_hyp, args)
    if args.runmode == "train": args.results_dir = os.path.join(args.results_dir, "train")
    else: args.results_dir = os.path.join(args.results_dir, "inference")
    args.results_dir = os.path.join(args.results_dir, args.model)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)    
    args.results_dir = os.path.join(args.results_dir, "run"+str(len(os.listdir(args.results_dir)))+"_"+time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(args.results_dir)

    # Resume Training
    # args.results_dir = "/data6/sobhan/rllm/results/train/t5/run3_20240822-152114"
    

    ############################################Load the main components########################################### 

    # Load Tokenizer
    source_tokenizer = get_tokenizer(tokenizer_name=args.tokenizer, vocab_size=args.vocab_size, seq_size=args.seq_size, tokenizer_path=args.source_tokenizer)
    rna_tokenizer = get_tokenizer(tokenizer_name=args.tokenizer, vocab_size=args.vocab_size, seq_size=args.seq_size, tokenizer_path=args.rna_tokenizer)

    # Train Tokenizer
    if args.train_tokenizer:
        from datasets import load_dataset
        dataset = load_dataset("text", data_files=args.train_data, split="train", cache_dir="/data6/sobhan/rllm/dataset/rph/cache")
        source_tokenizer.train_tokenizer(train_data=dataset)
        source_tokenizer.save("/data6/sobhan/rllm/dataset/tokenizers/proteins", "bpe_protein_{}_{}".format(1000, 1024))
        rna_tokenizer.train_tokenizer(train_data=dataset, which=False)
        rna_tokenizer.save("/data6/sobhan/rllm/dataset/tokenizers/rnas", "bpe_rna_{}_{}".format(1000, 1024))


    if args.runmode == "train":
        train_dataset, eval_dataset = get_datasets(args, source_tokenizer=source_tokenizer, rna_tokenizer=rna_tokenizer)
    model = get_model(args=args)

    args.model_size = sum(p.numel() for p in model.parameters())
    print("Model Size: ", sum(p.numel() for p in model.parameters()))
    # print(model)

    # Saving the configs
    args_dict = vars(args)
    with open(args.results_dir + '/Main Config.json', 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)
    print("Config saved to ", args.results_dir)


    print(args.runmode)
    if args.runmode == "train":
        train(args=args, wandb=wandb, model=model, train_dataset=train_dataset, eval_dataset=eval_dataset)
    elif args.runmode == "evaluate":
        evaluate(args.eval_dir, protein=args.proteins[0])
    elif args.runmode == "generate":
        generate(args=args, model=model, enc_tokenizer=source_tokenizer, dec_tokenizer=rna_tokenizer, result_dir=args.eval_dir)
    else:
        raise ValueError("Invalid runmode")
        

    end_time = datetime.now(pytz.timezone('Turkey')).strftime("%Y-%m-%d %H:%M")
    if args.runmode == "train":
        print("============================================================================================")
        print("Finished training at : ", end_time)
        print("============================================================================================")
    else:
        print("============================================================================================")
        print("Finished generating at : ", end_time)
        print("============================================================================================")


if __name__=="__main__":
    os.environ['WANDB_DISABLED'] = 'true'
    # Log in to your W&B account
    # wandb.login()
    # wandb.init(project="RNA-LLM")
    # os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    # os.environ["WANDB_WATCH"] = "all"

    args=parse_opt()
    main(args ,wandb)
    wandb.finish()


