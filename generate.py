import argparse
import os
# import wandb
from datetime import datetime
import pytz
import json

from sympy import true
import torch.nn.functional as F


# %load_ext autoreload
# # %autoreload 2
from src.models import get_model
from src.utils.helpers import set_hyps, top_k_top_p_filtering, generate_text
from src.utils.tokenizer import get_tokenizer
from src.data import get_datasets
from src.utils.tokenizer import BpeTokenizer
from src.utils.helpers import generate_text, postprocess_rna
from safetensors.torch import load_file
import accelerate

import torch


def generate(args:object, eval_dataset, model, dec_tokenizer)->None:
    args.model_size = sum(p.numel() for p in model.parameters())
    print("Model Size: ", sum(p.numel() for p in model.parameters()))
    # print(model)

    model = model.from_pretrained(args.checkpoints).to(args.device)
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(args.checkpoints))
    # model.eval()
    # accelerator.prepare_model(model)
    # accelerator.load_state(args.checkpoints)
    
    # state_dict = load_file(args.checkpoints)
    # model.load_state_dict(state_dict)

    eval_dataset_iterator = iter(eval_dataset)

    for i in range(args.validation_sample_number):
        data_point = next(eval_dataset_iterator)
        print(data_point)
        protein = data_point["input_ids"]
        natural_rna = data_point["labels"]
        print("RNA: ", data_point["labels"])

        inputs = torch.tensor(protein, dtype=torch.long).unsqueeze(0).to(args.device)
        # rna = generate_text(protein, self.model, 100, self.dec_tokenizer, self.args.device)
        rna = model.generate(torch.tensor(data_point['input_ids']).unsqueeze(0).to(model.device), max_length=100)
        print("Genrated RNA: ", rna)
        
        rna = dec_tokenizer.decode(rna[0].cpu().numpy().tolist())
        # print(rna)
        natural_rna = [0 if i == -100 else i for i in natural_rna]
        decoded_natural_rna = dec_tokenizer.decode(natural_rna)
        natural_rna = postprocess_rna(decoded_natural_rna)



# args = parse_opt()
# generate(args)
