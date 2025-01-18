from src.utils.helpers import generate_text, postprocess_rna, read_rna_from_fasta, get_random_rna

import argparse
import os
# import wandb
from datetime import datetime
import pytz
import json

import numpy as np
from sympy import true
import torch.nn.functional as F

from src.models import get_model
from src.utils.helpers import set_hyps, top_k_top_p_filtering, generate_text
from src.utils.tokenizer import get_tokenizer
from src.data import get_datasets
from src.utils.tokenizer import BpeTokenizer
from src.utils.helpers import generate_text, postprocess_rna
from src.utils.validations import compare_gc_content, compare_mfe_distribution
from src.utils.plots import *
from safetensors.torch import load_file
import accelerate

import torch


def evaluate(eval_dir):
    rnas = fasta_to_list(eval_dir+"/rnas.fasta")
    print(rnas)
    # with open(json_file_path, "r") as file:
    #     data = json.load(file)

    # binding_natural_scores = []
    # generated_scores = []
    # natural_scores = []
    # rnagen_scores = []


    # binding_natural_sequences = []
    # generated_sequences = []
    # natural_sequences = []
    # rnagen_sequences = []

    # for prediction in data['predictions']:
    #     if "Binding" in prediction["id"]:
    #         binding_natural_scores.append(prediction["score"])
    #         binding_natural_sequences.append(prediction["sequence"])

    #     elif "Natural" in prediction["id"]:
    #         natural_scores.append(prediction["score"])
    #         natural_sequences.append(prediction["sequence"])

    #     elif "Generated" in prediction["id"]:
    #         generated_scores.append(prediction["score"])
    #         generated_sequences.append(prediction["sequence"])

    #     elif "RNAGEN" in prediction["id"]:
    #         rnagen_scores.append(prediction["score"])
    #         rnagen_sequences.append(prediction["sequence"])

    # # binding_natural_scores.pop()
    # # natural_scores.pop()

    # best_num=128
    # generated_sequences=list(np.array(generated_sequences)[np.argsort(np.array(generated_scores))[-best_num:]])
    # rnagen_sequences=list(np.array(rnagen_sequences)[np.argsort(np.array(rnagen_scores))[-best_num:]])

    # print("binding_natural_scores", len(binding_natural_scores))
    # print("natural_scores", len(natural_scores))
    # print("generated_scores", len(generated_scores))
    # print("rnagen_scores", len(rnagen_scores))


    # print("binding_natural_scores", len(binding_natural_sequences))
    # print("natural_scores", len(natural_sequences))
    # print("generated_scores", len(generated_sequences))
    # print("rnagen_scores", len(rnagen_sequences))

    # binding_natural_scores = np.sort(np.array(binding_natural_scores))[-best_num:]
    # natural_scores = np.sort(np.array(natural_scores))[-best_num:]
    # generated_scores = np.sort(np.array(generated_scores))[-best_num:]
    # rnagen_scores = np.sort(np.array(rnagen_scores))[-best_num:]

    # print(len(binding_natural_scores))
    # print(len(natural_scores))
    # print(len(generated_scores))
    # print(len(rnagen_scores))

    # print(len(binding_natural_sequences))
    # print(len(generated_sequences))
    # print(len(rnagen_sequences))
    # print(len(natural_sequences))
 

    # data = [binding_natural_scores, generated_scores, rnagen_scores, natural_scores]
    # sequences=[binding_natural_sequences, generated_sequences, rnagen_sequences, natural_sequences]
    # labels = ["Natural Binding RNAs", "LM Generated RNAs", "RNAGEN Generated RNAs", "Random Natural RNAs"]

    # # print(data)
    # plot_violin_compare(data , labels, "Binding Score", "{}/binding_affinities_violin{}.png".format(dir))
    # plot_box_compare(data , labels, "Binding Score", "{}/binding_affinities_box{}.png".format(dir))
    # plot_ridge_compare(data , labels, "Binding Score", "{}/ridge_plot_output{}.png".format(dir))
    # plot_density_compare(data , labels, "Binding Score", '{}/density_plot_output{}.png'.format(dir))


    # compare_gc_content(sequences, labels, dir)
    # compare_mfe_distribution(sequences, labels, dir)