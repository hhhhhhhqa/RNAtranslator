import argparse
import os
# import wandb
from datetime import datetime
import pytz
import json
import tempfile

import numpy as np
import torch.nn.functional as F

from src.utils.helpers import *
from src.utils.validations import *
from src.utils.plots import *

import torch


def evaluate(eval_dir, protein=None):
    rnas = fasta_to_dict(eval_dir+"/rnas.fasta")
    
    
    if protein:
        scores = read_deepclip_output(eval_dir+f"/{protein}.json")
        # data = [binding_natural_scores, generated_scores, rnagen_scores, natural_scores]
        labels = list(scores.keys())
        data = list(scores.values())

        os.makedirs(f"{eval_dir}/deepclip", exist_ok=True)
        # # print(data)
        plot_violin_compare(data , labels, "Binding Score", "{}/deepclip/binding_affinities_violin.png".format(eval_dir))
        plot_box_compare(data , labels, "Binding Score", "{}/deepclip/binding_affinities_box.png".format(eval_dir))
        plot_ridge_compare(data , labels, "Binding Score", "{}/deepclip/ridge_plot_output.png".format(eval_dir))
        plot_density_compare(data , labels, "Binding Score", '{}/deepclip/density_plot_output.png'.format(eval_dir))

    # compare_dG_unfolding_distribution(rnas, eval_dir)
    compare_gc_content(rnas, eval_dir)
    compare_mfe_distribution(rnas, eval_dir)
    
    # compare_rna_similarity(rnas, 3, eval_dir)
    # compare_sampling_similarity(eval_dir +"/"+protein+"_pool.fasta", 3, eval_dir)

    compare_rna_length(rnas, eval_dir)
    # compare_structure_distribution(rnas, eval_dir)


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