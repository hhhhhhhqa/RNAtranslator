# import numpy as np
# from concurrent.futures import ProcessPoolExecutor

# from src.utils.validations import calculate_mfe, calculate_gc_content
from src.utils.helpers import generate_text, postprocess_rna, read_rna_from_fasta, get_random_rna

# import torch

# def eval(model, args, eval_dataset, dec_tokenizer):

#     print('Evaluation Begins ...')
    
#     # Custom validation logic
#     natural_sequences = []
#     generated_sequences = []

#     eval_dataset_iterator = iter(eval_dataset)
    
#     for i in range(100):

#         data_point = next(eval_dataset_iterator)
#         print(data_point)
#         protein = data_point["input_ids"]
#         natural_rna = data_point["labels"]

#         inputs = torch.tensor(protein, dtype=torch.long).unsqueeze(0).to(args.device)
#         rna = model.generate(inputs, max_length=100)
        
#         rna = dec_tokenizer.decode(rna[0].cpu().numpy().tolist())
#         # print(rna)
#         generated_sequences.append(postprocess_rna(rna))

#         natural_rna = [0 if i == -100 else i for i in natural_rna]
#         decoded_natural_rna = dec_tokenizer.decode(natural_rna)
#         natural_rna = postprocess_rna(decoded_natural_rna)
#         natural_sequences.append(natural_rna)

#     return
    

# def my_wrapper(protein, rna):
#     rna = postprocess_rna(rna)
#     mfe = calculate_mfe(rna)
#     gc_content = calculate_gc_content(rna)
#     return (protein, mfe, gc_content)

# def eval_natural_rna(dataset_path, results_dir):
#     results = {}
#     with open(dataset_path, 'r') as file:
#         with ProcessPoolExecutor() as executor:
#             futures = []
#             co = 0
#             for line in file:
#                 protein, rna = line.strip().split("$")
#                 futures.append(executor.submit(my_wrapper, protein, rna))
#                 co += 1
#                 if co == 50:
#                     break
            
#             for future in futures:
#                 protein, mfe, gc_content = future.result()
#                 if protein not in results:
#                     results[protein] = {'mfes': [], 'gc_contents': []}
#                 results[protein]['mfes'].append(mfe)
#                 results[protein]['gc_contents'].append(gc_content)
#     # np.save(results_dir, results)
#     print(results)


# eval_natural_rna("/data6/sobhan/rllm/dataset/rph/eval_rp.txt", "./")
# rna = postprocess_rna("JJBUZJJBBZBZZBBZJJJBZJJJJUUJBUZUUUZBBBZBZZZZBUUZBBUZZBBUUZJUZUZUUUJJZZBBZBBJZUZJBUJUUZJBBZBBZJJJUZZBBUJUZJUZJJJZZJBJBZBBUZUJZZUJBZZZUUZBBJJJJUJUJUJUJJZZJUUUZJUUZZJUUUJJUUUJBJJBJJUZUUBJJUJBUUUJBUUJBZBBUUJJUZUUZZUUUJUZJZJUZZBBUJJJUZBUUUUJJUJBBZUZUZZBBBBJBJBZJBJUZBUZBZBUBJJBZUUZJUUUUJBZJZZUBBBJUZBJBZBJUUZUBB")
# print(calculate_mfe(rna))



import argparse
import os
# import wandb
from datetime import datetime
import pytz
import json

import numpy as np
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
from src.utils.validations import compare_gc_content, compare_mfe_distribution
from src.utils.plots import *
from safetensors.torch import load_file
import accelerate

import torch


def evaluate(args:object, eval_dataset, model, dec_tokenizer)->None:
    args.model_size = sum(p.numel() for p in model.parameters())
    print("Model Size: ", sum(p.numel() for p in model.parameters()))
    # print(model)

    model = model.from_pretrained(args.checkpoints).to(args.device)
    model.eval()
    eval_dataset_iterator = iter(eval_dataset)

    natural_rnas = []
    generated_rnas = []

    for i in range(100):
        
        data_point = next(eval_dataset_iterator)
        cprotein, crna = data_point['text'].strip().split("$")
        # print("RNA ", postprocess_rna(crna))
        natural_rnas.append(postprocess_rna(crna))

        protein = data_point["input_ids"]
        natural_rna = data_point["labels"]
        # print("RNA: ", data_point["labels"])

        inputs = torch.tensor(protein, dtype=torch.long).unsqueeze(0).to(args.device)
        rna = model.generate(torch.tensor(data_point['input_ids']).unsqueeze(0).to(model.device), max_length=100)
        # print("Genrated RNA: ", rna)
        
        rna = dec_tokenizer.decode(rna[0].cpu().numpy().tolist())
        # print("Genrated RNA: ", postprocess_rna(rna))
        generated_rnas.append(postprocess_rna(rna))
        
        # natural_rna = [0 if i == -100 else i for i in natural_rna] 
        # decoded_natural_rna = dec_tokenizer.decode(natural_rna)
        # natural_rna = postprocess_rna(decoded_natural_rna)
    compare_gc_content(natural_rnas, generated_rnas, "/data6/sobhan/RLLM/results/validation", 4)
    compare_mfe_distribution(natural_rnas, generated_rnas, "/data6/sobhan/RLLM/results/validation", 4)


def evaluate_single(json_file_path, dir, step):
    with open(json_file_path, "r") as file:
        data = json.load(file)

    binding_natural_scores = []
    generated_scores = []
    natural_scores = []
    rnagen_scores = []


    binding_natural_sequences = []
    generated_sequences = []
    natural_sequences = []
    rnagen_sequences = []

    for prediction in data['predictions']:
        if "Binding" in prediction["id"]:
            binding_natural_scores.append(prediction["score"])
            binding_natural_sequences.append(prediction["sequence"])

        elif "Natural" in prediction["id"]:
            natural_scores.append(prediction["score"])
            natural_sequences.append(prediction["sequence"])

        elif "Generated" in prediction["id"]:
            generated_scores.append(prediction["score"])
            generated_sequences.append(prediction["sequence"])

        elif "RNAGEN" in prediction["id"]:
            rnagen_scores.append(prediction["score"])
            rnagen_sequences.append(prediction["sequence"])

    # binding_natural_scores.pop()
    # natural_scores.pop()

    best_num=128
    generated_sequences=list(np.array(generated_sequences)[np.argsort(np.array(generated_scores))[-best_num:]])
    rnagen_sequences=list(np.array(rnagen_sequences)[np.argsort(np.array(rnagen_scores))[-best_num:]])

    print("binding_natural_scores", len(binding_natural_scores))
    print("natural_scores", len(natural_scores))
    print("generated_scores", len(generated_scores))
    print("rnagen_scores", len(rnagen_scores))


    print("binding_natural_scores", len(binding_natural_sequences))
    print("natural_scores", len(natural_sequences))
    print("generated_scores", len(generated_sequences))
    print("rnagen_scores", len(rnagen_sequences))

    binding_natural_scores = np.sort(np.array(binding_natural_scores))[-best_num:]
    natural_scores = np.sort(np.array(natural_scores))[-best_num:]
    generated_scores = np.sort(np.array(generated_scores))[-best_num:]
    rnagen_scores = np.sort(np.array(rnagen_scores))[-best_num:]

    print(len(binding_natural_scores))
    print(len(natural_scores))
    print(len(generated_scores))
    print(len(rnagen_scores))

    print(len(binding_natural_sequences))
    print(len(generated_sequences))
    print(len(rnagen_sequences))
    print(len(natural_sequences))
 

    data = [binding_natural_scores, generated_scores, rnagen_scores, natural_scores]
    sequences=[binding_natural_sequences, generated_sequences, rnagen_sequences, natural_sequences]
    labels = ["Natural Binding RNAs", "LM Generated RNAs", "RNAGEN Generated RNAs", "Random Natural RNAs"]

    # print(data)
    plot_violin_compare(data , labels, "Binding Score", "{}/binding_affinities_violin{}.png".format(dir, step))
    plot_box_compare(data , labels, "Binding Score", "{}/binding_affinities_box{}.png".format(dir, step))
    plot_ridge_compare(data , labels, "Binding Score", "{}/ridge_plot_output{}.png".format(dir, step))
    plot_density_compare(data , labels, "Binding Score", '{}/density_plot_output{}.png'.format(dir, step))


    compare_gc_content(sequences, labels, dir, step)
    compare_mfe_distribution(sequences, labels, dir, step)


# For RNAGEN paper 

# def evaluate_single(json_file_path, dir, step):
#     with open(json_file_path, "r") as file:
#         data = json.load(file)

#     binding_natural_scores = []
#     # generated_scores = []
#     natural_scores = []
#     rnagen_scores = []


#     binding_natural_sequences = []
#     # generated_sequences = []
#     natural_sequences = []
#     rnagen_sequences = []

#     for prediction in data['predictions']:
#         if "Binding" in prediction["id"]:
#             binding_natural_scores.append(prediction["score"])
#             binding_natural_sequences.append(prediction["sequence"])

#         elif "pi_RNA" in prediction["id"]:
#             natural_scores.append(prediction["score"])
#             natural_sequences.append(prediction["sequence"])

#         # elif "Generated" in prediction["id"]:
#         #     generated_scores.append(prediction["score"])
#         #     generated_sequences.append(prediction["sequence"])

#         elif "RNAGEN" in prediction["id"]:
#             rnagen_scores.append(prediction["score"])
#             rnagen_sequences.append(prediction["sequence"])

#     # binding_natural_scores.pop()
#     # natural_scores.pop()

#     # best_num=128
#     # generated_sequences=list(np.array(generated_sequences)[np.argsort(np.array(generated_scores))[-best_num:]])
#     # rnagen_sequences=list(np.array(rnagen_sequences)[np.argsort(np.array(rnagen_scores))[-best_num:]])

#     print(len(binding_natural_scores))
#     print(len(natural_scores))
#     # print(len(generated_scores))
#     print(len(rnagen_scores))


#     # binding_natural_scores = np.sort(np.array(binding_natural_scores))[-best_num:]
#     # natural_scores = np.sort(np.array(natural_scores))[-best_num:]
#     # # generated_scores = np.sort(np.array(generated_scores))[-best_num:]
#     # rnagen_scores = np.sort(np.array(rnagen_scores))[-best_num:]

#     print(len(binding_natural_scores))
#     print(len(natural_scores))
#     # print(len(generated_scores))
#     print(len(rnagen_scores))

#     print(len(binding_natural_sequences))
#     # print(len(generated_sequences))
#     print(len(rnagen_sequences))
#     print(len(natural_sequences))
 

#     data = [binding_natural_scores, rnagen_scores, natural_scores]
#     sequences=[binding_natural_sequences, rnagen_sequences, natural_sequences]
#     labels = ["Natural Binding RNAs", "RNAGEN Generated RNAs", "Random Natural RNAs"]

#     # print(data)
#     plot_violin_compare(data , labels, "Binding Score", "{}/binding_affinities_violin{}.png".format(dir, step))
#     plot_box_compare(data , labels, "Binding Score", "{}/binding_affinities_box{}.png".format(dir, step))
#     plot_ridge_compare(data , labels, "Binding Score", "{}/ridge_plot_output{}.png".format(dir, step))
#     plot_density_compare(data , labels, "Binding Score", '{}/density_plot_output{}.png'.format(dir, step))


#     compare_gc_content(sequences, labels, dir, step)
#     compare_mfe_distribution(sequences, labels, dir, step)
