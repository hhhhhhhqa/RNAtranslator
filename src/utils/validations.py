import RNA
import numpy as np
import json

from src.utils.plots import *
from src.utils.helpers import shuffle_rna_sequences
from scipy.stats import mannwhitneyu

def calculate_mfe(sequence):
    _, mfe = RNA.fold(sequence)
    return mfe

def calculate_mfe_many(sequences):
    mfes = [calculate_mfe(seq) for seq in sequences]
    return mfes

def calculate_gc_content(rna_sequence):
    gc_content = ((rna_sequence.count('G') + rna_sequence.count('C')) / len(rna_sequence)) * 100
    return gc_content

def calculate_token_distribution(rna_sequences, vocab_size, tokenizer):
    # print(rna_sequences)
    # token_counts = {'A': 0, 'U': 0, 'G': 0, 'C': 0}
    inputs = [sequence.ids for sequence in tokenizer.encode(rna_sequences)]
    # print(np.shape(inputs))
    token_counts = {}
    for token in range(tokenizer.tokenizer.get_vocab_size()):
        token_counts[token] = 0
    total_tokens = 0

    for sequence in inputs:
        for token in sequence:
            if token in token_counts:
                token_counts[token] += 1
                total_tokens += 1

    token_frequencies = {token: count / total_tokens for token, count in token_counts.items()}

    return token_frequencies

def report_gc_content(generated_sequences, dir, step, save=True):
    generated_gc_contents = [calculate_gc_content(seq) for seq in generated_sequences]
    generated_mean = np.mean(generated_gc_contents)
    generated_std = np.std(generated_gc_contents)
    plot_histogram(generated_gc_contents, 'GC Content Distribution - Generated RNA', 'GC Content (%)', 'Frequency', dir+f'/generated_gc_histogram_{step}.png')
    return {
        'generated_gc_contents': generated_gc_contents,
        'generated_mean': generated_mean,
        'generated_std': generated_std
    }

def report_token_distribution(generated_sequences, tokenizer, dir, step, save=True):
    generated_distribution = calculate_token_distribution(generated_sequences,1000,tokenizer)

    tokens = [i for i in range(tokenizer.tokenizer.get_vocab_size())]
    generated_values = [generated_distribution[token] for token in tokens]

    path_ = dir + f'/token_distribution_{step}.png'

    plot_token_distribution_single(tokens, generated_values, path_)

    return {
        'generated_distribution': generated_distribution
    }

def report_mfe_distribution(generated_sequences, dir, step, save=True):
    generated_distribution = calculate_mfe_many(generated_sequences)

    path_ = dir + f'/mfe_distribution_{step}.png'

    plot_mfe_distribution_single(generated_distribution, path_)

    return {
        'generated_distribution': generated_distribution
    }


def compare_token_distribution(natural_sequences, generated_sequences, tokenizer, dir, step, save=True):
    natural_distribution = calculate_token_distribution(natural_sequences,1000,tokenizer)
    generated_distribution = calculate_token_distribution(generated_sequences,1000,tokenizer)

    tokens = [i for i in range(tokenizer.tokenizer.get_vocab_size())]
    natural_values = [natural_distribution[token] for token in tokens]
    generated_values = [generated_distribution[token] for token in tokens]

    path_ = dir + f'/token_distribution_{step}.png'

    plot_token_distribution(tokens, natural_values, generated_values, path_)

    return {
        'natural_distribution': natural_distribution,
        'generated_distribution': generated_distribution
    }

def compare_mfe_distribution(rna_sequences_list, labels, dir, step, save=True) -> None:
    distributions = []

    # Calculate MFE for each RNA sequence list
    for rna_seqs in rna_sequences_list:
        distribution = calculate_mfe_many(rna_seqs)
        # distribution_sorted = np.sort(np.array(distribution))[:500]  # Sort and limit to best 500 values
        distributions.append(distribution)

    # Mann-Whitney U Test for each pair
    results = {}
    for i in range(len(distributions)):
        for j in range(i + 1, len(distributions)):
            u_stat, p_value = mannwhitneyu(distributions[i], distributions[j], alternative='two-sided')
            results[f"List{i+1} vs List{j+1}"] = (u_stat, p_value)
    for comparison, (u_stat, p_value) in results.items():
        print(f"{comparison}: U statistic = {u_stat}, p-value = {p_value}")

    # Generate plots using dynamic functions
    plot_violin_compare(distributions, labels, "MFE", f"{dir}/mfe_violin_{step}.png")
    plot_box_compare(distributions, labels, "MFE", f"{dir}/mfe_box_{step}.png")
    plot_density_compare(distributions, labels, "MFE", f'{dir}/density_plot_mfe_output_{step}.png', False)

    return None

# Generalized GC Content Comparison
def compare_gc_content(rna_sequences_list, labels, dir, step, save=True) -> None:
    gc_distributions = []

    # Calculate GC content for each RNA sequence list
    for rna_seqs in rna_sequences_list:
        gc_content = [calculate_gc_content(seq) for seq in rna_seqs]
        gc_distributions.append(gc_content)

    # Mann-Whitney U Test for each pair
    results = {}
    for i in range(len(gc_distributions)):
        for j in range(i + 1, len(gc_distributions)):
            u_stat, p_value = mannwhitneyu(gc_distributions[i], gc_distributions[j], alternative='two-sided')
            results[f"List{i+1} vs List{j+1}"] = (u_stat, p_value)
    for comparison, (u_stat, p_value) in results.items():
        print(f"{comparison}: U statistic = {u_stat}, p-value = {p_value}")
        
    # Generate plots using dynamic functions
    plot_violin_compare(gc_distributions, labels, "GC Content", f"{dir}/gc_content_violin_{step}.png")
    plot_box_compare(gc_distributions, labels, "GC Content", f"{dir}/gc_box_{step}.png")
    plot_density_compare(gc_distributions, labels, "GC Content", f'{dir}/density_plot_gc_output_{step}.png', False)

    return None
# def single_validate(json_file_path, dir, step):
#     with open(json_file_path, "r") as file:
#         data = json.load(file)

#     binding_natural_scores = []
#     generated_scores = []
#     natural_scores = []
#     random_scores = []


#     binding_natural_sequences = []
#     generated_sequences = []
#     natural_sequences = []
#     random_sequences = []

#     for prediction in data['predictions']:
#         if "Binding" in prediction["id"]:
#             binding_natural_scores.append(prediction["score"])
#             binding_natural_sequences.append(prediction["sequence"])

#         elif "Natural" in prediction["id"]:
#             natural_scores.append(prediction["score"])
#             natural_sequences.append(prediction["sequence"])

#         elif "Generated" in prediction["id"]:
#             generated_scores.append(prediction["score"])
#             generated_sequences.append(prediction["sequence"])

#         elif "Random" in prediction["id"]:
#             random_scores.append(prediction["score"])
#             random_sequences.append(prediction["sequence"])


#     best_num=500
#     binding_natural_scores = np.sort(np.array(binding_natural_scores))[-best_num:]
#     natural_scores = np.sort(np.array(natural_scores))[-best_num:]
#     generated_scores = np.sort(np.array(generated_scores))[-best_num:]
#     random_scores = np.sort(np.array(random_scores))[-best_num:]

#     plot_violin_compare(binding_natural_scores, generated_scores, natural_scores, random_scores, ["Natural Binding RNAs", "Generated RNAs", "Random Natural RNAs", "Random Generated RNAs"], "Binding Score", "{}/binding_affinities_violin{}.png".format(dir, step))
#     plot_box_compare(binding_natural_scores, generated_scores, natural_scores, random_scores, ["Natural Binding RNAs", "Generated RNAs", "Random Natural RNAs", "Random Generated RNAs"], "Binding Score", "{}/binding_affinities_box{}.png".format(dir, step))
#     plot_ridge_compare(binding_natural_scores, generated_scores, natural_scores, random_scores, "Binding Score", "{}/ridge_plot_output{}.png".format(dir, step))
#     plot_density_compare(binding_natural_scores, generated_scores, natural_scores, random_scores, "Binding Score", '{}/density_plot_output{}.png'.format(dir, step))


#     compare_gc_content(binding_natural_sequences, generated_sequences, natural_sequences, random_sequences, dir, step)
#     compare_mfe_distribution(binding_natural_sequences, generated_sequences, natural_sequences, random_sequences, dir, step)
