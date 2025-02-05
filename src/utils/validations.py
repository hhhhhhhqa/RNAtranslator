import RNA
import numpy as np
import json
import os
import tempfile
from src.utils.plots import *
from src.utils.helpers import *
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

# def compare_mfe_distribution(rna_sequences_list, labels, dir) -> None:
#     distributions = []

#     # Calculate MFE for each RNA sequence list
#     for rna_seqs in rna_sequences_list:
#         distribution = calculate_mfe_many(rna_seqs)
#         distributions.append(distribution)

#     # Mann-Whitney U Test for each pair
#     results = {}
#     for i in range(len(distributions)):
#         for j in range(i + 1, len(distributions)):
#             u_stat, p_value = mannwhitneyu(distributions[i], distributions[j], alternative='two-sided')
#             results[f"List{i+1} vs List{j+1}"] = (u_stat, p_value)
#     for comparison, (u_stat, p_value) in results.items():
#         print(f"{comparison}: U statistic = {u_stat}, p-value = {p_value}")

#     # Generate plots using dynamic functions
#     plot_violin_compare(distributions, labels, "MFE", f"{dir}/mfe_violin.png")
#     plot_box_compare(distributions, labels, "MFE", f"{dir}/mfe_box.png")
#     plot_density_compare(distributions, labels, "MFE", f'{dir}/density_plot_mfe_output.png', False)

#     return None

# Generalized GC Content Comparison
# def compare_gc_content(rna_sequences_list, labels, dir) -> None:
#     gc_distributions = []

#     # Calculate GC content for each RNA sequence list
#     for rna_seqs in rna_sequences_list:
#         gc_content = [calculate_gc_content(seq) for seq in rna_seqs]
#         gc_distributions.append(gc_content)

#     # Mann-Whitney U Test for each pair
#     results = {}
#     for i in range(len(gc_distributions)):
#         for j in range(i + 1, len(gc_distributions)):
#             u_stat, p_value = mannwhitneyu(gc_distributions[i], gc_distributions[j], alternative='two-sided')
#             results[f"List{i+1} vs List{j+1}"] = (u_stat, p_value)
#     for comparison, (u_stat, p_value) in results.items():
#         print(f"{comparison}: U statistic = {u_stat}, p-value = {p_value}")
        
#     # Generate plots using dynamic functions
#     plot_violin_compare(gc_distributions, labels, "GC Content", f"{dir}/gc_content_violin.png")
#     plot_box_compare(gc_distributions, labels, "GC Content", f"{dir}/gc_box.png")
#     plot_density_compare(gc_distributions, labels, "GC Content", f'{dir}/density_plot_gc_output.png', False)

#     return None

def compare_mfe_distribution(rna_sequences_dict, dir) -> None:
    distributions = []
    labels = list(rna_sequences_dict.keys())

    for group_name, rna_seqs in rna_sequences_dict.items():
        distribution = calculate_mfe_many(rna_seqs)
        distributions.append(distribution)

    results = {}
    for i in range(len(distributions)):
        for j in range(i + 1, len(distributions)):
            u_stat, p_value = mannwhitneyu(distributions[i], distributions[j], alternative='two-sided')
            results[f"{labels[i]} vs {labels[j]}"] = (u_stat, p_value)

    for comparison, (u_stat, p_value) in results.items():
        print(f"{comparison}: U statistic = {u_stat}, p-value = {p_value}")

    os.makedirs(f"{dir}/mfe", exist_ok=True)

    plot_violin_compare(distributions, labels, "MFE", f"{dir}/mfe/mfe_violin.png")
    plot_box_compare(distributions, labels, "MFE", f"{dir}/mfe/mfe_box.png")
    plot_density_compare(distributions, labels, "MFE", f'{dir}/mfe/density_plot_mfe_output.png', False)

    return None

def compare_gc_content(rna_sequences_dict, dir) -> None:
    gc_distributions = []
    labels = list(rna_sequences_dict.keys())

    for group_name, rna_seqs in rna_sequences_dict.items():
        gc_content = [calculate_gc_content(seq) for seq in rna_seqs]
        gc_distributions.append(gc_content)

    # Mann-Whitney U Test for each pair
    results = {}
    for i in range(len(gc_distributions)):
        for j in range(i + 1, len(gc_distributions)):
            u_stat, p_value = mannwhitneyu(gc_distributions[i], gc_distributions[j], alternative='two-sided')
            results[f"{labels[i]} vs {labels[j]}"] = (u_stat, p_value)

    for comparison, (u_stat, p_value) in results.items():
        print(f"{comparison}: U statistic = {u_stat}, p-value = {p_value}")

    os.makedirs(f"{dir}/gc", exist_ok=True)
    # Generate plots using dynamic functions
    plot_violin_compare(gc_distributions, labels, "GC Content", f"{dir}/gc/gc_content_violin.png")
    plot_box_compare(gc_distributions, labels, "GC Content", f"{dir}/gc/gc_box.png")
    plot_density_compare(gc_distributions, labels, "GC Content", f'{dir}/gc/density_plot_gc_output.png', False)

    return None

def compare_rna_similarity(rna_sequences_dict, k, dir) -> None:
    similarity_distributions = []
    labels = list(rna_sequences_dict.keys())

    for group_name, rna_seqs in rna_sequences_dict.items():
        feature_matrix = calculate_kmer_features(rna_seqs, k)
        similarity_matrix = tanimoto_similarity(feature_matrix)
        upper_triangle = similarity_matrix[np.triu_indices(len(rna_seqs), k=1)]
        similarity_distributions.append(upper_triangle)

    results = {}
    for i in range(len(similarity_distributions)):
        for j in range(i + 1, len(similarity_distributions)):
            u_stat, p_value = mannwhitneyu(similarity_distributions[i], similarity_distributions[j], alternative='two-sided')
            results[f"{labels[i]} vs {labels[j]}"] = (u_stat, p_value)

    for comparison, (u_stat, p_value) in results.items():
        print(f"{comparison}: U statistic = {u_stat}, p-value = {p_value}")

    os.makedirs(f"{dir}/similarity", exist_ok=True)

    plot_violin_compare(similarity_distributions, labels, "Tanimoto Similarity", f"{dir}/similarity/tanimoto_violin.png")
    plot_box_compare(similarity_distributions, labels, "Tanimoto Similarity", f"{dir}/similarity/tanimoto_box.png")
    plot_density_compare(similarity_distributions, labels, "Tanimoto Similarity", f"{dir}/similarity/tanimoto_density.png", False)

    return None
    
def compare_structure_distribution(rna_sequences_dict, dir, path_to_rnafold="RNAfold") -> None:
    structure_distributions = {label: {key: 0 for key in ['F', 'T', 'I', 'H', 'M', 'S']} for label in rna_sequences_dict.keys()}
    labels = list(rna_sequences_dict.keys())

    for group_name, rna_seqs in rna_sequences_dict.items():
        for seq in rna_seqs:
            struct_annotation = get_struct_annotation_viennaRNA(seq, path_to_rnafold)
            total_length = len(seq)
            for struct in struct_annotation:
                if struct in structure_distributions[group_name]:
                    structure_distributions[group_name][struct] += 1
        for struct_type in structure_distributions[group_name]:
            structure_distributions[group_name][struct_type] /= total_length

    # Mann-Whitney U Test for each structure type between groups
    results = {}
    structure_types = ['F', 'T', 'I', 'H', 'M', 'S']
    
    for struct_type in structure_types:
        values = [structure_distributions[label][struct_type] for label in labels]
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                u_stat, p_value = mannwhitneyu(
                    values[i],
                    values[j],
                    alternative='two-sided'
                )
                results[f"{labels[i]} vs {labels[j]} ({struct_type})"] = (u_stat, p_value)

    for comparison, (u_stat, p_value) in results.items():
        print(f"{comparison}: U statistic = {u_stat}, p-value = {p_value}")

    os.makedirs(f"{dir}/structure", exist_ok=True)

    fig, axes = plt.subplots(2, 3)
    axes = axes.flatten()
    
    for idx, struct_type in enumerate(structure_types):
        proportions = [structure_distributions[label][struct_type] for label in labels]
        axes[idx].bar(labels, proportions)
        axes[idx].set_title(f"{struct_type} Structure Proportion")
        axes[idx].set_xlabel("Groups")
        axes[idx].set_ylabel("Proportion")
    
    plt.tight_layout()
    plt.savefig(f"{dir}/structure/secondary_structure_bar.png")
    plt.show()

    return None

def compare_rna_length(rna_sequences_dict, dir) -> None:
    length_distributions = []
    labels = list(rna_sequences_dict.keys())

    for group_name, rna_seqs in rna_sequences_dict.items():
        lengths = [len(seq) for seq in rna_seqs]
        length_distributions.append(lengths)

    results = {}
    for i in range(len(length_distributions)):
        for j in range(i + 1, len(length_distributions)):
            u_stat, p_value = mannwhitneyu(length_distributions[i], length_distributions[j], alternative='two-sided')
            results[f"{labels[i]} vs {labels[j]}"] = (u_stat, p_value)

    for comparison, (u_stat, p_value) in results.items():
        print(f"{comparison}: U statistic = {u_stat}, p-value = {p_value}")

    os.makedirs(f"{dir}/length", exist_ok=True)

    # Plot comparisons
    plot_violin_compare(length_distributions, labels, "RNA Sequence Length", f"{dir}/length/length_violin.png")
    plot_box_compare(length_distributions, labels, "RNA Sequence Length", f"{dir}/length/length_box.png")
    plot_density_compare(length_distributions, labels, "RNA Sequence Length", f"{dir}/length/length_density.png", False)

    return None
