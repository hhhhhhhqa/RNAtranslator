import RNA
import numpy as np

from src.utils.plots import plot_histogram, plot_token_distribution, plot_mfe_distribution, plot_token_distribution_single, plot_mfe_distribution_single

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

def compare_gc_content(natural_sequences, generated_sequences, dir, step, save=True):
    natural_gc_contents = [calculate_gc_content(seq) for seq in natural_sequences]
    generated_gc_contents = [calculate_gc_content(seq) for seq in generated_sequences]

    # Calculate mean and standard deviation
    natural_mean = np.mean(natural_gc_contents)
    natural_std = np.std(natural_gc_contents)
    generated_mean = np.mean(generated_gc_contents)
    generated_std = np.std(generated_gc_contents)

    # Plotting
    plot_histogram(natural_gc_contents, 'GC Content Distribution - Natural RNA', 'GC Content (%)', 'Frequency', dir+f'/natural_gc_histogram_{step}.png')
    plot_histogram(generated_gc_contents, 'GC Content Distribution - Generated RNA', 'GC Content (%)', 'Frequency', dir+f'/generated_gc_histogram_{step}.png')

    return {
        'natural_gc_contents': natural_gc_contents,
        'generated_gc_contents': generated_gc_contents,
        'natural_mean': natural_mean,
        'natural_std': natural_std,
        'generated_mean': generated_mean,
        'generated_std': generated_std
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

def compare_mfe_distribution(natural_sequences, generated_sequences, dir, step, save=True):
    natural_distribution = calculate_mfe_many(natural_sequences)
    generated_distribution = calculate_mfe_many(generated_sequences)

    path_ = dir + f'/mfe_distribution_{step}.png'

    plot_mfe_distribution(natural_distribution, generated_distribution, path_)

    return {
        'natural_distribution': natural_distribution,
        'generated_distribution': generated_distribution
    }
