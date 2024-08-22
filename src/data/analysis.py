import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from collections import defaultdict
import random

from ..utils.plots import *
from src.utils.tokenizer import *
from src.utils.helpers import *



import torch 

def parse_opt():
    ################################################################ Arguments
    parser = argparse.ArgumentParser(description='Data Analysis')

    # Input and Output path
    parser.add_argument('--data', default="/data6/sobhan/rllm/dataset/rpm/rpm.txt", type=str, help='Fasta File Path')
    parser.add_argument('--results_dir', default="/data6/sobhan/rllm/dataset/rpm/plots/protein", type=str, help='Save figures and reports path')

    # Check Duplicates
    parser.add_argument('--check_duplicates', default=False, type=bool, help='Check if there is any duplicates in the dataset')

    # Creat CSV for molecules
    parser.add_argument('--create_csv', default=False, type=bool, help='Read dataset text file and creat csv file for each molecule')

    # Analyzing sequences
    parser.add_argument('--analyze_csv', default=False, type=bool, help='Analyze the create')

    # Train Tokenizers
    parser.add_argument('--train_tokenizer', default=False, type=bool, help='Train Tokenizer')

    # Tokenize CSV
    parser.add_argument('--tokenize_csv', default=False, type=bool, help='Tokenize CSV')

    # Log Filter
    parser.add_argument('--log_filter', default=False, type=bool, help='Log Filter')

        # Log Filter
    parser.add_argument('--split_data', default=True, type=bool, help='Shuffle Data')

    args = parser.parse_args('')  # running in ipynb
    return args

def load_sequences(file_path, chunk_size=1000000):
    with open(file_path, 'r') as file:
        chunk = []
        for line in file:
            chunk.append(line.strip())
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def analyze_protein_sequences(sequences, sequence_counts):
    sequence_lengths = [len(seq) for seq in sequences]
    min_length = min(sequence_lengths)
    max_length = max(sequence_lengths)
    avg_length = np.mean(sequence_lengths)
    variance_length = np.var(sequence_lengths)

    amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    amino_acid_counts = {aa: 0 for aa in amino_acids}
    total_aa_count = 0
    for seq in sequences:
        for aa in seq:
            if aa in amino_acids:
                amino_acid_counts[aa] += 1
                total_aa_count += 1
        if seq in sequence_counts:
            sequence_counts[seq] += 1
        else:
            sequence_counts[seq] = 1

    amino_acid_freq = {aa: count / total_aa_count for aa, count in amino_acid_counts.items()}

    return sequence_lengths, min_length, max_length, avg_length, variance_length, amino_acid_freq

def analyze_rna_sequences(sequences, sequence_counts):
    sequence_lengths = [len(seq) for seq in sequences]
    min_length = min(sequence_lengths)
    max_length = max(sequence_lengths)
    avg_length = np.mean(sequence_lengths)
    variance_length = np.var(sequence_lengths)

    amino_acids = set('BUJZ')
    amino_acid_counts = {aa: 0 for aa in amino_acids}
    total_aa_count = 0
    for seq in sequences:
        for aa in seq:
            if aa in amino_acids:
                amino_acid_counts[aa] += 1
                total_aa_count += 1
        if seq in sequence_counts:
            sequence_counts[seq] += 1
        else:
            sequence_counts[seq] = 1

    nucleotide_freq = {aa: count / total_aa_count for aa, count in amino_acid_counts.items()}

    return sequence_lengths, min_length, max_length, avg_length, variance_length, nucleotide_freq

def update_stats(existing_stats, new_stats, count):
    total_count = count + len(new_stats[0])
    existing_stats[0].extend(new_stats[0])
    existing_stats[1] = min(existing_stats[1], new_stats[1])
    existing_stats[2] = max(existing_stats[2], new_stats[2])
    existing_stats[3] = (existing_stats[3] * count + new_stats[3] * len(new_stats[0])) / total_count
    existing_stats[4] = ((existing_stats[4] * count + new_stats[4] * len(new_stats[0])) / total_count)
    
    for aa in existing_stats[5]:
        existing_stats[5][aa] = (existing_stats[5][aa] * count + new_stats[5][aa] * len(new_stats[0])) / total_count
    return existing_stats, total_count

def save_sequence_counts_to_csv(sequence_counts, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Sequence', 'Length', 'Count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for seq, count in sequence_counts.items():
            writer.writerow({'Sequence': seq, 'Length': len(seq), 'Count': count})

def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data

def calculate_statistics(data):
    stats = {}
    stats['min_length'] = min(data['Length'])
    stats['max_length'] = max(data['Length'])
    stats['mean_length'] = data['Length'].mean()
    stats['min_count'] = min(data['Count'])
    stats['max_count'] = max(data['Count'])
    stats['mean_count'] = data['Count'].mean()
    stats['sum_count'] = data['Count'].sum()


    print("Statistical Information:")
    print(f"Min Length: {stats['min_length']:.2f}")
    print(f"Max Length: {stats['max_length']:.2f}")
    print(f"Mean Length: {stats['mean_length']:.2f}")

    print(f"Min Count: {stats['min_count']:.2f}")
    print(f"Max Count: {stats['max_count']:.2f}")
    print(f"Mean Count: {stats['mean_count']:.2f}")
    print(f"Sum of Count: {stats['sum_count']:.2f}")

    return stats


def read_and_create_csv(args):
    protein_stats = [[], float('inf'), float('-inf'), 0, 0, {aa: 0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}]
    rna_stats = [[], float('inf'), float('-inf'), 0, 0, {aa: 0 for aa in 'BUJZ'}]
    protein_count, rna_count = 0, 0
    protein_sequence_counts = {}
    rna_sequence_counts = {}

    for chunk in load_sequences(args.data):
        protein_sequences = [line.split('$')[0] for line in chunk]
        rna_sequences = [line.split('$')[1] for line in chunk]
        
        protein_new_stats = analyze_protein_sequences(protein_sequences, protein_sequence_counts)
        rna_new_stats = analyze_rna_sequences(rna_sequences, rna_sequence_counts)
        
        protein_stats, protein_count = update_stats(protein_stats, protein_new_stats, protein_count)
        rna_stats, rna_count = update_stats(rna_stats, rna_new_stats, rna_count)
        
    plot_bar(protein_stats[5], 'Amino Acid Composition of Protein Sequences', 'Amino Acid', 'Frequency', args.results_dir + "/aminoacid_compositions.png")
    plot_bar(rna_stats[5], 'Nucleotide Composition of RNA Sequences', 'Nucleotide', 'Frequency', args.results_dir + "/nucleotide_compositions.png")
    
    print("Protein Sequence Length Distribution:")
    print(f"  Min Length: {protein_stats[1]}")
    print(f"  Max Length: {protein_stats[2]}")
    print(f"  Average Length: {protein_stats[3]:.2f}")
    print(f"  Variance: {protein_stats[4]:.2f}")
    print("\nAmino Acid Composition:")
    for aa, freq in protein_stats[5].items():
        print(f"  {aa}: {freq:.2%}")
        
    print("RNA Sequence Length Distribution:")
    print(f"  Min Length: {rna_stats[1]}")
    print(f"  Max Length: {rna_stats[2]}")
    print(f"  Average Length: {rna_stats[3]:.2f}")
    print(f"  Variance: {rna_stats[4]:.2f}")
    print("\nNucleotide Composition:")
    for aa, freq in rna_stats[5].items():
        print(f"  {aa}: {freq:.2%}")

    save_sequence_counts_to_csv(protein_sequence_counts, args.results_dir + "/protein_" + args.output_csv)
    save_sequence_counts_to_csv(rna_sequence_counts, args.results_dir + "/rna_" + args.output_csv)


def analyze_csv(args):
    data = read_csv(args.input_csv)
    stats = calculate_statistics(data)
    
    length_columns = [col for col in data.columns if 'Length' in col]
    for col in length_columns:
        # plot_histogram(data[col], f'{col} Distribution', col, 'Frequency', f"{args.results_dir}/{col}_distribution.png")
        # plot_box(data[col], f'Box Plot of {col}', col, f"{args.results_dir}/box_{col}.png")
        plot_violin(data[col], f'Violin Plot of {col}', col, f"{args.results_dir}/violin_{col}.png")
        plot_log_histogram(data[col], f'Log-Scale Histogram of {col}', col, 'Frequency', f"{args.results_dir}/log_histogram_{col}.png")
        # plot_cdf(data[col], f'CDF of {col}', col, 'Cumulative Probability', f"{args.results_dir}/cdf_{col}.png")
        plot_scatter(data[col], data['Count'], f'{col} vs Interaction Count', 'Sequence Length', 'Sequence Interaction Count', args.results_dir + f"/{col}_vs_count.png")
    
    truncates = [512, 1024, 2048]
    # Filter lengths greater than N
    length_data = data[length_columns]
    for truncate in truncates:
        length_data_filtered = length_data[length_data > truncate]
        plot_multi_histogram(length_data_filtered, length_columns, title=f"Lost Data Comparison for {truncate}", xlabel="Length", ylabel="Interactions", filename=args.results_dir + f"/filtered_{truncate}.png")
        plot_lost_data_bar(length_data_filtered, length_columns, data, title=f"Lost Data Comparison for {truncate}", filename=args.results_dir + f"/lost_data_{truncate}.png")

    plot_histogram(data['Count'], 'Sequence Interaction Count Distribution', 'Interaction Count', 'Frequency', args.results_dir + "/count_distribution.png")
    plot_scatter(data['Length'], data['Count'], 'Length vs Interaction Count', 'Sequence Length', 'Sequence Interaction Count', args.results_dir + "/length_vs_count.png")
    plot_box(data['Count'], 'Box Plot of Sequence Interaction Counts', 'Sequence Interaction Count', args.results_dir + "/box_count.png")
    plot_violin(data['Count'], 'Violin Plot of Sequence Interaction Counts', 'Sequence Interaction Count', args.results_dir + "/violin_count.png")
    plot_log_histogram(data['Count'], 'Log-Scale Histogram of Sequence Interaction Counts', 'Interaction Count', 'Frequency', args.results_dir + "/log_histogram_count.png")
    plot_cdf(data['Count'], 'CDF of Sequence Counts', 'Sequence Interaction Count', 'Cumulative Probability', args.results_dir + "/cdf_count.png")
    plot_pairplot(data, args.results_dir + "/pairplot.png")
    plot_combined_histogram(data['Length'], data['Count'], 'Combined Histogram of Length and Count', 'Value', 'Frequency', args.results_dir + "/combined_histogram.png")
    plot_combined_log_histogram(data['Length'], data['Count'], 'Combined Log-Scale Histogram of Length and Interaction Count', 'Value', 'Frequency', args.results_dir + "/combined_log_histogram.png")

    print("Done!")



def check_duplicates(args):
    seen = set()
    duplicates = set()
    print("Start checking ...")
    # with open("/data6/sobhan/rllm/dataset/rpm/rpm.txt", 'w') as output_file:
    for chunk in load_sequences("/data6/sobhan/rllm/dataset/rpm/train_rpm.txt"):
        for line in chunk:
            if line in seen:
                duplicates.add(line)
            else:
                seen.add(line)
                    # output_file.write(line + '\n') 
    print("Done with training!")
    for chunk in load_sequences("/data6/sobhan/rllm/dataset/rpm/val_rpm.txt"):
        for line in chunk:
            if line in seen:
                duplicates.add(line)
            else:
                seen.add(line)
                # output_file.write(line + '\n') 
    print("Duplicates Count: ", len(duplicates), "Unique Count: ", len(seen))
    print("Done!")

    return duplicates, seen


def read_sequences_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    sequences = [{'text': line.strip()} for line in lines]
    print("read sequences")
    return sequences

def train_tokenizers(args):
    train_data = read_sequences_from_file("/data6/sobhan/rllm/dataset/rph/rp.txt")
    # seq_sizes = [512, 1024, 2048]
    seq_sizes = [1024]
    # vocab_sizes = [100, 1000, 25125, 50255]
    vocab_sizes = [1000]
    for seq_size in seq_sizes:
        for vocab_size in vocab_sizes:
            tokenizer = BpeTokenizer(vocab_size=vocab_size, seq_size=seq_size)
            tokenizer.train_tokenizer(train_data=train_data)
            tokenizer.save("/data6/sobhan/rllm/dataset/tokenizers/proteins", "bpe_{}_{}".format(vocab_size, seq_size))

def tokenize_csv(args):
    df = read_csv(args.input_csv)   

    seq_sizes = [512]#, 1024, 2048]
    vocab_sizes = [100, 1000, 25125, 50255]
    for seq_size in seq_sizes:
        for vocab_size in vocab_sizes:
            args.seq_size = seq_size
            args.vocab_size = vocab_size
            tokenizer = BpeTokenizer(args)
            tokenizer.load("/data6/sobhan/rllm/dataset/tokenizers/rnas", "bpe_{}_{}".format(vocab_size, seq_size))
            tokenizer.tokenizer.no_padding()
            tokenizer.tokenizer.no_truncation()
            def tokenize_sequence(sequence):
                return tokenizer.tokenize(sequence).ids
            
            df["bpe_{}_{}".format(vocab_size, seq_size)] = df['Sequence'].apply(tokenize_sequence)
            df["bpe_{}_{} Length".format(vocab_size, seq_size)] = df["bpe_{}_{}".format(vocab_size, seq_size)].apply(len)

    df.to_csv("/data6/sobhan/rllm/dataset/rph/tokenized_rna_sequences.csv", index=False)
    print(f"Tokenized sequences saved!")

def log_filter(args):
    data = read_csv(args.input_csv)
    length_columns = [col for col in data.columns if 'Length' in col]

    truncates = [512, 1024, 2048]
    for truncate in truncates:
        sum_counts = {}
        for col in length_columns:
            filtered_rows = data[data[col] > truncate]
            sum_counts[col] = filtered_rows['Count'].sum()
        for length_col, count_sum in sum_counts.items():
            print(f"The sum of Counts for {length_col} where the value is greater than {truncate} is: {count_sum}")


# Split dataset to train and evaluation data according to the proteins
def split_data(args):
    df = pd.read_csv('/data6/sobhan/rllm/dataset/rph/protein_sequences.csv')
    df_sorted = df.sort_values(by='Count')
    validation_sequences = set(df_sorted['Sequence'].iloc[::1600])

    train_counter = 0
    eval_counter = 0

    print("Start checking ...")
    with open("/data6/sobhan/rllm/dataset/rph/train_rp.txt", 'w') as train_file, open("/data6/sobhan/rllm/dataset/rph/eval_rp.txt", 'w') as eval_file:
        for chunk in load_sequences("/data6/sobhan/rllm/dataset/rph/rp.txt"):
            for line in chunk:
                protein, rna = line.strip().split('$')
                if protein in validation_sequences:
                    eval_file.write(line.strip() + '\n') 
                    eval_counter +=1
                else:
                    train_file.write(line.strip() + '\n') 
                    train_counter += 1
            print("Train size: {} and Validation size: {}".format(train_counter, eval_counter))
    print("Done!")


if __name__ == "__main__":
    args = parse_opt()

    if args.check_duplicates:
        duplicates, seen = check_duplicates(args)
        print("Duplicates Count: ", len(duplicates), "Unique Count: ", len(seen))

    if args.create_csv:
        read_and_create_csv(args)

    if args.analyze_csv:
        analyze_csv(args)

    if args.train_tokenizer:
        train_tokenizers(args=args)

    if args.tokenize_csv:
        tokenize_csv(args)

    if args.log_filter:
        log_filter(args)

    if args.split_data:
        split_data(args)
    print("Done!")