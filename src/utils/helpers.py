import yaml
import os
import torch
import random
import numpy as np
import json
import pandas as pd
import re
import lib_forgi
from lib_forgi import BulgeGraph
import tempfile
import subprocess
import RNA


# from generate import generate_
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform


def set_hyps(path, args):
    with open(path, errors="ignore") as f:
        hyps = yaml.safe_load(f)
        for k, v in hyps.items():
            setattr(args, k, v)
    return args

def reproducibility(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


def top_k_top_p_filtering(logits, top_k=1, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def generate_text(sequence, model, output_length, decoder_tokenizer, device, temperature=0.01, top_k=2, top_p=0.0):
    # inputs = encoder_tokenizer.tokenize(sequence).ids
    inputs = torch.tensor(sequence, dtype=torch.long).to(device)
    inputs = inputs.unsqueeze(0)
    print("inputs: ", inputs)

    # print(sequence)

    decoder_input_ids = torch.tensor([[1]],dtype=torch.long).to(model.device)

    with torch.no_grad():
        for _ in range(output_length):
            outputs = model(input_ids=inputs, decoder_input_ids=decoder_input_ids)  # Get logits
            next_token_logits = outputs[0][:, -1, :] / temperature  # Apply temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)  # Apply top-k and/or top-p
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)  # Sample
            decoder_input_ids = torch.cat((decoder_input_ids, next_token), dim=1)  # Add the token to the generated text
            print("Outputs", decoder_input_ids)
    # print(decoder_input_ids)
    generated_text = decoder_tokenizer.decode(decoder_input_ids[0].cpu().numpy().tolist())
    return generated_text


def postprocess_rna(rna):
    return rna.replace('b', 'A').replace('j', 'C').replace(
                    'u', 'U').replace('z', 'G').replace(' ', '').replace(
                    'B', 'A').replace('J', 'C').replace('U', 'U').replace('Z', 'G')

def preprocess_rna(rna):
    return rna.lower().replace(
                    'a', 'B').replace('c', 'J').replace('u', 'U').replace('g', 'Z')


def get_random_rna(length):
    rna_vocab = {"A":0,
             "C":1,
             "G":2,
             "T":3}

    rev_rna_vocab = {v:k for k,v in rna_vocab.items()}

    mapping = dict(zip([0,1,2,3],"ACGU"))
    rsample = ''
    for i in range(length):
        p = random.random()
        if p < 0.3:
            rsample += 'C'
        elif p < 0.6:
            rsample += 'G'
        elif p < 0.8:
            rsample += 'A'
        else:
            rsample += 'U'

    return rsample


def shuffle_rna_sequences(rna_sequences):
    shuffled_sequences = []
    for rna in rna_sequences:
        rna_list = list(rna) 
        random.shuffle(rna_list)
        shuffled_sequences.append(''.join(rna_list))
    return shuffled_sequences

def read_rna_from_fasta(fasta_file_path):
    natural_rnas = []
    with open(fasta_file_path, "r") as file:
        sequence = ""
        for line in file:
            if line.startswith(">"):  
                if sequence:
                    natural_rnas.append(sequence.strip()) 
                    sequence = ""  
            else:
                sequence += line.strip()  
        if sequence:
            natural_rnas.append(sequence.strip())
    return natural_rnas

def read_rna_from_text(text_file_path):
    rna_sequences = []
    with open(text_file_path, "r") as file:
        for line in file:
            rna_sequences.append(line.strip()) 
    return rna_sequences

def read_protein_from_csv(protein_name, file_path):
    try:
        # print(protein_name)
        data = pd.read_csv(file_path)
        if 'prot_name' not in data.columns or 'seq' not in data.columns:
            raise ValueError("The CSV file must contain 'prot_name' and 'seq' columns.")
        result = data.loc[data['prot_name'] == protein_name, 'seq']
        # print(result)
    
        if not result.empty:
            return result.iloc[0]
        else:
            return f"Protein '{protein_name}' not found in the file."

    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as e:
        return f"An error occurred: {str(e)}"


def write_fasta(filename, rnas, prefix):
    with open(filename, "w") as fasta_file:
        for i, rna in enumerate(rnas, start=1):
            fasta_file.write(f">{prefix} RNA {i}\n{rna}\n")


def combine_fasta_files(output_filename, input_filenames):
    with open(output_filename, "w") as outfile:
        for input_file in input_filenames:
            with open(input_file, "r") as infile:
                outfile.write(infile.read())

def write_rna_to_fasta(fasta_file, title, sequence):
    fasta_file.write(f">{title}\n{sequence}\n")


def fasta_to_dict(fasta_file):
    rna_groups = {}
    current_group = None

    with open(fasta_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                header = line[1:]
                prefix = header.split('_')[0] + " " +header.split('_')[1]
                current_group = prefix
                if current_group not in rna_groups:
                    rna_groups[current_group] = []
            else:
                if current_group is not None:
                    rna_groups[current_group].append(line)
    return rna_groups

def calculate_kmer_features(rna_sequences, k):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k), binary=True)
    return vectorizer.fit_transform(rna_sequences).toarray()

def tanimoto_similarity(matrix):
    def tanimoto(u, v):
        intersection = (u & v).sum()
        union = u.sum() + v.sum() - intersection
        return intersection / union if union > 0 else 0

    pairwise = pdist(matrix, metric=lambda u, v: tanimoto(u > 0, v > 0))
    return squareform(1 - pairwise)

def read_deepclip_output(json_path):
    with open(json_path, 'r') as f:
        json_file = f.read()
        data = json.loads(json_file)
        scores = {}

        for prediction in data.get("predictions", []):
            group_name = prediction["id"].split("_")[0]+ " "  + prediction["id"].split("_")[1]
            if group_name not in scores:
                scores[group_name] = []
            scores[group_name].append(prediction["score"])
    
    return scores


# def run_rosetta_rna(name, fasta_file, output_dir):
#     npz_file = f"{name}.npz"
#     command = [
#         "python predict.py",  # or the specific executable name
#         "-i", fasta_file,
#         "--output", npz_file
#     ]
#     subprocess.run(command, check=True)

#         command = [
#         "python fold.py",  # or the specific executable name
#         "-OUT", f'{output_dir}',
#         "-npz", npz_file,
#         "-fasta", fasta_file
#     ]
#     subprocess.run(command, check=True)

#     print("Structure Predicdocker --versiontion Done for ", name)


# def run_hdocklite(rna, protein, output_dir):
#     subprocess.run(["/data6/sobhan/docking/HDOCK/hdock", f"{protein}", f"{rna}", "-out", f"{output_dir}/Hdock.out"], check=True)
#     subprocess.run(["/data6/sobhan/docking/HDOCK/createpl", f"{output_dir}/Hdock.out", "top100.pdb", "-nmax", "100", "-complex", "-models"], check=True)
#     print("Hdock Done for ", rna, protein)


###################### RNA Secondary Structure Annotations ######################

ENTITY_LOOKUP = {
    'b': 0,  # Bulge
    'f': 1,  # Dangling start
    't': 2,  # Dangling end
    'i': 3,  # Internal loop
    'h': 4,  # Hairpin loop
    'm': 5,  # Multi-loop
    's': 6   # Stem
}
LABELS = ['B', 'F', 'T', 'I', 'H', 'M', 'S']

def detect_bulges(dot_bracket):
    """Detect bulge positions in a dot-bracket sequence."""
    bulge_positions = set()
    stack = []
    
    for i, char in enumerate(dot_bracket):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                stack.pop()
        elif char == '.' and 0 < i < len(dot_bracket) - 1:
            if dot_bracket[i - 1] == '(' and dot_bracket[i + 1] == ')':
                bulge_positions.add(i)
    
    return bulge_positions

def parse_dot_bracket_to_labels(dot_bracket: str):
    bg = BulgeGraph()
    bg.from_dotbracket(dot_bracket, None)
    
    num_positions = len(dot_bracket)
    structure_matrix = np.zeros((7, num_positions), dtype=int)

    for line in bg.to_bg_string().split('\n'):
        line = line.strip()
        if line.startswith('define'):
            parts = line.split()
            entity = parts[1][0]
            entity_index = ENTITY_LOOKUP.get(entity)
            if entity_index is not None and len(parts) > 2:
                start_idx, end_idx = int(parts[2]) - 1, int(parts[3]) - 1
                structure_matrix[entity_index, start_idx:end_idx + 1] = 1

    for i in detect_bulges(dot_bracket):
        structure_matrix[ENTITY_LOOKUP['b'], i] = 1
    
    labels = []
    for pos in range(num_positions):
        if structure_matrix[ENTITY_LOOKUP['b'], pos] == 1:
            labels.append('B')
        else:
            row_indices = np.where(structure_matrix[:, pos] == 1)[0]
            labels.append(LABELS[row_indices[0]] if row_indices.size > 0 else ' ')
    
    return labels


def get_structure_annotations(sequence, method="rnafold"):
    if method == "rnafold":
        dot_bracket, mfe = RNA.fold(sequence)
    elif method == "mxfold2":
        mx_fold=MXFold2API()
        dot_bracket = mx_fold.predict("sdada", sequence)
    else:
        raise ValueError(f"Invalid method: {method}")

    return parse_dot_bracket_to_labels(dot_bracket) if dot_bracket else [" "] * len(rna_sequence)