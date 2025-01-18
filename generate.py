import argparse
import os
from datetime import datetime
import pytz
import json

import torch.nn.functional as F
import torch

from src.utils.helpers import *

def gen_rna_batch(model, prot_ids, dec_tok, num_samps, max_len=70):
    # Generate RNAs, but ensure final length is within max_len
    inputs = torch.tensor(prot_ids, dtype=torch.long).unsqueeze(0).to(model.device)

    valid_rnas = []
    ratio = 0
    while len(valid_rnas) < num_samps:
        ratio += 1
        with torch.no_grad():
            # Generate 10x the required samples to ensure enough valid RNAs
            seqs = model.generate(
                inputs,
                max_length=max_len/ratio,  # Allow longer generation to ensure diversity
                min_length=5,
                temperature=1.0,
                do_sample=True,
                num_beams=1,
                repetition_penalty=1.5,
                encoder_repetition_penalty=1.3,
                num_return_sequences=num_samps  # Generate more sequences
            )

        decoded_rnas = [postprocess_rna(dec_tok.decode(seq.cpu().numpy().tolist())) for seq in seqs]
        valid_rnas.extend([rna for rna in decoded_rnas if len(rna) <= max_len])
        valid_rnas = valid_rnas[:num_samps]

    return valid_rnas

def filter_rna(input_path, output_path, max_rnas, min_len=10, max_len=70):
    rnas = []

    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        if input_path.endswith(".txt"):
            # TXT mode: filter and write in one pass
            for count, line in enumerate(infile):
                if count >= max_rnas:
                    break
                rna = line.strip()
                if min_len <= len(rna) <= max_len:
                    rnas.append(rna)
                    outfile.write(f">RNA_{count}\n{rna}\n")
        else:
            # FASTA mode: parse and filter
            seq = ""
            head = ""
            for line in infile:
                if line.startswith(">"):
                    if seq and min_len <= len(seq) <= max_len and len(rnas) < max_rnas:
                        rnas.append(seq)
                        outfile.write(head + seq + "\n")
                    head = line
                    seq = ""
                else:
                    seq += line.strip()
            # Write the last sequence if valid
            if seq and min_len <= len(seq) <= max_len and len(rnas) < max_rnas:
                rnas.append(seq)
                outfile.write(head + seq + "\n")

    return rnas

def generate(args, model, enc_tokenizer, dec_tokenizer, result_dir):
    args.model_size = sum(p.numel() for p in model.parameters())
    print("Model size:", args.model_size)
    protein = read_protein_from_csv(protein_name=args.proteins, file_path="/data6/sobhan/dataset/proteins/protein_seqs.csv")
    prot_ids = enc_tokenizer.tokenize(protein).ids

    model = model.from_pretrained(args.checkpoints).to(args.device)
    model.eval()

    rna_files = {
        "natural": "/data6/sobhan/RLLM/dataset/rph/natural/natural_rnas.fna",
        "binding": "/data6/sobhan/dataset/CLIP/CLIP_CLUSTERED_FASTA/RBM5_clustered.fa",
        "rnagen": "/data6/sobhan/RNAGEN/output/RBM5_inv_distance_softmax_method_maxiters_3000_v1/RBM5_best_binding_sequences.txt",
    }
    filtered_paths = {
        key: os.path.join(result_dir, f"{key}_rnas.fasta") for key in rna_files
    }

    # Filter RNAs per type
    filtered_rnas = {
        key: filter_rna(path, filtered_paths[key], args.rna_num)
        for key, path in rna_files.items()
    }
    print("Filtered RNAs:", filtered_rnas)

    # Generate new RNAs
    print("Generating RNAs...")
    gen_rnas = gen_rna_batch(model, prot_ids, dec_tokenizer, args.rna_num)

    generated_path = os.path.join(result_dir, "generated_rnas.fasta")
    with open(generated_path, "w") as gen_outfile:
        for idx, rna in enumerate(gen_rnas):
            gen_outfile.write(f">Generated_RNA_{idx}\n{rna}\n")

    # Save all RNAs into one file
    combined_path = os.path.join(result_dir, "rnas.fasta")
    with open(combined_path, "w") as outfile:
        for key, rnas in filtered_rnas.items():
            for idx, rna in enumerate(rnas):
                outfile.write(f">{key}_RNA_{idx}\n{rna}\n")
        for idx, rna in enumerate(gen_rnas):
            outfile.write(f">Generated_RNA_{idx}\n{rna}\n")

    print(f"All RNAs combined and saved at {combined_path}")
