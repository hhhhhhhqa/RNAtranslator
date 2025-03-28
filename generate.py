#!/usr/bin/env python
import os
import torch
import random
from transformers import T5ForConditionalGeneration
from src.utils.helpers import postprocess_rna, read_protein_from_fasta
from src.utils.tokenizer import get_tokenizer

def gen_rna_batch(model, rna_tokenizer, prot_ids, num_candidates, max_token=32):
    # Prepare input tensor
    inputs = torch.tensor(prot_ids, dtype=torch.long).unsqueeze(0).to(model.device)
    gen_args = {
        'max_length': int(max_token / 3),
        'repetition_penalty': 1.5,
        'encoder_repetition_penalty': 1.3,
        'num_return_sequences': num_candidates,
        'top_k': 30, 
        'temperature': 1.5, 
        'num_beams': 1,
        'do_sample': True,
    }
    with torch.no_grad():
        seqs = model.generate(inputs, **gen_args)
    
    candidate_rnas = []
    for seq in seqs:
        decoded = postprocess_rna(rna_tokenizer.decode(seq.cpu().numpy().tolist()))
        candidate_rnas.append(decoded)
    return candidate_rnas

def generate(args):
    print("Loading model from", args.checkpoints)
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoints).to(args.device)
    model.eval()
    
    source_tokenizer = get_tokenizer(
        tokenizer_name=args.tokenizer,
        vocab_size=args.vocab_size,
        seq_size=args.seq_size,
        tokenizer_path=args.source_tokenizer
    )
    rna_tokenizer = get_tokenizer(
        tokenizer_name=args.tokenizer,
        vocab_size=args.vocab_size,
        seq_size=args.seq_size,
        tokenizer_path=args.rna_tokenizer
    )

    if args.protein_fasta:
        protein_name, protein_seq = read_protein_from_fasta(args.protein_fasta)
    elif args.protein_seq:
        protein_seq = args.protein_seq
    else:
        raise ValueError("For generation, provide either --protein-fasta or both --protein-name and --protein-seq.")
    
    print("Generating RNAs for protein:", protein_name)
    prot_ids = source_tokenizer.tokenize(protein_seq).ids
    
    num_candidates = getattr(args, "rna_num", 10)
    max_token = getattr(args, "max_len", 32)
    
    candidate_rnas = gen_rna_batch(model, rna_tokenizer, prot_ids, num_candidates, max_token)
    
    os.makedirs(args.results_dir, exist_ok=True)
    output_file = os.path.join(args.results_dir, f"{protein_name}_generated.fasta")
    with open(output_file, "w") as f:
        for idx, rna in enumerate(candidate_rnas):
            f.write(f">RNA_{idx}\n{rna}\n")
    print("Generated RNAs saved to", output_file)
