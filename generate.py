import argparse
import os

from src.utils.helpers import * 

clip_directory = "/data6/sobhan/RLLM_OPT/deepclip_models" 

def gen_rna_batch(model, prot_ids, dec_tok, num_candidates, tolerance=5, max_token=32, strategy='beam_search', temperature=1.0, num_beams=5, top_k=None, top_p=None):
    inputs = torch.tensor(prot_ids, dtype=torch.long).unsqueeze(0).to(model.device)
    
    candidate_rnas = []
    if strategy == 'beam_search':
        num_candidates = num_beams

    gen_args = {
        'max_length': int(max_token/3),
        'min_length': 20,
        'repetition_penalty': 1.5,
        'encoder_repetition_penalty': 1.3,
        'num_return_sequences': 128,
    }
    if strategy == 'beam_search':
        # For beam search, ensure that num_return_sequences <= num_beams.
        gen_args.update({
            'do_sample': False,
            'num_beams': num_beams,
            'num_return_sequences': num_candidates,
        })
    elif strategy == 'top_k':
        gen_args.update({
            'do_sample': True,
            'temperature': temperature,
            'top_k': top_k if top_k is not None else 50,
            'num_beams': num_beams,
        })
    elif strategy == 'top_p':
        gen_args.update({
            'do_sample': True,
            'temperature': temperature,
            'top_p': top_p if top_p is not None else 0.92,
            'num_beams': num_beams,
        })
    else:  # Simple sampling
        gen_args.update({
            'do_sample': True,
            'temperature': temperature,
            'num_beams': num_beams,
        })

    while len(candidate_rnas) < num_candidates:
        print("GEnerating: ", len(candidate_rnas), "/nNum Candidates: ", num_candidates)
        with torch.no_grad():
            seqs = model.generate(inputs, **gen_args)
            
        decoded_rnas = [
            postprocess_rna(dec_tok.decode(seq.cpu().numpy().tolist()))
            for seq in seqs
        ]
        if strategy == 'beam_search':
            candidate_rnas.extend(decoded_rnas)
            break
        
        new_candidates = [
            rna for rna in decoded_rnas
            if 10 <= len(rna) <= (max_token)
        ]
        candidate_rnas.extend(new_candidates)
        candidate_rnas = candidate_rnas[:num_candidates]
    return candidate_rnas



def fasta_to_fasta(input_path, output_path, max_rnas, min_len=10, max_len=37):
    rnas = []
    sequences = []
    headers = []

    with open(input_path, "r") as infile:
        if input_path.endswith(".txt"):
            for line in infile:
                sequences.append(line.strip())
        else:
            seq = ""
            head = ""
            for line in infile:
                if line.startswith(">"):
                    if seq:
                        sequences.append(seq)
                        headers.append(head)
                    head = line
                    seq = ""
                else:
                    seq += line.strip().upper().replace("T", "U")
            if seq:
                sequences.append(seq)
                headers.append(head)

    valid_indices = [i for i, seq in enumerate(sequences) if min_len <= len(seq) <= max_len]
    if not valid_indices:
        return []

    sampled_indices = random.sample(valid_indices, min(max_rnas, len(valid_indices)))
    sampled_indices.sort() #sort to keep the order of the original file as much as possible, while still sampling.
    with open(output_path, "w") as outfile:
        for index in sampled_indices:
            rnas.append(sequences[index])
            if input_path.endswith(".txt"):
              outfile.write(f">RNA_{len(rnas)-1}\n{sequences[index]}\n")
            else:
              outfile.write(headers[index]+sequences[index]+"\n")

    return rnas



def create_pool(args, model, source_tokenizer, rna_tokenizer):
    grid_config = {
        'beam_search': [
            {'num_beams': 1},
            {'num_beams': 25},
            {'num_beams': 50},
            {'num_beams': 100},
            {'num_beams': 200},
        ],
        'top_k': [
            # {'top_k': 30, 'temperature': 0.7, 'num_beams': 1},
            {'top_k': 30, 'temperature': 1.0, 'num_beams': 1},
            {'top_k': 30, 'temperature': 1.5, 'num_beams': 1},
            {'top_k': 100, 'temperature': 0.7, 'num_beams': 1},
            {'top_k': 100, 'temperature': 1.0, 'num_beams': 1},
            # {'top_k': 100, 'temperature': 1.5, 'num_beams': 1},
        ],
        'top_p': [
            # {'top_p': 0.8, 'temperature': 0.7, 'num_beams': 1},
            {'top_p': 0.8, 'temperature': 1.0, 'num_beams': 1},
            {'top_p': 0.8, 'temperature': 1.5, 'num_beams': 1},
            {'top_p': 0.95, 'temperature': 0.7, 'num_beams': 1},
            {'top_p': 0.95, 'temperature': 1.0, 'num_beams': 1},
            # {'top_p': 0.95, 'temperature': 1.5, 'num_beams': 1},
        ],
        'sample': [
            {'temperature': 0.7, 'num_beams': 1},
            {'temperature': 1.0, 'num_beams': 1},
            {'temperature': 1.5, 'num_beams': 1},
        ]
    }
    
    os.makedirs(args.eval_dir, exist_ok=True)
    for protein_name in args.proteins:
        protein_seq = read_protein_from_csv(protein_name, file_path="/data6/sobhan/dataset/proteins/protein_seqs.csv")
        if protein_seq is None:
            print(f"Warning: Protein {protein_name} not found.")
            continue
        print("Tokenizing Protein:", protein_name)
        prot_ids = source_tokenizer.tokenize(protein_seq).ids
        
        pool = []
        for strategy, hyper_list in grid_config.items():
            for hyperparams in hyper_list:
                temperature = hyperparams.get('temperature', 1.0)
                num_beams = hyperparams.get('num_beams', 1)
                top_k = hyperparams.get('top_k', None)
                top_p = hyperparams.get('top_p', None)
                
                print(f"Generating for Protein: {protein_name}, Strategy: {strategy}, Hyperparameters: {hyperparams}")
                candidate_rnas = gen_rna_batch(
                    model,
                    prot_ids,
                    rna_tokenizer,
                    args.pool_size,
                    max_token=args.max_len,
                    strategy=strategy,
                    temperature=temperature,
                    num_beams=num_beams,
                    top_k=top_k,
                    top_p=top_p
                )
                for idx, rna in enumerate(candidate_rnas):
                    pool.append({
                        "rna": rna,
                        "strategy": strategy,
                        "hyperparams": hyperparams,
                        "id": f"RNA_{idx}_{strategy}_{'_'.join([f'{k}_{v}' for k,v in hyperparams.items()])}"
                    })
        
        # pool_file = pool_to_fasta(protein_name, pool, args.eval_dir)
        pool_filename = os.path.join(args.eval_dir, f"{protein_name}_pool.fasta")
        with open(pool_filename, "w") as f:
            for idx, cand in enumerate(pool):
                param_str = "_".join([f"{k}_{v}" for k, v in cand["hyperparams"].items()])
                header = f">RNA_{idx}_{cand['strategy']}_{param_str}"
                f.write(header + "\n")
                f.write(cand["rna"] + "\n")
        print(f"Pool file saved: {pool_filename}")
        return pool_filename

def aggregate(args):
    rna_files = {
        "Natural_non-binding": "/data6/sobhan/dataset/DeepCLIP/dataset/RBM5/RBM5_negatives.txt",
        "Natural_Binding": f"/data6/helya/dataset/CLIPdb_cluster/cd_hit_results_RBPs/identity_100/{args.proteins[0].upper()}_rnas_cdhit_100.fa",
        # "RNAGEN_ ": "/data6/sobhan/RNAGEN/output/RBM5_inv_distance_softmax_method_maxiters_3000_v1/RBM5_best_binding_sequences.txt",
        # "GenerRNA_ ": "/data6/sobhan/GenerRNA/rbm5_pool.fasta",
        "RNAtranslator_ ": f"{args.eval_dir}/{args.proteins[0]}_filtered.fasta"
    }

    filtered_paths = {
        key: os.path.join(args.eval_dir, f"{key}_rnas.fasta") for key in rna_files
    }
    filtered_rnas = {
        key: fasta_to_fasta(path, filtered_paths[key], args.rna_num, min_len=10, max_len=args.max_len+5)
        for key, path in rna_files.items()
    }
    # print(filtered_rnas)

    combined_path = os.path.join(args.eval_dir, "rnas.fasta")
    with open(combined_path, "w") as outfile:
        for key, rnas in filtered_rnas.items():
            for idx, rna in enumerate(rnas):
                outfile.write(f">{key}_RNA_{idx}\n{rna}\n")
        # for idx, rna in enumerate(gen_rnas):
        #     outfile.write(f">Generated_RNA_{idx}\n{rna}\n")

    print(f"All RNAs combined and saved at {combined_path}") 


def calc_deep_clip(rna_sequences, protein_name) -> int:
    clip_file = next(
        (os.path.join(root1, file1) for root1, _, files1 in os.walk(clip_directory)
            for file1 in files1 if protein_name.lower() in file1.lower()),
        None
    )
    print("DeepCLIP file: ", clip_file)
    try:
        net, freq = network.load_network(clip_file)
        options = net.options
        predict_fn, outpar = net.compile_prediction_function()
        output_shape = net.network['l_in'].output_shape
    except Exception as e:
        raise ValueError(f"Error loading network: {e}")

    max_filter_size = max(options["FILTER_SIZES"]) / len(constants.VOCAB)
    max_network_length = int(options["SEQ_SIZE"] - 2 * (max_filter_size - 1))
    max_input_length = max(map(len, rna_sequences))

    if max_input_length > max_network_length:
        raise ValueError(f"Input sequences exceed the maximum network length ({max_network_length}).")

    # Encode and predict
    seq_list = encode_input_data(rna_sequences, max_network_length + 2 * (max_filter_size - 1))
    X_test = onehot_encode(seq_list, freq, vocab=constants.VOCAB)
    results = network.predict_without_network(predict_fn, options, output_shape, X_test, outpar)
    predictions = results["predictions"]
    return predictions


def read_pool(fasta_file):
    candidates = []
    with open(fasta_file, 'r') as f:
        header = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    candidates.append({"header": header, "rna": "".join(seq_lines)})
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            candidates.append({"header": header, "rna": "".join(seq_lines)})
    return candidates


def calc_scores(args, candidates, phi_cons=1, phi2_mfe=1, phi3_bind=1):
    rnas = [cand['rna'] for cand in candidates]

    if args.max_len < 75 and not args.ignore_clip:
        #Binding Affinity
        predictions = calc_deep_clip(copy.deepcopy(rnas), protein_name) 

    #Foldability
    mfes = calculate_mfe_many(copy.deepcopy(rnas)) 
    np_mfes = np.array(copy.copy(mfes))
    normalized_mfes = list(1 - (mfes - np.min(np_mfes)) / (np.max(np_mfes) - np.min(np_mfes)))

    # Consistency score
    vote_counts = Counter(copy.deepcopy(rnas))
    max_vote = vote_counts.most_common(1)[0][1]

    for index, cand in enumerate(candidates):
        if args.max_len < 75 and not args.ignore_clip:
            cand["deepclip_score"] = predictions.pop(0)[0]
        cand["MFE"] = mfes.pop(0)
        cand["MFE_score"] = normalized_mfes.pop(0)
        cand["vote_score"] = vote_counts[cand["rna"]] / max_vote

        if args.max_len < 75 and not args.ignore_clip:
            cand["final_score"] = phi3_bind*cand["deepclip_score"] + phi2_mfe*cand["MFE_score"] + phi_cons*cand["vote_score"]
        else:
            cand["final_score"] = phi2_mfe*cand["MFE_score"] + phi_cons*cand["vote_score"]

    return candidates


def parse_opt():
    ################################################################ Arguments
    parser = argparse.ArgumentParser(description='Sampling Hyperparameters')

    # Training Configuration
    parser.add_argument('--runmode', default="create_pool", choices=["create_pool", "filter", "aggregate"], type=str, help='Runmode')
    parser.add_argument('--train-hyp', default="/data6/sobhan/RLLM/hyps/train.yaml", type=str, help='Training hyperparameters')
    parser.add_argument('--model-hyp', default="/data6/sobhan/RLLM/hyps/t5.yaml", type=str, help='Model hyperparameters')

    # Generation Configurations
    parser.add_argument('--checkpoints', default='/data6/sobhan/RLLM/finetune/checkpoint-374800', type=str, help='Load Model')
    parser.add_argument('--eval-dir', default="/data6/sobhan/RLLM/results/validation/pool", type=str, help='Output dir of the evaluation')
    parser.add_argument('--proteins', nargs='+', default=['hnrpnc', 'ago2', 'elavl1', 'rbm5'], type=str, help='List of protein names or IDs')
    parser.add_argument('--rna_num', default=128, type=int, help='Number of RNAs to aggregate for evaluation')
    parser.add_argument('--pool_size', default=128, type=int, help='Number of RNAs to generate per setting')
    parser.add_argument('--max_len', default=32, type=int, help='Maximum length of RNA sequence')
    parser.add_argument('--ignore_clip', default=False, type=bool, help='If ignore the deepclip score')


    args = parser.parse_args()    
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} ID: cuda:{i}")
    return args


def encode_input_data(seqs, max_length):
    pad_sequences_with_N(seqs, max_length)
    return seqs

def pad_sequences_with_N(added_seqs, length):
    for i in range(len(added_seqs)):
        begin = end = 0  # make sure these are zero
        if len(added_seqs[i]) < length:
            missing = int(length - len(added_seqs[i]))
            begin = int(missing/2)
            end = int(missing - begin)
        added_seqs[i] = begin*'n' + added_seqs[i] + end*'n'
        if len(added_seqs[i]) != length:
            print(str(len(added_seqs[i])))
            print(str(i))
            break
    return added_seqs


if __name__ == "__main__":
    args = parse_opt()
    args = set_hyps(args.train_hyp, args)
    args = set_hyps(args.model_hyp, args)

    if args.runmode == "create_pool":
        from src.utils.tokenizer import *
        import torch.nn.functional as F
        import torch
        from transformers import T5ForConditionalGeneration

        print("Loading Model: ", args.checkpoints)
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
        create_pool(args, model, source_tokenizer, rna_tokenizer)

    if args.runmode == "filter":
        from collections import Counter
        import numpy as np
        import sys
        import copy

        from src.utils.validations import calculate_mfe_many
        from src.utils.helpers import read_protein_from_csv


        project1_path = '/data6/sobhan/deepclip'
        sys.path.append(project1_path)
        import constants
        import network
        from data_gen import onehot_encode
        from data_gen import onehot_binary
        # from data_gen import onehot_binary_encode

        for protein_name in args.proteins:
            protein_seq = read_protein_from_csv(protein_name, file_path="/data6/sobhan/dataset/proteins/protein_seqs.csv")
            print(protein_name)
            if protein_seq is None:
                continue
            pool_file = os.path.join(args.eval_dir, f"{protein_name}_pool.fasta")
            if os.path.exists(pool_file):
                candidates = read_pool(pool_file) # READING POOL FILE
                candidates = calc_scores(args, candidates) # CALCULATING SCORES
                sorted_candidates = sorted(candidates, key=lambda x: x["final_score"], reverse=True) # SORTING

                # Remove the duplicated RNAs
                seen = set()
                unique_candidates = []
                for cand in sorted_candidates:
                    if cand["rna"] not in seen:
                        unique_candidates.append(cand)
                        seen.add(cand["rna"])
                    if len(unique_candidates) >= args.rna_num:
                        break

                filtered_filename = os.path.join(os.path.dirname(pool_file), f"{protein_name}_filtered.fasta")
                with open(filtered_filename, "w") as f:
                    for idx, cand in enumerate(unique_candidates):
                        header = f">RNA_{idx}_{cand.get('header', 'NA')}_score_{cand['final_score']:.3f}"
                        f.write(header + "\n")
                        f.write(cand["rna"] + "\n")
            else:
                print(f"Pool file for {protein_name} not found.")

    if args.runmode == "aggregate":
        aggregate(args)
