import torch
import json
import RNA
import os

from transformers import Trainer, TrainingArguments, TrainerCallback , get_cosine_schedule_with_warmup
from tokenizers import Tokenizer
from transformers.trainer_callback import TrainerControl, TrainerState

from src.utils.validations import report_gc_content, report_token_distribution, report_mfe_distribution
from src.utils.helpers import generate_text, postprocess_rna

from concurrent.futures import ProcessPoolExecutor 


def calculate_mfe(sequence):
    _, mfe = RNA.fold(sequence)
    return mfe

def calculate_mfe_many(sequences):
    mfes = [calculate_mfe(seq) for seq in sequences]
    return mfes

def calculate_gc_content(rna_sequence):
    gc_content = ((rna_sequence.count('G') + rna_sequence.count('C')) / len(rna_sequence)) * 100
    return gc_content

def validation(dirpath, args, plots_dir, dec_tokenizer, step):
    exists = True
    index = 0
    while exists:
        seq_path = dirpath + f'_{str(index)}'
        checkpoint_path = dirpath + f'checkpoint_{index}'
        index += step
        if os.path.exists(seq_path):
            model = model.from_pretrained().to(args.device)
            seqs = generate_from_checkpoint(checkpoint_path, )
            with open(seq_path,'r') as f:
                seqs = f.readlines()
            seqs = [seq.replace('\n','') for seq in seqs]
            generated_sequences = seqs

            validation_results_gc = report_gc_content(generated_sequences, plots_dir, step=step, save=True)
            print('GC Content Comparison Completed ...')
            validation_results_token = report_token_distribution(generated_sequences, tokenizer=dec_tokenizer, dir=plots_dir, step=step, save=True)
            print('Token Distribution Comparison Completed ...')
            validation_results_mfe = report_mfe_distribution(generated_sequences, plots_dir, step=step, save=True)
            print('MFE Distribution Comparison Completed ...')

        else:
            exists = False
        

def generate_from_checkpoint(model, args, eval_dataset, dec_tokenizer):

    print('Evaluation Begins ...')
    
    # Custom validation logic
    natural_sequences = []
    generated_sequences = []

    eval_dataset_iterator = iter(eval_dataset)
    
    for i in range(args.validation_sample_number):

        data_point = next(eval_dataset_iterator)
        print(data_point)
        protein = data_point["input_ids"]
        natural_rna = data_point["labels"]

        inputs = torch.tensor(protein, dtype=torch.long).unsqueeze(0).to(args.device)
        rna = model.generate(inputs, max_length=100)
        
        rna = dec_tokenizer.decode(rna[0].cpu().numpy().tolist())
        # print(rna)
        generated_sequences.append(postprocess_rna(rna))

        natural_rna = [0 if i == -100 else i for i in natural_rna]
        decoded_natural_rna = dec_tokenizer.decode(natural_rna)
        natural_rna = postprocess_rna(decoded_natural_rna)
        natural_sequences.append(natural_rna)

    return
    

def my_wrapper(protein, rna):
    rna = postprocess_rna(rna)
    mfe = calculate_mfe(rna)
    gc_content = calculate_gc_content(rna)
    return (protein, mfe, gc_content)

def eval_natural_rna(dataset_path, results_dir):
    results = {}
    with open(dataset_path, 'r') as file:
        with ProcessPoolExecutor() as executor:
            futures = []
            co = 0
            for line in file:
                protein, rna = line.strip().split("$")
                futures.append(executor.submit(my_wrapper, protein, rna))
                co += 1
                if co == 50:
                    break
            
            for future in futures:
                protein, mfe, gc_content = future.result()
                if protein not in results:
                    results[protein] = {'mfes': [], 'gc_contents': []}
                results[protein]['mfes'].append(mfe)
                results[protein]['gc_contents'].append(gc_content)
    # np.save(results_dir, results)
    print(results)


eval_natural_rna("/data6/sobhan/rllm/dataset/rph/eval_rp.txt", "./")
rna = postprocess_rna("JJBUZJJBBZBZZBBZJJJBZJJJJUUJBUZUUUZBBBZBZZZZBUUZBBUZZBBUUZJUZUZUUUJJZZBBZBBJZUZJBUJUUZJBBZBBZJJJUZZBBUJUZJUZJJJZZJBJBZBBUZUJZZUJBZZZUUZBBJJJJUJUJUJUJJZZJUUUZJUUZZJUUUJJUUUJBJJBJJUZUUBJJUJBUUUJBUUJBZBBUUJJUZUUZZUUUJUZJZJUZZBBUJJJUZBUUUUJJUJBBZUZUZZBBBBJBJBZJBJUZBUZBZBUBJJBZUUZJUUUUJBZJZZUBBBJUZBJBZBJUUZUBB")
print(calculate_mfe(rna))