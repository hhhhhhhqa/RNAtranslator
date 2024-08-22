import numpy as np
from concurrent.futures import ProcessPoolExecutor

from src.utils.validations import calculate_mfe, calculate_gc_content
from src.utils.helpers import generate_text, postprocess_rna

import torch

def eval(model, args, eval_dataset, dec_tokenizer):

    print('Evaluation Begins ...')
    
    # Custom validation logic
    natural_sequences = []
    generated_sequences = []

    eval_dataset_iterator = iter(eval_dataset)
    
    for i in range(self.args.validation_sample_number):

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