import argparse
import os
# import wandb
from datetime import datetime
import pytz
import json

from sympy import true
import torch.nn.functional as F


# %load_ext autoreload
# # %autoreload 2
from src.models import get_model
from src.utils.helpers import set_hyps, top_k_top_p_filtering, generate_text
from src.utils.tokenizer import get_tokenizer
from src.data import get_datasets
from src.utils.tokenizer import BpeTokenizer
from src.utils.helpers import generate_text, postprocess_rna, read_rna_from_fasta, get_random_rna, read_rna_from_text
import accelerate

import torch


def generate(args:object, eval_dataset, model, dec_tokenizer)->None:
    args.model_size = sum(p.numel() for p in model.parameters())
    print("Model Size: ", sum(p.numel() for p in model.parameters()))
    # print(model)

    model = model.from_pretrained(args.checkpoints).to(args.device)
    model.eval()
    eval_dataset_iterator = iter(eval_dataset)

    for i in range(10):
        data_point = next(eval_dataset_iterator)
        print(data_point)
        protein = data_point["input_ids"]
        natural_rna = data_point["labels"]
        print("RNA: ", data_point["labels"])

        inputs = torch.tensor(protein, dtype=torch.long).unsqueeze(0).to(args.device)
        rna = model.generate(torch.tensor(data_point['input_ids']).unsqueeze(0).to(model.device), max_length=50)
        print("Genrated RNA: ", rna)
        
        rna = dec_tokenizer.decode(rna[0].cpu().numpy().tolist())
        # print(rna)
        natural_rna = [0 if i == -100 else i for i in natural_rna]
        decoded_natural_rna = dec_tokenizer.decode(natural_rna)
        natural_rna = postprocess_rna(decoded_natural_rna)




# args = parse_opt()
# generate(args)


def generate_single(args: object, model, enc_tokenizer, dec_tokenizer, num_rnas, result_dir) -> None:
    args.model_size = sum(p.numel() for p in model.parameters())
    print("Model Size: ", args.model_size)

    # ELAVL1
    # protein = "MSNGYEDHMAEDCRGDIGRTNLIVNYLPQNMTQDELRSLFSSIGEVESAKLIRDKVAGHSLGYGFVNYVTAKDAERAINTLNGLRLQSKTIKVSYARPSSEVIKDANLYISGLPRTMTQKDVEDMFSRFGRIINSRVLVDQTTGLSRGVAFIRFDKRSEAEEAITSFNGHKPPGSSEPITVKFAANPNQNKNVALLSQLYHSPARRFGGPVHHQAQRFRFSPMGVDHMSGLSGVNVPGNASSGWCIFIYNLGQDADEGILWQMFGPFGAVTNVKVIRDFNTNKCKGFGFVTMTNYEEAAMAIASLNGYRLGDKILQVSFKTNKSHK"
    #RBM45
    #protein = "MDEAGSSASGGGFRPGVDSLDEPPNSRIFLVISKYTPESVLRERFSPFGDIQDIWVVRDKHTKESKGIAFVKFARSSQACRAMEEMHGQCLGPNDTKPIKVFIAQSRSSGSHRDVEDEELTRIFVMIPKSYTEEDLREKFKVYGDIEYCSIIKNKVTGESKGLGYVRYLKPSQAAQAIENCDRSFRAILAEPKNKASESSEQDYYSNMRQEALGHEPRVNMFPFVGEQQSEFSSFDKNDSRGQEAISKRLSVVSRVPFTEEQLFSIFDIVPGLEYCEVQRDPYSNYGHGVVQYFNVASAIYAKYKLHGFQYPPGNRIGVSFIDDGSNATDLLRKMATQMVAAQLASMVWNNPSQQQFMQFGGSSGSQLPQIQTDVVLPSCKKKAPAETPVKERLFIVFNPHPLPLDVLEDIFCRFGNLIEVYLVSGKNVGYAKYADRISANDAIATLHGKILNGVRLKVMLADSPREESNKRQRTY"
    #RBM5
    protein = "MGSDKRVSRTERSGRYGSIIDRDDRDERESRSRRRDSDYKRSSDDRRGDRYDDYRDYDSPERERERRNSDRSEDGYHSDGDYGEHDYRHDISDERESKTIMLRGLPITITESDIREMMESFEGPQPADVRLMKRKTGVSRGFAFVEFYHLQDATSWMEANQKKLVIQGKHIAMHYSNPRPKFEDWLCNKCCLNNFRKRLKCFRCGADKFDSEQEVPPGTTESVQSVDYYCDTIILRNIAPHTVVDSIMTALSPYASLAVNNIRLIKDKQTQQNRGFAFVQLSSAMDASQLLQILQSLHPPLKIDGKTIGVDFAKSARKDLVLSDGNRVSAFSVASTAIAAAQWSSTQSQSGEGGSVDYSYLQPGQDGYAQYAQYSQDYQQFYQQQAGGLESDASSASGTAVTTTSAAVVSQSPQLYNQTSNPPGSPTEEAQPSTSTSTQAPAASPTGVVPGTKYAVPDTSTYQYDESSGYYYDPTTGLYYDPNSQYYYNSLTQQYLYWDGEKETYVPAAESSSHQQSGLPPAKEGKEKKEKPKSKTAQQIAKDMERWAKSLNKQKENFKNSFQPVNSLREEERRESAAADAGFALFEKKGALAERQQLIPELVRNGDEENPLKRGLVAAYSGDSDNEEELVERLESEEEKLADWKKMACLLCRRQFPNKDALVRHQQLSDLHKQNMDIYRRSRLSEQELEALELREREMKYRDRAAERREKYGIPEPPEPKRKKQFDAGTVNYEQPTKDGIDHSNIGNKMLQAMGWREGSGLGRKCQGITAPIEAQVRLKGAGLGAKGSAYGLSGADSYKDAVRKAMFARFTEME"
    protein_ids = enc_tokenizer.tokenize(protein).ids

    model = model.from_pretrained(args.checkpoints).to(args.device)
    model.eval()

    generated_rnas = []
    random_rnas = []

    natural_binding_rnas = read_rna_from_fasta("/data6/bilginer/CLIP_CLUSTERED_FASTA/RBM5_clustered.fa")
    natural_rnas = read_rna_from_fasta("/data6/sobhan/RLLM/dataset/rph/natural/natural_rnas.fna")
    rnagen_rnas = read_rna_from_text("/data6/sobhan/RNAGEN/output/RBM5_inv_distance_softmax_method_maxiters_3000_v1/RBM5_best_binding_sequences.txt")

    # Write in the output rnas.fasta file
    with open(result_dir + "/rnas.fasta", "w") as fasta_file:
        i = 0
        j=0
        while j < num_rnas:
            natural_len = len(natural_binding_rnas[i])
            if natural_len < 70:
                fasta_file.write(f">Binding Natural RNA {j+1}\n{natural_binding_rnas[i]}\n")
                
                if len(natural_rnas[i]) > 70:
                    fasta_file.write(f">Natural RNA {j+1}\n{natural_rnas[j][:natural_len]}\n")
                else:
                    fasta_file.write(f">Natural RNA {j+1}\n{natural_rnas[j]}\n")

                # random_rna = get_random_rna(natural_len)
                # random_rnas.append(random_rna)
                fasta_file.write(f">RNAGEN RNA {(2*j)+1}\n{rnagen_rnas[(2*j)+1][:natural_len]}\n")
                fasta_file.write(f">RNAGEN RNA {(2*j)}\n{rnagen_rnas[(2*j)][:natural_len]}\n")

                inputs = torch.tensor(protein_ids, dtype=torch.long).unsqueeze(0).to(model.device)

                for k in range(2):
                    rna = model.generate(
                        inputs, 
                        max_length=70, 
                        min_length = 5,
                        temperature=1, 
                        do_sample=True, 
                        num_beams=1,        
                        # num_beam_groups=3,
                        # diversity_penalty=0.6, 
                        repetition_penalty=1.5, 
                        encoder_repetition_penalty=1.3
                    )

                    generated_rna = dec_tokenizer.decode(rna[0].cpu().numpy().tolist())
                    generated_rna = postprocess_rna(generated_rna)[:natural_len]
                    generated_rnas.append(generated_rna)

                    fasta_file.write(f">Generated RNA {j+1}{k}\n{generated_rna}\n")
                # print(f"Generated RNA {i+1}: ", generated_rna, "generated size: ", len(generated_rna), "natural size: ", natural_len)
                # print(i)
                print(j)
                j+=1
            else:
                print("Ignored!!")
            i+=1
# args = parse_opt()
# generate(args)

# def eval_single(args: object, model, enc_tokenizer, dec_tokenizer, result_dir, generate=False) -> None:
#     args.model_size = sum(p.numel() for p in model.parameters())
#     print("Model Size: ", args.model_size)

#     protein = "MSNGYEDHMAEDCRGDIGRTNLIVNYLPQNMTQDELRSLFSSIGEVESAKLIRDKVAGHSLGYGFVNYVTAKDAERAINTLNGLRLQSKTIKVSYARPSSEVIKDANLYISGLPRTMTQKDVEDMFSRFGRIINSRVLVDQTTGLSRGVAFIRFDKRSEAEEAITSFNGHKPPGSSEPITVKFAANPNQNKNVALLSQLYHSPARRFGGPVHHQAQRFRFSPMGVDHMSGLSGVNVPGNASSGWCIFIYNLGQDADEGILWQMFGPFGAVTNVKVIRDFNTNKCKGFGFVTMTNYEEAAMAIASLNGYRLGDKILQVSFKTNKSHK"
#     protein_ids = enc_tokenizer.tokenize(protein).ids

#     model = model.from_pretrained(args.checkpoints).to(args.device)
#     model.eval()

#     # natural_rnas = []
#     generated_rnas = []
#     random_rnas = []

#     # Read RNAs from Fasta files
#     natural_binding_rnas = read_rna_from_fasta("/data6/bilginer/CLIP_CLUSTERED_FASTA/ELAVL1_clustered.fa")
#     natural_rnas = read_rna_from_fasta("/data6/sobhan/RLLM/dataset/rph/natural/natural_rnas.fna")
#     my_rnas = read_rna_from_fasta("/data6/sobhan/RLLM/results/validation/second/my_rnas.fasta")

#     # Write in the output rnas.fasta file
#     with open(result_dir + "/rnas.fasta", "w") as fasta_file:
#         for i in range(2555):
#             if len(natural_binding_rnas[i]) < 70:
#                 fasta_file.write(f">Binding Natural RNA {i+1}\n{natural_binding_rnas[i]}\n")
                
#                 if len(natural_rnas[i]) > 70:
#                     fasta_file.write(f">Natural RNA {i+1}\n{natural_rnas[i][:70]}\n")
#                 else:
#                     fasta_file.write(f">Natural RNA {i+1}\n{natural_rnas[i]}\n")


#                 random_rna = get_random_rna(len(natural_binding_rnas[i]))
#                 random_rnas.append(random_rna)
#                 fasta_file.write(f">Random RNA {i+1}\n{random_rna}\n")

#                 if generate:
#                     inputs = torch.tensor(protein_ids, dtype=torch.long).unsqueeze(0).to(model.device)
#                     rna = model.generate(
#                         inputs, 
#                         max_length=len(natural_binding_rnas[i])+50, 
#                         min_length = 60,
#                         temperature=1, 
#                         do_sample=True, 
#                         num_beams=1,        
#                         # num_beam_groups=3,
#                         # diversity_penalty=0.6, 
#                         repetition_penalty=1.5, 
#                         encoder_repetition_penalty=1.3
#                     )

#                     generated_rna = dec_tokenizer.decode(rna[0].cpu().numpy().tolist())
#                     generated_rna = postprocess_rna(generated_rna)
#                     if len(generated_rna[50:]) < 70:
#                         generated_rnas.append(generated_rna[50:])

#                         fasta_file.write(f">Generated RNA {i+1}\n{generated_rna[50:]}\n")
#                         print(f"Generated RNA {i+1}: ", generated_rna[50:])
#                     else:
#                         print("Long RNA ignored, size was: ", len(generated_rna[50:]))
#                         generated_rnas.append(generated_rna[50:120])

#                         fasta_file.write(f">Generated RNA {i+1}\n{generated_rna[50:120]}\n")
#                         print(f"Generated RNA {i+1}: ", generated_rna[50:120])
#                 else:
#                     fasta_file.write(f">Generated RNA {i+1}\n{my_rnas[i]}\n")
#                     generated_rnas.append(my_rnas[i])


#     compare_gc_content(natural_binding_rnas, generated_rnas, natural_rnas, random_rnas, result_dir, 1)
#     compare_mfe_distribution(natural_binding_rnas, generated_rnas, natural_rnas, random_rnas, result_dir, 1)

