from matplotlib import colors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
from scipy.stats import ks_2samp,kstest,ttest_ind, mannwhitneyu, norm
import seaborn as sns
from tqdm import tqdm
import random
random.seed(1337)
import os
import numpy as np
np.random.seed(1337)
import pandas as pd
pd.options.mode.chained_assignment = None
import RNA
import time
import itertools
from src.utils.tokenizer import get_tokenizer


colors = ["#3c5068", "#acbab6", "#dcd3cd", "#d4a6a6"]

customPalette = {'Generated':colors[0],'Random':colors[3],'Natural':colors[1]}

UTR_LEN = 128
Z_DIM = 40
DIM = Z_DIM
BATCH_SIZE = 2048
MAX_LEN = UTR_LEN
gpath = './../models/checkpoint_3000.h5'
data_path = './../data/utrdb2.csv'
mrl_path = './../models/utr_model_combined_residual_new.h5'

sns.set()
sns.set_style('ticks')

#POSTER
params = {'legend.fontsize': 48,
        'figure.figsize': (54, 32),
        'axes.labelsize': 60,
        'axes.titlesize':60,
        'xtick.labelsize':60,
        'ytick.labelsize':60}

plt.rcParams.update(params)

count = 0
natural_samples = []

def calc_mfe_all(seqs):
    randpreds = []
    for i in range(len(seqs)):
        (ss, mfe) = RNA.fold(seqs[i])
        randpreds.append(mfe)

def calc_mfe(seq):
    _, mfe = RNA.fold(seq)
    return mfe

def gc_percentage(seq):
    count = 0.0
    for char in seq:
        if char == 'C' or char == 'G':
            count +=1

    return float(count/len(seq))

def get_gc_content(data):
    gc_content = []
    for seq in data:
        seq.replace('\n','')
        seq.replace('*','')
        gc = gc_percentage(seq)
        gc_content.append(gc)

    return gc_content

rna_tokenizer = get_tokenizer(tokenizer_name="bpe", vocab_size=1000, seq_size=2048, tokenizer_path='/data6/sobhan/rllm/dataset/tokenizers/rnas/bpe_1000_2048.json')


def get_token_dist(seq,tokenizer=rna_tokenizer):
    inputs = [tokenizer.encode(seq).ids]
    
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

#################### METRIC DISTRIBUTION ######################
save_dir = '/data6/sobhan/rllm/dataset/rph/'
# can be 'MFE', 'GC' or 'TOKEN'
METRIC = 'MFE'
SAMPLE = True
SAMPLE_SIZE = 10
DATA_SIZE = 100000

def calculate_metric(metric,path):
    count = 0
    nat_values = []
    sampled_count = 0
    with open(path,'r') as f:
        while True:
            count += 1
            line = f.readline()
            # print(line)
            if len(line)>0 and count>0 and count%(DATA_SIZE/SAMPLE_SIZE):
                line = line.split('$')[1].replace('\n','')
                print(count)

                if metric == 'MFE':
                    nat_values.append(calc_mfe(line))
                if metric == 'GC':
                    nat_values.append(get_gc_content(line))
                if metric == 'TOKEN':
                    nat_values.append(get_token_dist(line))

            if not line:
                break
            # print("Line{}: {}".format(count, line.strip()))

    return nat_values

file_path = '/data6/sobhan/rllm/dataset/rph/train_rp.txt'


start = time.time()
nat_values = calculate_metric(METRIC,file_path)
end = time.time()


save_path = save_dir + file_path.split('/')[-1].split('.')[0] + f'_{METRIC}_values.npy'

with open(save_path, 'wb') as f:
    np.save(f, nat_values)

print(f'Processing Time: {end-start}')

# gent_te = ttest_ind(genpreds, realpreds)
# randt_te = ttest_ind(randpreds, realpreds)
# genu_te = mannwhitneyu(genpreds, realpreds)
# randu_te = mannwhitneyu(randpreds, realpreds)
