from datasets import load_dataset
from copy import deepcopy
from src.data.preprocess import tokenize_dataset

def get_datasets(args, protein_tokenizer, rna_tokenizer):

    # Get the dataset
    dataset = load_dataset("text", data_files=args.train_data, split="train", cache_dir="/data6/sobhan/rllm/dataset/rph/cache")

    iterable_dataset = dataset.to_iterable_dataset()
    # # Filter dataset
    filter_protein_tokenizer = deepcopy(protein_tokenizer)
    filter_protein_tokenizer.tokenizer.no_truncation()
    filter_rna_tokenizer = deepcopy(rna_tokenizer)
    filter_rna_tokenizer.tokenizer.no_truncation()
    filtered = iterable_dataset.filter(lambda sample: (len(filter_protein_tokenizer.tokenize(sample['text'].strip().split('$')[0]).ids) <= 1024 and len(filter_rna_tokenizer.tokenize(sample['text'].strip().split('$')[1]).ids) <= 1024))
    # # # Shuffle dataset
    shuffled = filtered.shuffle(buffer_size = 10000)
    # Tokenize dataset
    tokenized = shuffled.map(lambda sample: tokenize_dataset(sample, protein_tokenizer, rna_tokenizer))
    

    # Get the dataset
    eval_dataset = load_dataset("text", data_files=args.eval_data, split="train", cache_dir="/data6/sobhan/rllm/dataset/rph/cache")
    # eval_dataset = eval_dataset.remove_columns("text")
    iterable_eval_dataset = eval_dataset.to_iterable_dataset()
    # filtered_eval_dataset = iterable_eval_dataset.filter(lambda sample: (len(filter_protein_tokenizer.tokenize(sample['text'].strip().split('$')[0]).ids) <= 2048 and len(filter_rna_tokenizer.tokenize(sample['text'].strip().split('$')[1]).ids) <= 2048))
    # shuffled_eval_dataset = filtered_eval_dataset.shuffle(buffer_size = 10000)
    tokenized_eval_dataset = iterable_eval_dataset.map(lambda sample: tokenize_dataset(sample, protein_tokenizer, rna_tokenizer))
    
    return tokenized, tokenized_eval_dataset


