from datasets import load_dataset
from copy import deepcopy
from src.data.preprocess import tokenize_dataset

def get_datasets(args, source_tokenizer, rna_tokenizer, iterable=True):

    # Get the dataset
    dataset = load_dataset("text", data_files=args.train_data, split="train", cache_dir="/data6/sobhan/rllm/dataset/rph/cache")

    if iterable:
        train_dataset = dataset.to_iterable_dataset()
    else:
        train_dataset = dataset

    # # Filter dataset
    filter_source_tokenizer = deepcopy(source_tokenizer)
    filter_source_tokenizer.tokenizer.no_truncation()
    filter_rna_tokenizer = deepcopy(rna_tokenizer)
    filter_rna_tokenizer.tokenizer.no_truncation()
    filtered = train_dataset.filter(lambda sample: (len(filter_source_tokenizer.tokenize(sample['text'].strip().split('$')[0]).ids) <= 1024 and len(filter_rna_tokenizer.tokenize(sample['text'].strip().split('$')[1]).ids) <= 1024))
    # # # Shuffle dataset
    if iterable:
        shuffled = filtered.shuffle(buffer_size = 10000)
    else:
        shuffled = filtered.shuffle()

    # Tokenize dataset
    tokenized = shuffled.map(lambda sample: tokenize_dataset(sample, source_tokenizer, rna_tokenizer))
    

    # TODO: Evaldataset needs to be fixed
    # Get the dataset
    eval_dataset = load_dataset("text", data_files=args.eval_data, split="train", cache_dir="/data6/sobhan/rllm/dataset/rph/cache")
    # eval_dataset = eval_dataset.remove_columns("text")
    iterable_eval_dataset = eval_dataset.to_iterable_dataset()
    # filtered_eval_dataset = iterable_eval_dataset.filter(lambda sample: (len(filter_protein_tokenizer.tokenize(sample['text'].strip().split('$')[0]).ids) <= 2048 and len(filter_rna_tokenizer.tokenize(sample['text'].strip().split('$')[1]).ids) <= 2048))
    # shuffled_eval_dataset = filtered_eval_dataset.shuffle(buffer_size = 10000)
    tokenized_eval_dataset = iterable_eval_dataset.map(lambda sample: tokenize_dataset(sample, source_tokenizer, rna_tokenizer))
    
    return tokenized, tokenized_eval_dataset


