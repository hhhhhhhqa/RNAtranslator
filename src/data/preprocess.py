from src.utils.tokenizer import BpeTokenizer

def tokenize_dataset(sample, source_tokenizer, rna_tokenizer):
    text = sample['text']
    source, rna  = text.strip().split('$')

    source_tokenized = source_tokenizer.tokenize(source)
    rna_tokenized = rna_tokenizer.tokenize(rna)
    
    # need to set these to -100 to calculate the loss properly
    rna_labels = [-100 if i == 0 else i for i in rna_tokenized.ids]

    return {
        "input_ids": source_tokenized.ids,
        "attention_mask": source_tokenized.attention_mask,
        "labels": rna_labels,
    }
