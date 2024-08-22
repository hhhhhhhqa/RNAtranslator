from src.utils.tokenizer import BpeTokenizer

def tokenize_dataset(sample, protein_tokenizer, rna_tokenizer):
    text = sample['text']
    protein, rna  = text.strip().split('$')

    protein_tokenized = protein_tokenizer.tokenize(protein)
    rna_tokenized = rna_tokenizer.tokenize(rna)
    
    # need to set these to -100 to calculate the loss properly
    rna_labels = [-100 if i == 0 else i for i in rna_tokenized.ids]

    return {
        "input_ids": protein_tokenized.ids,
        "attention_mask": protein_tokenized.attention_mask,
        "labels": rna_labels,
    }
