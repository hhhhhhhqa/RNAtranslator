import json
import os
import numpy as np


from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import Sequence, Lowercase

import torch


class BpeTokenizer:
    def __init__(self, seq_size, vocab_size):
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.normalizer = Sequence([Lowercase()])
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

        self.tokenizer.enable_padding(max_length=seq_size, direction='right')
        self.tokenizer.enable_truncation(max_length=seq_size)

        self.special_tokens = {
            "pad": {"id": 0, "token": "<pad>"},
            "eos": {"id": 1, "token": "</s>"},
            "unk": {"id": 2, "token": "<unk>"},
        }


        self.special_tokens_list = [None] * len(self.special_tokens)
        for token_dict in self.special_tokens.values():
            self.special_tokens_list[token_dict["id"]] = token_dict["token"]

        self.tokenizer.post_processor = TemplateProcessing(
            single=f"$A {self.special_tokens['eos']['token']}",
            special_tokens=[
                (self.special_tokens["eos"]["token"], self.special_tokens["eos"]["id"]),
            ],
        )

        self.trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=self.special_tokens_list,
            show_progress=True
        )

    def train_tokenizer(self, train_data, which=True):
        def iterator(data, which):
            for sequence in data:
                text = sequence['text']
                mol, rna  = text.strip().split('$')
                if which:
                    yield mol
                else:
                    yield rna
        self.tokenizer.train_from_iterator(iterator(data=train_data, which=which), trainer=self.trainer)
        self.add_unk_id()

    def train_from_files(self, data_files: str):
        self.tokenizer.train(files=[data_files], trainer=self.trainer)
        self.add_unk_id()

    def add_unk_id(self):
        tokenizer_json = json.loads(self.tokenizer.to_str())
        tokenizer_json["model"]["unk_id"] = self.special_tokens["unk"]["id"]
        self.tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))

    def save(self, path: str, name: str):
        if not os.path.exists(path):
            os.makedirs(path)
        self.tokenizer.save(os.path.join(path, f"{name}.json"))

    def load(self, path: str):
        self.tokenizer = Tokenizer.from_file(path)

    def tokenize(self, sequence):
        # print(sequence)
        return self.tokenizer.encode(sequence)

    def decode(self, sequence):
        # print(sequence)
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.detach().cpu().numpy().astype(dtype=np.int64)
        return self.tokenizer.decode(sequence)
    
    def encode(self, tokenized):
        
        return [self.tokenizer.encode(seq) for seq in tokenized]


def get_tokenizer(tokenizer_name:str, vocab_size:int, seq_size:int, tokenizer_path:str=None):
    # Choose tokenizer
    if tokenizer_name=="bpe":
        my_tokenizer = BpeTokenizer(vocab_size=vocab_size, seq_size=seq_size)
    else:
        raise NotImplementedError
    
    # Load pre-trained tokenizer or train tokenizer
    if tokenizer_path:
        my_tokenizer.load(tokenizer_path)

    if vocab_size != my_tokenizer.tokenizer.get_vocab_size():
        assert "There is a conflict Tokenizer's vocab size and arguments'"

    return my_tokenizer
    


