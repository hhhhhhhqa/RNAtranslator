import unittest
from datasets import Dataset


from src.utils.tokenizer import BpeTokenizer
from src.data.preprocess import tokenize_dataset


class TestBpeTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = BpeTokenizer(seq_size=512, vocab_size=100)
        self.tokenizer.load('/data6/sobhan/rllm/dataset/tokenizers/proteins/', 'bpe_100_512')

        self.rna_tokenizer = BpeTokenizer(seq_size=512, vocab_size=100)
        self.rna_tokenizer.load('/data6/sobhan/rllm/dataset/tokenizers/rnas/', 'bpe_100_512')

        self.dataset = Dataset.from_dict({'text': ['MEEPQSDPSV$BVZ']})

    def test_tokenize(self):
        protein_sequence = 'MEEPQSDPSV'
        encoded = self.tokenizer.tokenize(protein_sequence)
        self.assertIsNotNone(encoded)
        print(f'Encoded: {encoded.ids}')

    def test_decode(self):
        protein_sequence = 'MEEPQSDPSV'
        encoded = self.tokenizer.tokenize(protein_sequence)
        decoded = self.tokenizer.decode(encoded.ids)
        postprocessed_text = decoded.replace(' ', '').upper()

        self.assertEqual(postprocessed_text, protein_sequence)
        print(f'Decoded: {postprocessed_text}')

    def test_tokenize_dataset(self):
        tokenized_dataset = self.dataset.map(lambda sample: tokenize_dataset(sample, self.tokenizer, self.rna_tokenizer))
        
        print(tokenized_dataset[0])
        self.assertIsNotNone(tokenized_dataset[0]['input_ids'])
        self.assertIsNotNone(tokenized_dataset[0]['attention_mask'])
        self.assertIsNotNone(tokenized_dataset[0]['labels'])


if __name__ == '__main__':
    unittest.main()