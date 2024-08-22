import unittest
from src.data import get_datasets
from src.utils.tokenizer import get_tokenizer
from src.utils.helpers import postprocess_rna

class TestDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        class Args:
            train_data = "/data6/sobhan/rllm/dataset/rph/train_rp.txt"
            eval_data = "/data6/sobhan/rllm/dataset/rph/test_rp.txt"
            protein_tokenizer= "/data6/sobhan/rllm/dataset/tokenizers/proteins/bpe_1000_1024.json"
            rna_tokenizer= "/data6/sobhan/rllm/dataset/tokenizers/rnas/bpe_1000_2048.json"
        
        cls.args = Args()
        protein_tokenizer = get_tokenizer(tokenizer_name="bpe", vocab_size=1000, seq_size=2048, tokenizer_path=cls.args.protein_tokenizer)
        cls.p_tokenizer = protein_tokenizer

        rna_tokenizer = get_tokenizer(tokenizer_name="bpe", vocab_size=1000, seq_size=2048, tokenizer_path=cls.args.rna_tokenizer)
        cls.r_tokenizer = rna_tokenizer

        cls.train_dataset, _ = get_datasets(cls.args, protein_tokenizer=protein_tokenizer, rna_tokenizer=rna_tokenizer)


    def format_and_sizes_wtrunctation(self):
        protein_tokenizer = get_tokenizer(tokenizer_name="bpe", vocab_size=1000, seq_size=2048, tokenizer_path=self.args.protein_tokenizer)
        protein_tokenizer.tokenizer.no_truncation()
        rna_tokenizer = get_tokenizer(tokenizer_name="bpe", vocab_size=1000, seq_size=2048, tokenizer_path=self.args.rna_tokenizer)
        rna_tokenizer.tokenizer.no_truncation()
        train_dataset, _ = get_datasets(self.args, protein_tokenizer=protein_tokenizer, rna_tokenizer=rna_tokenizer)

        # Iterate over the dataset and check the format and sizes
        for data in train_dataset:
            self.assertIn('input_ids', data)
            self.assertIn('attention_mask', data)
            self.assertIn('labels', data)

            input_ids = data['input_ids']
            labels = data['labels']

            self.assertLessEqual(len(input_ids), 2048)
            self.assertLessEqual(len(labels), 2048)

    def format_and_sizes(self):
        # Iterate over the dataset and check the format and sizes
        for data in self.train_dataset:
            self.assertIn('input_ids', data)
            self.assertIn('attention_mask', data)
            self.assertIn('labels', data)

            input_ids = data['input_ids']
            labels = data['labels']

            self.assertEqual(len(input_ids), 2048)
            self.assertEqual(len(labels), 2048)

    def test_sample_decoding(self):
        # Sample some sequences and print decoded sequences
        for data in list(iter(self.train_dataset))[:5]:  # Print first 5 items for verification
            decoded = self.p_tokenizer.decode(data['input_ids'])
            print(f"Decoded Protein: {decoded.replace(' ', '').upper()}")
            rna_labels = [0 if i == -100 else i for i in data['labels']]
            decoded = self.r_tokenizer.decode(rna_labels)
            print(f"Decoded RNA: {postprocess_rna(decoded)}")

    def test_data_shuffling(self):
        # Check if the data is shuffled
        initial_sequences = [data['input_ids'] for data in list(iter(self.train_dataset))[:10]]  # First 10 sequences
        shuffled_dataset = list(iter(self.train_dataset))[:]
        import random
        random.seed(42)
        random.shuffle(shuffled_dataset)
        shuffled_sequences = [data['input_ids'] for data in shuffled_dataset[:10]]
        
        self.assertNotEqual(initial_sequences, shuffled_sequences)

if __name__ == '__main__':
    unittest.main()
