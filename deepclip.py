import argparse
import os
import sys
import time
import random
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt 


VOCAB = ['a', 'c', 'g', 'tu']  # Note: 'tu' here stands for T/U.
VOCAB_SIZE = len(VOCAB)  # Should be 4

FILTER_SIZES = [4, 5, 6, 7, 8]
MINI_BATCH_SIZE = 24
GRAD_CLIP = 100
DROPOUT_CONV = 0.0    # not explicitly used in this replication
DROPOUT_IN = 0.0
DROPOUT_OUT = 0.0
PADDING = 'valid'     # only "valid" is implemented here
L2 = 0.0
LSTM_LAYERS = 1
LSTM_NODES = 10
LSTM_DROPOUT = 0.1
NUM_FILTERS = 1
LEARNING_RATE = 0.0002

#################################MODEL

def winner_takes_all(x):
    m = x.max(dim=2, keepdim=True)[0]
    return (x + m) ** 2

def shape_convolutions2(x, FS, filter_size, vocab_size, SEQ_SIZE):
    B, C, L = x.shape  # C is 1, L == FS
    # replicate x filter_size times along channel dimension and permute:
    x_rep = x.repeat(1, filter_size, 1)   # shape (B, filter_size, L)
    x_rep = x_rep.permute(0, 2, 1)          # shape (B, L, filter_size)
    # reshape to (B, 1, L*filter_size)
    x_reshaped = x_rep.reshape(B, 1, L * filter_size)
    out_list = []
    total_length = vocab_size * SEQ_SIZE
    for i in range(L):
        start = i * filter_size
        end = start + filter_size
        slice_x = x_reshaped[:, :, start:end]  # shape (B, 1, filter_size)
        left_pad = i * vocab_size
        right_pad = total_length - left_pad - filter_size
        padded = F.pad(slice_x, (left_pad, right_pad), mode='constant', value=0)
        out_list.append(padded)
    out_tensor = torch.cat(out_list, dim=1)  # shape (B, L, total_length)
    return out_tensor


class ConvBranch(nn.Module):
    def __init__(self, conv, filter_size, FS):
        super(ConvBranch, self).__init__()
        self.conv = conv          # This is an nn.Module
        self.filter_size = filter_size  # int (stored as attribute, not registered as sub-module)
        self.FS = FS              # int

    def forward(self, x):
        return self.conv(x)


class DeepCLIP(nn.Module):
    def __init__(self, SEQ_SIZE, vocab=VOCAB, filter_sizes=FILTER_SIZES,
                 num_filters=NUM_FILTERS, lstm_nodes=LSTM_NODES, lstm_layers=LSTM_LAYERS,
                 dropout_in=DROPOUT_IN, dropout_out=DROPOUT_OUT, lstm_dropout=LSTM_DROPOUT):
        super(DeepCLIP, self).__init__()
        self.SEQ_SIZE = SEQ_SIZE
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.lstm_nodes = lstm_nodes
        self.lstm_layers = lstm_layers
        
        self.BS_PR_SEQ = SEQ_SIZE
        
        self.FS_list = [int(SEQ_SIZE - (fs / self.vocab_size) + 1) for fs in filter_sizes]
        self.input_dropout = nn.Dropout(p=dropout_in)
        
        self.conv_branches = nn.ModuleList()
        for i, fs in enumerate(filter_sizes):
            conv = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=fs,
                            stride=self.vocab_size, bias=False)
            nn.init.constant_(conv.weight, 0.01)
            branch = ConvBranch(conv, fs, self.FS_list[i])
            self.conv_branches.append(branch)
        
        self.num_conv_branches = len(filter_sizes)
        lstm_input_size = (self.num_conv_branches + 1) * self.vocab_size
        self.blstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_nodes,
                             num_layers=lstm_layers, batch_first=True,
                             bidirectional=True, dropout=lstm_dropout)

        self.out_dropout = nn.Dropout(p=dropout_out)
        
    def forward(self, x):
        B = x.size(0)
        l_in = x  # shape (B, 1, SEQ_SIZE*vocab_size)
        # Apply input dropout:
        x = self.input_dropout(x)
    
        branch_outputs = []  # will collect outputs (reshaped to (B, SEQ_SIZE, vocab_size))
        for branch in self.conv_branches:
            conv = branch.conv
            filter_size = branch.filter_size
            FS = branch.FS

            # Apply conv: note that stride=vocab_size makes the input (B, 1, SEQ_SIZE*vocab_size) be seen as SEQ_SIZE time steps.
            conv_out = conv(x)  # shape: (B, num_filters, L) where L = FS (typically num_filters=1)
            # If num_filters==1, then conv_out is (B, 1, FS).
            # Winner-takes-all enhancement: add the max value (along time axis) and square.
            conv_enh = winner_takes_all(conv_out)  # shape (B, 1, FS)
            # Now apply custom reshaping via shape_convolutions2:
            branch_shaped = shape_convolutions2(conv_enh, FS, filter_size, self.vocab_size, self.SEQ_SIZE)
            # branch_shaped shape: (B, FS, self.vocab_size * SEQ_SIZE)
            # Sum along axis=1 (over the FS dimension) -> (B, 1, self.vocab_size * SEQ_SIZE)
            branch_sum = branch_shaped.sum(dim=1, keepdim=True)
            # Multiply elementwise later with original input. We save this branch output.
            # Reshape to (B, SEQ_SIZE, vocab_size)
            branch_out = branch_sum.view(B, self.SEQ_SIZE, self.vocab_size)
            branch_outputs.append(branch_out)
        
        # Concatenate all branch outputs along the channel (last) dimension:
        # branch_outputs: list of (B, SEQ_SIZE, vocab_size); concatenated becomes (B, SEQ_SIZE, num_branches*vocab_size)
        if branch_outputs:
            cn_layers = torch.cat(branch_outputs, dim=2)
        else:
            cn_layers = torch.zeros(B, self.SEQ_SIZE, 0, device=x.device)
        
        # Also reshape the original input to (B, SEQ_SIZE, vocab_size)
        inp_reshaped = l_in.view(B, self.SEQ_SIZE, self.vocab_size)
        # Concatenate: final input to LSTM is (B, SEQ_SIZE, (num_branches+1)*vocab_size)
        l_lstmin = torch.cat([cn_layers, inp_reshaped], dim=2)
        
        # Pass through BLSTM:
        blstm_out, _ = self.blstm(l_lstmin)  # shape: (B, SEQ_SIZE, 2 * lstm_nodes)
        # Sum along last dimension (as in Sum_last_ax)
        h_sum = blstm_out.sum(dim=2, keepdim=True)  # shape: (B, SEQ_SIZE, 1)
        # Replicate the sum 4 times (vocab_size is 4 by default)
        h_rep = h_sum.repeat(1, 1, self.vocab_size)  # shape: (B, SEQ_SIZE, vocab_size)
        # Reshape to (B, 1, SEQ_SIZE*vocab_size)
        h_rep_flat = h_rep.view(B, 1, self.SEQ_SIZE * self.vocab_size)
        # Elementwise multiply with original l_in:
        mult = l_in * h_rep_flat  # shape: (B, 1, SEQ_SIZE*vocab_size)
        # Reshape to (B, SEQ_SIZE, vocab_size)
        mult_reshaped = mult.view(B, self.SEQ_SIZE, self.vocab_size)
        # Sum along last dimension -> (B, SEQ_SIZE)
        l_profile = mult_reshaped.sum(dim=2)
        # Apply dropout on profile
        l_profile = self.out_dropout(l_profile)
        # Finally, apply a “dense” layer with fixed weight 1.0: i.e. sum over sequence and then sigmoid.
        # (This replicates DenseLayer(l_profile, num_units=1, nonlinearity=sigmoid, W=Constant(1.0), b=None))
        profile_sum = l_profile.sum(dim=1, keepdim=True)  # sum over sequence positions, shape: (B, 1)
        output = torch.sigmoid(profile_sum)  # final prediction between 0 and 1
        
        return output  # shape: (B, 1)


###################################DATA

def onehot_encode(seqs, vocab=VOCAB):
    """
    Given a list of sequences (strings), produce a numpy array of one–hot encoded sequences.
    Each base is one-hot encoded into a vector of length len(vocab). The sequences are assumed to
    be padded to equal length.
    The output shape is (num_seqs, 1, len(seq)*len(vocab)) (flattened along the bases).
    """
    num_seqs = len(seqs)
    if num_seqs == 0:
        raise ValueError("No sequences provided!")
    seq_len = len(seqs[0])
    # create a mapping from letter to index (case insensitive)
    mapping = {}
    for i, char in enumerate(vocab):
        mapping[char.lower()] = i
    onehot = np.zeros((num_seqs, seq_len, len(vocab)), dtype=np.float32)
    for i, seq in enumerate(seqs):
        for j, base in enumerate(seq):
            # check each vocabulary entry (we check if base is in the vocab string)
            for k, v in enumerate(vocab):
                if base.lower() in v.lower():
                    onehot[i, j, k] = 1.0
                    break
    # flatten the last two dimensions as in the original: (num_seqs, 1, seq_len * len(vocab))
    onehot = onehot.reshape(num_seqs, 1, seq_len * len(vocab))
    return onehot


def read_fasta_file(fasta_file, min_length=1, max_length=400):
    seqs = []
    ids = []
    with open(fasta_file, "r") as f:
        seq = ""
        id_line = None
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if id_line is not None and seq:
                    if len(seq) >= min_length and len(seq) <= max_length:
                        seqs.append(seq.lower())
                        ids.append(id_line)
                id_line = line[1:]
                seq = ""
            else:
                seq += line
        # append last
        if id_line is not None and seq:
            if len(seq) >= min_length and len(seq) <= max_length:
                seqs.append(seq.lower())
                ids.append(id_line)
    return seqs, ids

def pad_sequences(seqs, length):
    padded = []
    for seq in seqs:
        if len(seq) < length:
            # center-pad with 'n'
            missing = length - len(seq)
            left = missing // 2
            right = missing - left
            padded_seq = "n" * left + seq + "n" * right
        else:
            padded_seq = seq[:length]
        padded.append(padded_seq)
    return padded

def split_data(seqs, ids, ratios=[0.8, 0.1, 0]):
    # Split lists into train, validation, test (lists)
    zipped = list(zip(seqs, ids))
    random.shuffle(zipped)
    seqs, ids = zip(*zipped)
    n = len(seqs)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train = (list(seqs[:n_train]), list(ids[:n_train]))
    val = (list(seqs[n_train:n_train+n_val]), list(ids[n_train:n_train+n_val]))
    test = (list(seqs[n_train+n_val:]), list(ids[n_train+n_val:]))
    return train, val, test

class SequenceDataset(Dataset):
    def __init__(self, onehot_data, targets):
        """
        onehot_data: numpy array of shape (N, 1, SEQ_SIZE*vocab_size)
        targets: numpy array of shape (N, 1) (binary labels)
        """
        self.data = torch.from_numpy(onehot_data)
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

##############################Train Helper

def train_model(model, train_loader, val_loader, num_epochs, optimizer, device, grad_clip, network_file):
    model.to(device)
    criterion = nn.BCELoss()
    best_val_auc = 0
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_targets_all = []
        train_preds_all = []
        t0 = time.time()
        for batch_data, batch_targets in train_loader:
            batch_data = batch_data.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(loss.item())
            train_targets_all.extend(batch_targets.detach().cpu().numpy().flatten().tolist())
            train_preds_all.extend(outputs.detach().cpu().numpy().flatten().tolist())
        train_loss = np.mean(train_losses)
        try:
            train_auc = roc_auc_score(train_targets_all, train_preds_all)
        except Exception:
            train_auc = 0.5

        # Validation
        model.eval()
        val_losses = []
        val_targets_all = []
        val_preds_all = []
        with torch.no_grad():
            for batch_data, batch_targets in val_loader:
                batch_data = batch_data.to(device)
                batch_targets = batch_targets.to(device)
                outputs = model(batch_data)
                loss = criterion(outputs, batch_targets)
                val_losses.append(loss.item())
                val_targets_all.extend(batch_targets.detach().cpu().numpy().flatten().tolist())
                val_preds_all.extend(outputs.detach().cpu().numpy().flatten().tolist())
        val_loss = np.mean(val_losses)
        try:
            val_auc = roc_auc_score(val_targets_all, val_preds_all)
        except Exception:
            val_auc = 0.5

        print(f"Epoch {epoch+1}/{num_epochs}  "
              f"Train Loss: {train_loss:.6f}  Train AUC: {train_auc:.4f}  "
              f"Val Loss: {val_loss:.6f}  Val AUC: {val_auc:.4f}  "
              f"Time: {time.time()-t0:.3f}s")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict()
                    # if args.network_file:
            torch.save(model.state_dict(), network_file)
            print(f"Model saved to {network_file}")
    # load best model weights
    model.load_state_dict(best_model_state)
    return model

def predict_model(model, data_loader, device):
    model.to(device)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_data, _ in data_loader:
            batch_data = batch_data.to(device)
            outputs = model(batch_data)
            all_preds.extend(outputs.detach().cpu().numpy().flatten().tolist())
    return np.array(all_preds)


################## ARGUMENTS
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="DeepCLIP: PyTorch re-implementation for predicting protein-RNA binding.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("--runmode", required=False, default="train",
                        choices=["cv", "train", "predict", "predict_long", "test"],
                        type=str, help="Operation: train/cv, predict/predict_long, or test")
    
    # Network options
    parser.add_argument("-n", "--network_file", required=False, type=str, default=None,
                        help="Path to save/load network parameters")
    parser.add_argument("--lstm_layers", required=False, type=int, default=1,
                        help="Number of LSTM layers")
    parser.add_argument("--lstm_nodes", required=False, type=int, default=10,
                        help="Number of LSTM nodes per direction")
    parser.add_argument("--lstm_dropout", required=False, type=float, default=0.1,
                        help="LSTM dropout")
    parser.add_argument("--dropout_in", required=False, type=float, default=0.0,
                        help="Input dropout")
    parser.add_argument("--dropout_out", required=False, type=float, default=0.0,
                        help="Output dropout")
    parser.add_argument("--num_filters", required=False, type=int, default=1,
                        help="Number of filters per convolution")
    parser.add_argument("--filter_sizes", required=False, nargs='+', type=int, default=[4,5,6,7,8],
                        help="List of convolutional filter sizes (bp)")
    parser.add_argument("--learning_rate", required=False, type=float, default=0.0002,
                        help="Learning rate")
    parser.add_argument("--l2", required=False, type=float, default=0.0,
                        help="L2 regularization (not used explicitly here)")
    parser.add_argument("--early_stopping", required=False, type=int, default=10,
                        help="Early stopping patience (not implemented exactly as original)")
    
    # Training options
    parser.add_argument("-e", "--num_epochs", required=False, type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", required=False, type=int, default=24,
                        help="Batch size")
    parser.add_argument("--random_seed", required=False, type=int, default=1234,
                        help="Random seed")
    
    # Input/Output file options
    parser.add_argument("--sequences", required=False, type=str, default=None,
                        help="FASTA file with input sequences")
    parser.add_argument("--background_sequences", required=False, type=str, default=None,
                        help="FASTA file with background sequences")
    # Only one definition for test output file:
    parser.add_argument("--test_output_file", required=False, type=str, default="",
                        help="File to write test AUROC and accuracy results (JSON)")
    parser.add_argument("--predict_output_file", required=False, type=str, default="",
                        help="File to write prediction results (JSON/TSV)")
    
    # Additional options for test runmode:
    parser.add_argument("--test_positive", required=False, type=str, default=None,
                        help="FASTA file with positive test sequences")
    parser.add_argument("--test_negative", required=False, type=str, default=None,
                        help="FASTA file with negative test sequences")
    parser.add_argument("--plot_file", required=False, type=str, default="",
                        help="File to save the ROC plot (if not provided, the plot is shown)")
    
    # Option to force maximum input length (padding)
    parser.add_argument("--force_max_length", required=False, type=int, default=75,
                        help="Force maximum length of sequences")
    
    return parser.parse_args()


#############MAIN

def main():
    args = parse_arguments()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # For this replication we assume that if sequences file is provided,
    # it contains positive (binding) sequences; if background_sequences is provided,
    # they are used as negatives.
    if args.runmode in ["train", "cv"]:
        if args.sequences is None or args.background_sequences is None:
            sys.exit("For training mode, please provide both --sequences and --background_sequences FASTA files.")
        
        # Read sequences from FASTA
        pos_seqs, pos_ids = read_fasta_file(args.sequences)
        neg_seqs, neg_ids = read_fasta_file(args.background_sequences)
        
        # Optionally force max length
        max_len = max(len(s) for s in pos_seqs)
        if args.force_max_length > 0:
            max_len = args.force_max_length
        pos_seqs = pad_sequences(pos_seqs, max_len)
        neg_seqs = pad_sequences(neg_seqs, max_len)
        
        # Balance input if needed (here we simply take the minimum count)
        n = min(len(pos_seqs), len(neg_seqs))
        pos_seqs = pos_seqs[:n]
        neg_seqs = neg_seqs[:n]
        
        # One-hot encode sequences. The original code uses onehot_binary (which combines negative and positive).
        pos_X = onehot_encode(pos_seqs)
        neg_X = onehot_encode(neg_seqs)
        pos_y = np.ones((len(pos_seqs), 1), dtype=np.float32)
        neg_y = np.zeros((len(neg_seqs), 1), dtype=np.float32)
        
        # For training, we assume [negative, positive] order as in original code
        all_X = np.concatenate([pos_X, neg_X], axis=0)
        all_y = np.concatenate([pos_y, neg_y], axis=0)
        
        # Split data into train, validation, test
        (train_seqs, train_ids), (val_seqs, val_ids), (test_seqs, test_ids) = split_data(list(all_X), list(all_y))

        print(len(train_seqs), len(val_seqs), len(test_seqs))

        # Save the splits to separate FASTA files
        # write_fasta_file("./RBM5/train.fasta", train_seqs, train_ids)
        # write_fasta_file("./RBM5/val.fasta", val_seqs, val_ids)
        # write_fasta_file("./RBM5test.fasta", test_seqs, test_ids)

        print("Train, validation, and test FASTA files have been saved.")

        # Here our split is performed on numpy arrays converted to lists;
        # for simplicity, we then convert back to numpy arrays.
        train_X = np.array(train_seqs)
        train_y = np.array(train_ids, dtype=np.float32).reshape(-1, 1)  # note: here train_ids holds labels actually
        val_X = np.array(val_seqs)
        val_y = np.array(val_ids, dtype=np.float32).reshape(-1, 1)
        test_X = np.array(test_seqs)
        test_y = np.array(test_ids, dtype=np.float32).reshape(-1, 1)
        # (In a full replication, the splitting of positive/negative labels should be kept separate.
        # Here we simply use the order from concatenation.)
        
        # Create Datasets and DataLoaders
        train_dataset = SequenceDataset(train_X, train_y)
        val_dataset = SequenceDataset(val_X, val_y)
        test_dataset = SequenceDataset(test_X, test_y)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Build the network. SEQ_SIZE is max_len.
        model = DeepCLIP(SEQ_SIZE=max_len, vocab=VOCAB, filter_sizes=args.filter_sizes,
                         num_filters=args.num_filters, lstm_nodes=args.lstm_nodes,
                         lstm_layers=args.lstm_layers, dropout_in=args.dropout_in,
                         dropout_out=args.dropout_out, lstm_dropout=args.lstm_dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2)
        
        print("Training model ...")
        model = train_model(model, train_loader, val_loader, args.num_epochs, optimizer, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), grad_clip=GRAD_CLIP, network_file=args.network_file)
        # Evaluate on test set:
        test_preds = predict_model(model, test_loader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            test_auc = roc_auc_score(test_y.flatten(), test_preds)
        except Exception:
            test_auc = 0.5
        print(f"Test AUROC: {test_auc:.4f}")
        
        # Save the model if network_file is provided.
        if args.network_file:
            torch.save(model.state_dict(), args.network_file)
            print(f"Model saved to {args.network_file}")
            
    elif args.runmode in ["predict", "predict_long"]:
        # In prediction mode, we load a model and run predictions on input sequences.
        if args.sequences is None:
            sys.exit("Please provide --sequences FASTA file for prediction.")
        # Load sequences
        seqs, seq_ids = read_fasta_file(args.sequences, min_length=1, max_length=10000)
        # For prediction, we must ensure that sequence lengths do not exceed the network’s SEQ_SIZE.
        # Here we assume that the network was trained with a fixed SEQ_SIZE.
        # For simplicity, we take the maximum length among the sequences and pad/truncate them.
        max_len = max(len(s) for s in seqs)
        if args.force_max_length > 0:
            max_len = args.force_max_length
        seqs = pad_sequences(seqs, max_len)
        X = onehot_encode(seqs)
        # Build a dataset and loader
        pred_dataset = SequenceDataset(X, np.zeros((X.shape[0], 1), dtype=np.float32))
        pred_loader = DataLoader(pred_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Build the network with the same SEQ_SIZE as training.
        model = DeepCLIP(SEQ_SIZE=max_len, vocab=VOCAB, filter_sizes=args.filter_sizes,
                         num_filters=args.num_filters, lstm_nodes=args.lstm_nodes,
                         lstm_layers=args.lstm_layers, dropout_in=args.dropout_in,
                         dropout_out=args.dropout_out, lstm_dropout=args.lstm_dropout)
        # Load network parameters:
        if args.network_file:
            model.load_state_dict(torch.load(args.network_file, map_location=torch.device("cpu")))
            print(f"Loaded model from {args.network_file}")
        else:
            sys.exit("For prediction, please provide a --network_file to load the model.")
        preds = predict_model(model, pred_loader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # Write predictions to file if specified:
        if args.predict_output_file:
            # Here we write JSON output: list of {id, sequence, score}
            results = []
            for seq, seq_id, score in zip(seqs, seq_ids, preds):
                results.append({'id': seq_id, 'sequence': seq, 'score': float(score)})
            with open(args.predict_output_file, "w") as f:
                json.dump({"predictions": results}, f, indent=4)
            print(f"Predictions written to {args.predict_output_file}")
        else:
            # Otherwise, print to stdout.
            for seq_id, score in zip(seq_ids, preds):
                print(seq_id, score)
    elif args.runmode == "test":
        if args.test_positive is None or args.test_negative is None:
            sys.exit("For test mode, please provide both --test_positive and --test_negative FASTA files.")
        if args.network_file is None:
            sys.exit("For test mode, please provide a --network_file to load the trained model.")
        
        pos_seqs, pos_ids = read_fasta_file(args.test_positive)
        neg_seqs, neg_ids = read_fasta_file(args.test_negative)
        
        all_seqs = pos_seqs + neg_seqs
        all_ids = pos_ids + neg_ids
        pos_labels = [1]*len(pos_seqs)
        neg_labels = [0]*len(neg_seqs)
        all_labels = np.array(pos_labels + neg_labels, dtype=np.float32).reshape(-1, 1)
        
        max_len = max(max(len(s) for s in pos_seqs), max(len(s) for s in neg_seqs))
        if args.force_max_length > 0:
            max_len = args.force_max_length
        all_seqs = pad_sequences(all_seqs, max_len)
        X = onehot_encode(all_seqs)
        
        test_dataset = SequenceDataset(X, all_labels)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        device='cuda:0'
        
        model = DeepCLIP(SEQ_SIZE=max_len, vocab=VOCAB, filter_sizes=args.filter_sizes,
                         num_filters=args.num_filters, lstm_nodes=args.lstm_nodes,
                         lstm_layers=args.lstm_layers, dropout_in=args.dropout_in,
                         dropout_out=args.dropout_out, lstm_dropout=args.lstm_dropout)
        model.load_state_dict(torch.load(args.network_file, map_location=device))
        print(f"Loaded model from {args.network_file}")
        
        test_preds = predict_model(model, test_loader, device)
        try:
            auroc = roc_auc_score(all_labels.flatten(), test_preds)
        except Exception:
            auroc = 0.5
        pred_labels = (test_preds >= 0.5).astype(np.float32)
        acc = accuracy_score(all_labels.flatten(), pred_labels)
        
        print(f"Test AUROC: {auroc:.4f}")
        print(f"Test Accuracy: {acc:.4f}")
        
        fpr, tpr, thresholds = roc_curve(all_labels.flatten(), test_preds)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (area = {auroc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        # plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        if args.plot_file:
            plt.savefig(args.plot_file)
            print(f"ROC plot saved to {args.plot_file}")
        else:
            plt.show()
            
        if args.test_output_file:
            results = {"AUROC": auroc, "Accuracy": acc}
            with open(args.test_output_file, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Test results written to {args.test_output_file}")
    else:
        sys.exit("Invalid runmode specified.")

if __name__ == "__main__":
    main()


    
