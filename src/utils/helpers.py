import yaml
import os
import torch
import random
import numpy as np

# from generate import generate_
import torch.nn.functional as F

def set_hyps(path, args):
    with open(path, errors="ignore") as f:
        hyps = yaml.safe_load(f)
        for k, v in hyps.items():
            setattr(args, k, v)
    return args

def reproducibility(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


def top_k_top_p_filtering(logits, top_k=1, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def generate_text(sequence, model, output_length, decoder_tokenizer, device, temperature=0.01, top_k=2, top_p=0.0):
    # inputs = encoder_tokenizer.tokenize(sequence).ids
    inputs = torch.tensor(sequence, dtype=torch.long).to(device)
    inputs = inputs.unsqueeze(0)
    print("inputs: ", inputs)

    # print(sequence)

    decoder_input_ids = torch.tensor([[1]],dtype=torch.long).to(model.device)

    with torch.no_grad():
        for _ in range(output_length):
            outputs = model(input_ids=inputs, decoder_input_ids=decoder_input_ids)  # Get logits
            next_token_logits = outputs[0][:, -1, :] / temperature  # Apply temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)  # Apply top-k and/or top-p
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)  # Sample
            decoder_input_ids = torch.cat((decoder_input_ids, next_token), dim=1)  # Add the token to the generated text
            print("Outputs", decoder_input_ids)
    # print(decoder_input_ids)
    generated_text = decoder_tokenizer.decode(decoder_input_ids[0].cpu().numpy().tolist())
    return generated_text


def postprocess_rna(rna):
    return rna.replace('b', 'A').replace('j', 'C').replace(
                    'u', 'U').replace('z', 'G').replace(' ', '').replace(
                    'B', 'A').replace('J', 'C').replace('U', 'U').replace('Z', 'G')