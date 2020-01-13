#  Copyright (c) polakowo
#  Licensed under the MIT license.

import random
import numpy as np
import torch
import torch.nn.functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
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
        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, context, config):
    # Parse parameters
    no_cuda = config.getboolean('model', 'no_cuda')
    num_samples = config.getint('decoder', 'num_samples')
    length = config.getint('decoder', 'length')
    temperature = config.getfloat('decoder', 'temperature')
    top_k = config.getint('decoder', 'top_k')
    top_p = config.getfloat('decoder', 'top_p')

    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in range(length):
            inputs = {'input_ids': generated}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0.0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    return generated

def generate_response(model, tokenizer, context, config):
    # Parse parameters
    num_samples = config.getint('decoder', 'num_samples')
    seed = config.get('decoder', 'seed')
    seed = int(seed) if seed is not None else None

    # Make answers reproducible only if wanted
    if seed is not None:
        set_seed(seed)

    # Generate response
    context_tokens = tokenizer.encode(context)
    out = sample_sequence(model, context_tokens, config)
    out = out[:, len(context_tokens):].tolist()
    texts = []
    for o in out:
        text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
        text = text[: text.find(tokenizer.eos_token)]
        texts.append(text)
    if num_samples == 1:
        return texts[0]
    return texts