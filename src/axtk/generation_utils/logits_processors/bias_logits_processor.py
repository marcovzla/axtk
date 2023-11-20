import torch
from axtk.torch_utils import to_tensor

class BiasLogitsProcessor:
    def __init__(self, vocab_size: int, logit_bias: dict[int, float]):
        self.bias_vector = torch.zeros(vocab_size)
        for token_id, bias in logit_bias.items():
            self.bias_vector[token_id] = bias

    def __call__(self, input_ids, scores):
        # sometimes the model has a vocab_size that is bigger than the tokenizer
        # so that the embedding matrix can have a length that is a power of 2,
        # but these extra tokens are never used
        scores = scores[:, :self.tokenizer.vocab_size]
        scores = to_tensor(scores)
        return scores + self.bias_vector
