import torch
from axtk.torch_utils import to_tensor

class BiasLogitsProcessor:
    def __init__(self, vocab_size: int, logit_bias: dict[int, float]):
        self.bias_vector = torch.zeros(vocab_size)
        for token_id, bias in logit_bias.items():
            self.bias_vector[token_id] = bias

    def __call__(self, input_ids, scores):
        scores = to_tensor(scores)
        return scores + self.bias_vector
