from collections.abc import Sequence
import torch
from axtk.torch_utils import to_tensor
from axtk.generation_utils.tokenizer import HuggingFaceLikeTokenizer



class AcceptableLogitsProcessor:

    def __init__(
            self,
            tokenizer: HuggingFaceLikeTokenizer,
            prefix_length: int = 0,
            is_greedy: bool = False,
            max_probability_mass: float = 0.95,
            n_sigmas: float = 6.0,
    ):
        self.tokenizer = tokenizer
        self.prefix_length = prefix_length
        self.is_greedy = is_greedy
        self.max_probability_mass = max_probability_mass
        self.n_sigmas = n_sigmas
        self.bias_vector = torch.zeros(self.tokenizer.vocab_size)

    def __call__(self, input_ids, scores):
        # sometimes the model has a vocab_size that is bigger than the tokenizer
        # so that the embedding matrix can have a length that is a power of 2,
        # but these extra tokens are never used
        scores = scores[:, :self.tokenizer.vocab_size]
        # handle 1D inputs
        input_ids, scores, one_dim = self.handle_dimensions(input_ids, scores)
        # update processor state
        self.update_state(input_ids)
        # compute the bias values
        return self.compute_bias_values(input_ids, scores, one_dim)

    def handle_dimensions(self, input_ids, scores) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Ensure correct tensor dimensions."""
        one_dim = False
        if not isinstance(input_ids[0], Sequence) and not (hasattr(input_ids[0], 'shape') and len(input_ids[0].shape) > 0):
            one_dim = True
            input_ids = to_tensor(input_ids).unsqueeze(0)
            scores = to_tensor(scores).unsqueeze(0)
        return input_ids, scores, one_dim

    def compute_bias_values(self, input_ids: torch.Tensor, scores: torch.Tensor, one_dim: bool) -> torch.Tensor:
        """Compute and return the bias values based on input and scores."""
        self.bias_vector.fill_(0)
        sort_inds = torch.argsort(scores, dim=1, descending=True)
        to_bias = self.get_acceptable_token_ids(sort_inds, scores)

        # if we found no more valid tokens then we just end the sequence
        if not to_bias:
            to_bias = [self.tokenizer.eos_token_id]

        # make sure the tokens that fit the pattern have higher scores than the top value
        max_score = scores[0, sort_inds[0, 0]]
        min_to_bias = scores[0, to_bias].min()
        score_offset = self.n_sigmas * scores[0].std()
        bias_value = max_score - min_to_bias + score_offset
        self.bias_vector[to_bias] = bias_value
        out = scores + self.bias_vector.to(scores.device)

        return out[0] if one_dim else out

    def get_acceptable_token_ids(self, sort_inds: torch.Tensor, scores: torch.Tensor) -> list[int]:
        """Return the indices to bias based on the sorted indices and scores."""
        # list to collect token ids that are acceptable
        acceptable = []
        # calculate probability distribution
        probabilities = torch.nn.functional.softmax(scores, dim=1)
        seen_probability_mass = 0.0

        for i in range(sort_inds.shape[1]):
            token_id = sort_inds[0, i].item()
            if self.is_acceptable(token_id):
                acceptable.append(token_id)
                # if working in greedy mode, stop after the first acceptable token is found
                if self.is_greedy:
                    break
            # stop if we have covered enough of the probability mass
            # and at least one acceptable token has been found
            seen_probability_mass += probabilities[0, token_id]
            if acceptable and seen_probability_mass > self.max_probability_mass:
                break
        return acceptable

    def update_state(self, input_ids: torch.Tensor):
        """Update state with new input."""
        raise NotImplementedError()

    def is_acceptable(self, proposed_token_id: int) -> bool:
        """Returns True if the proposed token is acceptable."""
        raise NotImplementedError()
