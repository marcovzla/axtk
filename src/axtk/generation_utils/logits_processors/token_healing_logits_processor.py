from typing import Optional
from itertools import chain
import torch
from pygtrie import CharTrie
from axtk.generation_utils.tokenizer import HuggingFaceLikeTokenizer
from axtk.torch_utils import to_tensor


def build_token_prefix_map(tokenizer: HuggingFaceLikeTokenizer) -> CharTrie:
    token_prefix_map = CharTrie()
    for i in range(tokenizer.vocab_size):
        s = tokenizer.convert_ids_to_tokens(i)
        if s in token_prefix_map:
            token_prefix_map[s].append(i)
        else:
            token_prefix_map[s] = [i]
    return token_prefix_map


# based on https://github.com/marcovzla/rab/blob/b020c5100539d6cce6fe2260bf18f61db5313e74/src/rab/lm/transformers/base.py#L447
class TokenHealingLogitsProcessor:
    def __init__(
            self,
            prompt_ids: torch.Tensor,
            tokenizer: HuggingFaceLikeTokenizer,
            token_prefix_map: Optional[CharTrie] = None,
            bias_value: float = 100.0,
    ):
        self.tokenizer = tokenizer
        self.token_prefix_map = token_prefix_map or build_token_prefix_map(tokenizer)

        # loop backwards through the prompt tokens looking for places where there are
        # possible extensions that cross the prompt boundary
        prefix_str = ''
        self.extension_tokens = []
        for i in range(len(prompt_ids)-1, max(len(prompt_ids)-10, -1), -1):
            token_str = self.tokenizer.convert_ids_to_tokens([prompt_ids[i]])[0]
            prefix_str = token_str + prefix_str
            try:
                extensions = self.prefix_matches(prefix_str)
            except KeyError:
                # this must be a special token outside the vocab, so we assume it does not have any valid extensions
                extensions = []
            self.extension_tokens.append(extensions)
            if i != len(prompt_ids) - 1:
                # add the token used in the input prompt to the list of possible extensions
                self.extension_tokens[-1].append(prompt_ids[i])
        self.extension_tokens = self.extension_tokens[::-1]

        # prune off any extension token positions that don't have multiple possible extensions
        found_extensions = False
        for i in range(len(self.extension_tokens)):
            if len(self.extension_tokens[i]) > 1:
                self.extension_tokens = self.extension_tokens[i:]
                found_extensions = True
                break
        if found_extensions:
            self.healed_token_ids = prompt_ids[len(prompt_ids)-len(self.extension_tokens):]
        else:
            self.extension_tokens = []
            self.healed_token_ids = []

        # if we have multiple possible completions past the last token, then biasing is needed
        if len(self.extension_tokens) > 0:
            # build a set of masks for each possible extension position
            self.token_masks = []
            for i in range(len(self.extension_tokens)):
                token_mask = torch.zeros(self.tokenizer.vocab_size)
                token_mask.scatter_(0, torch.tensor(self.extension_tokens[i]), bias_value)
                self.token_masks.append(token_mask)

        self.num_extensions = 0

    def __call__(self, input_ids, scores):

        # we only bias the first token generated
        if self.num_extensions >= len(self.extension_tokens):
            return scores
        self.num_extensions += 1

        # check if the last token was from the original prompt
        # (if not then we have already "healed" by choosing a token that crosses the prompt boundary)
        if self.num_extensions > 1 and input_ids[0][-1] != self.healed_token_ids[self.num_extensions-2]:
            return scores

        # handle list inputs
        scores = to_tensor(scores)

        # make only allowed tokens possible
        return scores + self.token_masks[self.num_extensions-1]

    def prefix_matches(self, prefix: str) -> list[int]:
        """Returns the list of tokens that match the given prefix."""
        return list(chain.from_iterable(self.token_prefix_map.values(prefix)))
