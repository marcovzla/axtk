import regex
import torch
from axtk.generation_utils.logits_processors.acceptable_logits_processor import AcceptableLogitsProcessor


class RegexLogitsProcessor(AcceptableLogitsProcessor):
    def __init__(
            self,
            pattern: str,
            stop_regex: str | list[str],
            **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(stop_regex, str):
            stop_regex = [stop_regex]

        self.pattern = regex.compile(pattern + '(?:' + '|'.join(stop_regex) + ')?')

        self.current_token_ids = []
        self.current_length = 0
        self.forced_chars = 0

    def update_state(self, input_ids: torch.Tensor):
        assert input_ids.size(0) == 1, 'batched inference not supported'
        new_token_ids = input_ids[0][self.current_length:]
        self.current_token_ids.extend(new_token_ids)
        self.current_length = input_ids.size(1)

    def is_acceptable(self, proposed_token_id: int) -> bool:
        proposed_string = self.get_proposed_string(proposed_token_id)
        m = self.pattern.fullmatch(proposed_string, partial=True)
        return m is not None

    def get_proposed_string(self, proposed_token_id: int) -> str:
        string = self.tokenizer.decode(self.current_token_ids + [proposed_token_id])
        return string[self.prefix_length:]
