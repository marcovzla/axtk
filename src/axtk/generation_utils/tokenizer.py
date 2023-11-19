from typing import Protocol, Optional, Union
import numpy as np
import torch


class HuggingFaceLikeTokenizer(Protocol):
    """
    Tokenizer compatible with HuggingFace tokenizers.

    This class serves as a protocol for tokenizers, ensuring a consistent
    interface that includes methods for encoding and decoding text, as well as
    converting between tokens and their corresponding IDs.
    """

    vocab_size: int

    bos_token: Optional[str] = None
    eos_token: Optional[str] = None
    unk_token: Optional[str] = None
    sep_token: Optional[str] = None
    pad_token: Optional[str] = None
    cls_token: Optional[str] = None
    mask_token: Optional[str] = None

    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    unk_token_id: Optional[int] = None
    sep_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    cls_token_id: Optional[int] = None
    mask_token_id: Optional[int] = None

    def encode(
            self,
            text: str,
            add_special_tokens: bool = True,
    ) -> list[int]:
        raise NotImplementedError()

    def decode(
            self,
            ids: Union[int, list[int], np.ndarray, torch.Tensor],
            skip_special_tokens: bool = False,
    ) -> str:
        raise NotImplementedError()

    def convert_ids_to_tokens(
            self,
            ids: Union[int, list[int], np.ndarray, torch.Tensor],
    ) -> Union[str, list[str]]:
        raise NotImplementedError()

    def convert_tokens_to_ids(
            self,
            tokens: Union[str, list[str]],
    ) -> Union[int, list[int]]:
        raise NotImplementedError()

    def convert_tokens_to_string(
            self,
            tokens: list[str],
    ) -> str:
        raise NotImplementedError()
