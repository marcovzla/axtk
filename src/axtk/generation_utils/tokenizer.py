from typing import Protocol, Optional, Union, List


class HuggingFaceLikeTokenizer(Protocol):
    """
    Tokenizer compatible with HuggingFace tokenizers.

    This class serves as a protocol for tokenizers, ensuring a consistent
    interface that includes methods for encoding and decoding text, as well as
    converting between tokens and their corresponding IDs.
    """

    @property
    def vocab_size(self) -> int:
        pass

    @property
    def bos_token(self) -> Optional[str]:
        pass

    @property
    def bos_token_id(self) -> Optional[int]:
        pass

    @property
    def eos_token(self) -> Optional[str]:
        pass

    @property
    def eos_token_id(self) -> Optional[int]:
        pass

    @property
    def unk_token(self) -> Optional[str]:
        pass

    @property
    def unk_token_id(self) -> Optional[int]:
        pass

    @property
    def sep_token(self) -> Optional[str]:
        pass

    @property
    def sep_token_id(self) -> Optional[int]:
        pass

    @property
    def pad_token(self) -> Optional[str]:
        pass

    @property
    def pad_token_id(self) -> Optional[int]:
        pass

    @property
    def cls_token(self) -> Optional[str]:
        pass

    @property
    def cls_token_id(self) -> Optional[int]:
        pass

    @property
    def mask_token(self) -> Optional[str]:
        pass

    @property
    def mask_token_id(self) -> Optional[int]:
        pass

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        raise NotImplementedError()

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        raise NotImplementedError()

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        raise NotImplementedError()

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        raise NotImplementedError()

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        raise NotImplementedError()
