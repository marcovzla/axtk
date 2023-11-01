import re
from typing import Union
from axtk.generation_utils.tokenizer import HuggingFaceLikeTokenizer

try:
    import llama_cpp  # type: ignore
except ImportError:
    raise ImportError(
        'Please install llama-cpp-python. '
        'Installation instructions can be found here: '
        'https://llama-cpp-python.readthedocs.io/en/latest/#installation-from-pypi'
    )



ENCODING = 'utf-8'

SENTENCEPIECE_WHITESPACE = b'\xe2\x96\x81'.decode(ENCODING)



def llama_unescape_whitespace(s: str) -> str:
    return s.replace(SENTENCEPIECE_WHITESPACE, ' ')

def llama_parse_byte_token(token: str) -> str:
    return chr(int(token[3:-1], base=16))



class LlamaCppTokenizer(HuggingFaceLikeTokenizer):
    """Simulates a HuggingFace tokenizer for llama_cpp models."""

    def __init__(self, model: llama_cpp.Llama):
        self.llama = model
        self.vocab_type = llama_cpp.llama_vocab_type(self.llama.ctx)
        self.vocab_size = llama_cpp.llama_n_vocab(self.llama.ctx)
        self.bos_token_id = llama_cpp.llama_token_bos(self.llama.ctx)
        self.eos_token_id = llama_cpp.llama_token_eos(self.llama.ctx)
        self.bos_token = self._id_to_token(self.bos_token_id)
        self.eos_token = self._id_to_token(self.eos_token_id)

        self.token_ids = {
            self._id_to_token(i): i
            for i in range(self.vocab_size)
        }

        special_tokens_patterns = [
            re.escape(self._id_to_token(i))
            for i in range(self.vocab_size)
            if self._id_to_token_type(i) in (llama_cpp.LLAMA_TOKEN_TYPE_CONTROL, llama_cpp.LLAMA_TOKEN_TYPE_UNKNOWN)
        ]

        # note the capturing parenthesis, they are needed so that split() returns the separators,
        # which correspond to special tokens
        self.special_token_pattern = re.compile('(' + '|'.join(special_tokens_patterns) + ')')

    def encode(self, text: str, add_special_tokens: bool = True, **kwargs) -> list[int]:
        token_ids = [self.bos_token_id] if add_special_tokens else []
        # iterate over chunks of text and special tokens
        # even position: chunk of text
        # odd position: special token
        for i, chunk_or_special_token in enumerate(self.special_token_pattern.split(text)):
            if i % 2 == 0:
                # tokenize the chunk of text
                chunk_token_ids = self.llama.tokenize(chunk_or_special_token.encode(ENCODING), add_bos=False)
                if i != 0 and chunk_token_ids and self.vocab_type == llama_cpp.LLAMA_VOCAB_TYPE_SPM:
                    # if sentencepiece prepended a space to the first token of the chunk
                    # and this is not the first chunk then we need to remove the space
                    first_token = self.convert_ids_to_tokens(chunk_token_ids[0])
                    if first_token.startswith(SENTENCEPIECE_WHITESPACE) and len(first_token) > 1:
                        chunk_token_ids[0] = self.convert_tokens_to_ids(first_token[1:])
                token_ids += chunk_token_ids
            else:
                # get the id of the special token
                token_ids.append(self.convert_tokens_to_ids(chunk_or_special_token))
        return token_ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False, **kwargs) -> str:
        return self._tokens_to_string(ids, skip_special_tokens)

    def convert_ids_to_tokens(self, ids: Union[int, list[int]]) -> Union[str, list[str]]:
        if isinstance(ids, int):
            return self._id_to_token(ids)
        else:
            return [self._id_to_token(id) for id in ids]

    def convert_tokens_to_ids(self, tokens: Union[str, list[str]]) -> Union[int, list[int]]:
        if isinstance(tokens, str):
            return self.token_ids[tokens]
        else:
            return [self.token_ids[t] for t in tokens]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self._tokens_to_string(tokens, skip_special_tokens=False)

    def _id_to_token(self, id: int) -> str:
        return llama_cpp.llama_token_get_text(self.llama.ctx, id).decode(ENCODING)

    def _id_to_token_type(self, id: int) -> int:
        return llama_cpp.llama_token_get_type(self.llama.ctx, id)

    def _tokens_to_string(self, tokens: list[Union[str, int]], skip_special_tokens: bool = False) -> str:
        return ''.join(self._token_to_piece(t, skip_special_tokens) for t in tokens)

    def _token_to_piece(self, token: Union[str, int], skip_special_tokens: bool) -> str:
        # get token and token id
        if isinstance(token, str):
            token_id = self.token_ids[token]
        else:
            token_id = token
            token = self._id_to_token(token_id)
        # get corresponding token type
        token_type = self._id_to_token_type(token_id)
        # convert token to piece based on its type
        if token_type == llama_cpp.LLAMA_TOKEN_TYPE_NORMAL:
            if self.vocab_type == llama_cpp.LLAMA_VOCAB_TYPE_SPM:
                return llama_unescape_whitespace(token)
            else:
                return token
        elif token_type == llama_cpp.LLAMA_TOKEN_TYPE_UNKNOWN:
            return '' if skip_special_tokens else token
        elif token_type == llama_cpp.LLAMA_TOKEN_TYPE_CONTROL:
            return '' if skip_special_tokens else token
        elif token_type == llama_cpp.LLAMA_TOKEN_TYPE_BYTE:
            return llama_parse_byte_token(token)
        else:
            return ''
