from __future__ import annotations
from typing import Optional
from collections.abc import Iterable, Iterator


try:
    import torch
    def handle_tensor(x):
        return x.tolist() if isinstance(x, torch.Tensor) else x
except:
    def handle_tensor(x):
        return x


class Trie:
    def __init__(self):
        self.is_end: bool = False
        self.children: dict[int, Trie] = {}

    def num_sequences(self) -> int:
        """Returns the number of sequences stored in the trie."""
        return sum(node.is_end for node in self.traverse())

    def insert(self, sequence: Iterable[int]):
        """Inserts a sequence in the trie."""
        node = self.get_node(sequence, insert=True)
        node.is_end = True

    def search(self, sequence: Iterable[int]) -> bool:
        """Returns True if sequence is a complete sequence stored in the trie."""
        if node := self.get_node(sequence):
            return node.is_end
        return False

    def next_allowed_tokens(self, batch_id: int, input_ids: Iterable[int]) -> list[int]:
        """
        Method that returns a list with the allowed tokens for the next
        generation step conditioned on the previously generated tokens
        input_ids. This method can be used as the prefix_allowed_tokens_fn
        parameter of the model.generate() method in huggingface transformers.
        """
        if node := self.get_node(input_ids):
            return list(node.children.keys())
        return []

    def get_node(self, prefix: Iterable[int], insert: bool = False) -> Optional[Trie]:
        """
        Follows the prefix and returns the next node after the prefix.
        If the prefix is not in the trie, returns None.
        However, if insert is set to True, prefix nodes not in the trie
        are inserted and the next node after the prefix is returned.
        """
        node = self
        for i in handle_tensor(prefix):
            if i not in node.children:
                if insert:
                    node.children[i] = Trie()
                else:
                    return
            node = node.children[i]
        return node

    def traverse(self) -> Iterator[Trie]:
        """Iterates over all nodes in the trie in topological order."""
        yield self
        for child in self.children.values():
            yield from child.traverse()
