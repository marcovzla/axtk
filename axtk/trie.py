from __future__ import annotations
from typing import Optional
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
import torch


def handle_tensor(x: Iterable[int]) -> Iterable[int]:
    """Helper function to convert tensors into lists."""
    return x.tolist() if isinstance(x, torch.Tensor) else x


@dataclass(repr=False)
class Trie:
    children: dict[int, Trie] = {}
    is_end: bool = False

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

    def prefix_allowed_tokens(self, batch_id: int, input_ids: Iterable[int]) -> list[int]:
        """
        Method that returns a list with the allowed tokens for the next
        generation step conditioned on the previously generated tokens
        input_ids. This method can be used as the prefix_allowed_tokens_fn
        parameter of the model.generate() method in huggingface transformers.
        """
        return self.next_allowed_tokens(input_ids)

    def next_allowed_tokens(self, prefix: Iterable[int]) -> list[int]:
        """Returns a list with the next allowed tokens conditioned on the provided prefix."""
        if node := self.get_node(prefix):
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
