from __future__ import annotations
from typing import Optional, Generic, TypeVar
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
import torch


T = TypeVar('T')


@dataclass(repr=False)
class Trie(Generic[T]):
    children: dict[T, Trie] = field(default_factory=dict)
    is_end: bool = False

    def __repr__(self) -> str:
        return f'<Trie(is_end={self.is_end}, children_count={len(self.children)})>'

    def traverse(self) -> Iterator[Trie]:
        """Iterates over all nodes in the trie in topological order."""
        yield self
        for child in self.children.values():
            yield from child.traverse()

    def num_sequences(self) -> int:
        """Returns the number of sequences stored in the trie."""
        return sum(node.is_end for node in self.traverse())

    def insert(self, sequence: Iterable[T]):
        """Insert a sequence into the trie."""
        node = self._traverse_or_insert(sequence, insert=True)
        node.is_end = True

    def search(self, sequence: Iterable[T]) -> bool:
        """Check if the sequence is stored in the trie."""
        node = self._traverse_or_insert(sequence)
        return node.is_end if node else False

    def next_allowed_tokens(self, sequence: Iterable[T] = ()) -> list[T]:
        """Return the next allowed tokens based on the provided sequence."""
        node = self._traverse_or_insert(sequence)
        return list(node.children.keys()) if node else []

    def prefix_allowed_tokens(self, batch_id: int, input_ids: torch.Tensor) -> list[int]:
        """
        Return allowed tokens for the next generation step based on previously generated tokens.
        
        This method is compatible with the `prefix_allowed_tokens_fn` parameter of the `generate()`
        method in the HuggingFace transformers library.
        
        Args:
            batch_id (int): The ID of the batch being processed.
            input_ids (torch.Tensor): The token IDs of previously generated tokens.
            
        Returns:
            list[int]: List of token IDs that are allowed for the next generation step.
        """
        return self.next_allowed_tokens(input_ids.tolist())

    def _traverse_or_insert(self, sequence: Iterable[T], insert: bool = False) -> Optional[Trie]:
        """Traverse the trie based on the sequence. If 'insert' is True, insert missing nodes."""
        if not isinstance(sequence, Iterable):
            raise ValueError('Input sequence should be iterable.')
        node = self
        for token in sequence:
            if token not in node.children:
                if insert:
                    node.children[token] = Trie()
                else:
                    return
            node = node.children[token]
        return node
