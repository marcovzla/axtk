from typing import Optional, Union
from collections.abc import Iterable
from dataclasses import dataclass
from axtk.utils import deduplicate_preserve_order


@dataclass(repr=False)
class Vocabulary:
    id2label: dict[int, str]
    label2id: dict[str, int]

    @classmethod
    def from_labels(cls, labels: Iterable[str]):
        labels = deduplicate_preserve_order(labels)
        id2label = {i: label for i, label in enumerate(labels)}
        label2id = {label: i for i, label in enumerate(labels)}
        return cls(id2label, label2id)

    def __len__(self):
        return len(self.id2label)

    def __iter__(self):
        return iter(self.id2label.items())

    def __contains__(self, value: Union[str, int]):
        if isinstance(value, str):
            return value in self.label2id
        elif isinstance(value, int):
            return value in self.id2label
        else:
            raise TypeError('value must be int or str')

    def __getitem__(self, key: Union[str, int]):
        if isinstance(key, str):
            if key not in self.label2id:
                raise KeyError('label not in vocabulary')
            return self.label2id[key]
        elif isinstance(key, int):
            if key not in self.id2label:
                raise KeyError('id not in vocabulary')
            return self.id2label[key]
        else:
            raise TypeError('key must be int or str')

    def labels(self) -> list[str]:
        return list(self.label2id.keys())

    def add(self, label: str) -> Optional[int]:
        if label not in self.label2id:
            i = len(self.label2id)
            self.label2id[label] = i
            self.id2label[i] = label
            return i
