from typing import Optional, Union, Any
from collections.abc import Iterable
from dataclasses import dataclass
from axtk.utils import deduplicate_preserve_order


@dataclass(repr=False)
class Vocabulary:
    """
    Represents a bidirectional mapping between labels (str) and their IDs (int).

    Attributes:
        id2label (dict[int, str]): A mapping from IDs to labels.
        label2id (dict[str, int]): A mapping from labels to IDs.
    """

    id2label: dict[int, str]
    label2id: dict[str, int]

    def __post_init__(self):
        """Post-initialization to ensure consistency between id2label and label2id."""
        if self.id2label.keys() != self.label2id.values() or self.label2id.keys() != self.id2label.values():
            raise ValueError('id2label and label2id must be consistent')

    def __repr__(self):
        """Represent the Vocabulary instance as a string."""
        return f'<{self.__class__.__name__}: {len(self):,} labels>'

    def __len__(self):
        """Return the number of labels in the vocabulary."""
        return len(self.id2label)

    def __iter__(self):
        """Iterate over the id-label pairs in the vocabulary."""
        return iter(self.id2label.items())

    def __contains__(self, value: Union[str, int]):
        """
        Check if a label or ID is in the vocabulary.

        Args:
            value (Union[str, int]): The label or ID to check.

        Raises:
            TypeError: If the value is not a string or integer.

        Returns:
            bool: True if the value is in the vocabulary, False otherwise.
        """
        if isinstance(value, str):
            return value in self.label2id
        elif isinstance(value, int):
            return value in self.id2label
        else:
            raise TypeError('value must be int or str')

    def __getitem__(self, key: Union[str, int]):
        """
        Retrieve the corresponding label or ID from the vocabulary.

        Args:
            key (Union[str, int]): The label or ID to retrieve.

        Raises:
            KeyError: If the key is not in the vocabulary.
            TypeError: If the key is not a string or integer.

        Returns:
            Union[str, int]: The corresponding label or ID.
        """
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

    @classmethod
    def from_labels(cls, labels: Iterable[str]):
        """
        Create a Vocabulary instance from an iterable of labels.

        Args:
            labels (Iterable[str]): An iterable of label strings.

        Returns:
            Vocabulary: A new Vocabulary instance.
        """
        labels = deduplicate_preserve_order(labels)
        id2label = {i: label for i, label in enumerate(labels)}
        label2id = {label: i for i, label in enumerate(labels)}
        return cls(id2label, label2id)

    @property
    def labels(self) -> list[str]:
        """
        Return a list of labels in the vocabulary.

        Returns:
            list[str]: The list of labels.
        """
        return list(self.label2id.keys())

    def get(self, key: Union[str, int], default: Optional[Any]= None) -> Optional[Any]:
        """
        Safely retrieve a value from the vocabulary with a default.

        Args:
            key (Union[str, int]): The label or ID to retrieve.
            default (Optional[Any]): The default value to return if the key is not found.

        Returns:
            Optional[Any]: The corresponding value or the default.
        """
        try:
            return self[key]
        except KeyError:
            return default

    def add(self, label: str) -> Optional[int]:
        """
        Add a new label to the vocabulary.

        Args:
            label (str): The label to add.

        Returns:
            Optional[int]: The ID of the newly added label, or None if the label already exists.

        Raises:
            TypeError: If the label is not a string.
        """
        if not isinstance(label, str):
            raise TypeError('label must be a string')

        if label not in self.label2id:
            i = len(self.label2id)
            self.label2id[label] = i
            self.id2label[i] = label
            return i

    def get_or_add(self, label: str) -> int:
        """
        Retrieve the ID of a label or add it to the vocabulary if it doesn't exist.

        This method checks if the given label is already in the vocabulary. If it is,
        the method returns its corresponding ID. If the label is not in the vocabulary,
        it is added, and its new ID is returned.

        Args:
            label (str): The label to retrieve or add.

        Returns:
            int: The ID of the label.

        Raises:
            TypeError: If the label is not a string.
        """
        if not isinstance(label, str):
            raise TypeError('label must be a string')

        return self[label] if label in self else self.add(label)
