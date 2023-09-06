import os
from typing import Any
from collections.abc import Iterator, Iterable


def is_namedtuple(obj) -> bool:
    """Returns True if obj is a namedtuple."""
    return isinstance(obj, tuple) and hasattr(obj, '_fields')


def is_pathlike(obj) -> bool:
    """Returns True if obj can be a Path."""
    return isinstance(obj, (str, os.PathLike))


def deduplicate_preserve_order(xs: Iterable[Any]) -> list[Any]:
    """
    Removes duplicate elements from an iterable while preserving their original order.

    This function takes an iterable and returns a list containing the unique elements from
    the iterable, maintaining their original order. Duplicate elements are removed,
    and the order of the remaining elements is preserved.

    Args:
        iterable (Iterable[Any]): The input iterable containing elements to be deduplicated.

    Returns:
        list[Any]: A list containing the unique elements from the input iterable in their
        original order.

    Example:
        >>> input_list = [3, 2, 1, 2, 3, 4, 5, 4, 6]
        >>> deduplicated_list = deduplicate_preserve_order(input_list)
        >>> deduplicated_list
        [3, 2, 1, 4, 5, 6]
    """
    return list(dict.fromkeys(xs))


def recursive_flatten(
        xs: Iterable[Any],
        *,
        flatten_strings: bool = False,
        flatten_bytes: bool = False,
) -> Iterator[Any]:
    """
    Recursively flattens an iterable, yielding its elements one by one.

    This function takes an iterable and returns an iterator that yields its elements in
    a flattened manner, recursively processing nested iterables. The `flatten_strings`
    and `flatten_bytes` flags control whether strings and bytes-like objects should be
    treated as leaf elements or recursively flattened.

    Args:
        iterable (Iterable[Any]): The input iterable to be recursively flattened.
        flatten_strings (bool, optional): If True, strings will be recursively flattened.
            If False, strings will be treated as leaf elements. Defaults to False.
        flatten_bytes (bool, optional): If True, bytes-like objects will be recursively
            flattened. If False, bytes-like objects will be treated as leaf elements.
            Defaults to False.

    Returns:
        Iterator[Any]: An iterator that yields elements from the input iterable in a
        flattened manner.

    Example:
        >>> input_list = [1, [2, [3, 4]], [5, 6]]
        >>> flattened_iter = recursive_flatten(input_list)
        >>> list(flattened_iter)
        [1, 2, 3, 4, 5, 6]

    Note:
        - By default, strings and bytes-like objects are treated as leaf elements.
          Use the `flatten_strings` and `flatten_bytes` flags to control their behavior.
    """
    for x in xs:
        if isinstance(x, Iterable):
            if isinstance(x, str) and not flatten_strings:
                yield x
            elif isinstance(x, (bytes, bytearray)) and not flatten_bytes:
                yield x
            else:
                yield from recursive_flatten(x)
        else:
            yield x
