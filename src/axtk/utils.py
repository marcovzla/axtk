import os
import hashlib
from typing import Any, Optional, TypeVar
from collections.abc import Iterator, Iterable, Hashable
from collections import defaultdict
from operator import itemgetter
from pathlib import Path
from axtk.typing import PathLike


T = TypeVar('T', bound=Hashable)


def is_namedtuple(obj) -> bool:
    """Returns True if obj is a namedtuple."""
    return isinstance(obj, tuple) and hasattr(obj, '_fields')


def is_pathlike(obj) -> bool:
    """Returns True if obj can be a Path."""
    return isinstance(obj, (str, os.PathLike))


def deduplicate_preserve_order(xs: Iterable[T]) -> list[T]:
    """
    Removes duplicate elements from an iterable while preserving their original order.

    This function takes an iterable and returns a list containing the unique elements from
    the iterable, maintaining their original order. Duplicate elements are removed,
    and the order of the remaining elements is preserved.

    Args:
        iterable (Iterable[T]): The input iterable containing elements to be deduplicated.

    Returns:
        list[T]: A list containing the unique elements from the input iterable in their
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


def file_md5_hash(filepath: PathLike, chunk_size: int = 8192) -> str:
    """
    Calculate the MD5 hash of a given file.

    Args:
        filepath (Union[str, Path]): Path to the file for which the MD5 hash needs to be calculated.
        chunk_size (int, optional): Size of chunks to read from the file for hashing. 
            Defaults to 8192 bytes (8KB).

    Returns:
        str: The MD5 hash of the file content.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File '{path}' not found.")

    hasher = hashlib.md5()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def reciprocal_rank_fusion(
        ranked_lists: list[list[T]],
        weights: Optional[list[float]] = None,
        k: float = 60.0,
) -> tuple[list[T], list[float]]:
    """
    Compute Weighted Reciprocal Rank Fusion (WRRF) for multiple ranked lists.

    Args:
        ranked_lists (list[list[T]]): List of ranked lists containing items of type T.
        weights (Optional[list[float]]): Weights for each ranked list. Defaults to None,
            in which case uniform weights are applied.
        k (float, optional): Constant added to the rank. Default is 60.0.

    Returns:
        tuple[list[T], list[float]]: A tuple containing two lists:
            A list of unique items from the input ranked lists sorted in descending order
            by their WRRF scores, and a list of the corresponding WRRF scores.

    Reference:
        Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).
        Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.
        SIGIR '09. https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
    """
    rrf_scores = defaultdict(float)

    if weights is None:
        weights = [1.0] * len(ranked_lists)
    elif len(ranked_lists) != len(weights):
        raise ValueError('Length of weights must match the number of ranked lists')

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(ranked_list, start=1):
            rrf_scores[doc_id] += weight / (k + rank)

    sorted_items = sorted(rrf_scores.items(), key=itemgetter(1), reverse=True)
    items, scores = zip(*sorted_items)

    return list(items), list(scores)
