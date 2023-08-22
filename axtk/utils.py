from typing import Any
from collections.abc import Iterator, Iterable
import os


def is_namedtuple(obj) -> bool:
    """Returns True if obj is a namedtuple."""
    return isinstance(obj, tuple) and hasattr(obj, '_fields')


def is_pathlike(obj) -> bool:
    """Returns True if obj can be a Path."""
    return isinstance(obj, (str, os.PathLike))


def drop_duplicates(xs: Iterable[Any]) -> list[Any]:
    """Deduplicates elements in iterable, preserving their order."""
    return list(dict.fromkeys(xs))


def flatten(
        xs: Iterable[Any],
        *,
        flatten_strings: bool = False,
        flatten_bytes: bool = False,
) -> Iterator[Any]:
    """Flattens iterable recursively."""
    for x in xs:
        if isinstance(x, Iterable):
            if isinstance(x, str) and not flatten_strings:
                yield x
            elif isinstance(x, (bytes, bytearray)) and not flatten_bytes:
                yield x
            else:
                yield from flatten(x)
        else:
            yield x
