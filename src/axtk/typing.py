import os
from typing import Any, Union, Callable, TypeVar
from numbers import Number

Args = tuple[Any, ...]
KwArgs = dict[str, Any]

PathLike = Union[str, os.PathLike[str]]

F = TypeVar('F', bound=Callable)
"""Type variable bound to Callable."""

N = TypeVar('N', bound=Number)
"""Type variable bound to Number."""
