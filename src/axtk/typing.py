import os
from typing import Any, Union, Callable, TypeVar

Args = tuple[Any, ...]
KwArgs = dict[str, Any]

PathLike = Union[str, os.PathLike[str]]

F = TypeVar('F', bound=Callable)
"""Type variable bound to Callable."""
