import os
import math
import random
import dataclasses
from numbers import Number
from collections.abc import MutableMapping, MutableSequence
from typing import Any, Optional, Literal
import torch
import numpy as np
from axtk.utils import is_namedtuple


def random_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return torch.manual_seed(seed)


def enable_full_determinism(seed: int, warn_only: bool = False):
    random_seed(seed)
    # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#concurrent-execution-between-host-and-device
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # https://pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms
    torch.use_deterministic_algorithms(mode=True, warn_only=warn_only)
    # https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-determinism
    torch.backends.cudnn.deterministic = True
    # https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    """
    Function that can be used as DataLoader's worker_init_fn to preserve reproducibility.
    See https://pytorch.org/docs/stable/notes/randomness.html#dataloader.
    """
    worker_seed = torch.initial_seed() % 2**32
    random_seed(worker_seed)


def get_device(module: torch.nn.Module) -> torch.device:
    """Returns the device where the (first parameter of) the module is stored."""
    return next(module.parameters()).device


def move_to_device(obj: Any, device: str | torch.device):
    """Searches for tensors in containers and moves them to the specified device."""
    if torch.is_tensor(obj):
        return obj.to(device)
    elif dataclasses.is_dataclass(obj):
        return obj.__class__(*(move_to_device(x, device) for x in dataclasses.astuple(obj)))
    elif is_namedtuple(obj):
        return obj.__class__(*(move_to_device(x, device) for x in obj))
    elif isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)
    elif isinstance(obj, MutableSequence):
        for i, x in enumerate(obj):
            obj[i] = move_to_device(x, device)
        return obj
    elif isinstance(obj, MutableMapping):
        for k, v in obj.items():
            obj[k] = move_to_device(v, device)
        return obj
    else:
        return obj


def unravel_index(
        index,
        shape: tuple[int, ...],
        order: Literal['C', 'F'] = 'C',
) -> tuple[torch.Tensor, ...]:
    """Converts a flat index or array of flat indices into a tuple of coordinate arrays."""
    if isinstance(index, torch.Tensor):
        index = index.clone()
    elif not isinstance(index, Number):
        index = torch.tensor(index)
    # validate index
    size = math.prod(shape)
    if isinstance(index, Number):
        if index >= size:
            raise ValueError(f'index {index} is out of bounds for array with size {size}')
    else:
        oob = index[index >= size]
        if len(oob) > 0:
            raise ValueError(f'index {oob[0].item()} is out of bounds for array with size {size}')
    # unravel coordinates
    unraveled_coords = []
    if order == 'C':
        shape = reversed(shape)
    for dim in shape:
        unraveled_coords.append(index % dim)
        index //= dim
    if order == 'C':
        unraveled_coords = reversed(unraveled_coords)
    return tuple(unraveled_coords)


def make_first_subword_mask(word_ids: list[Optional[int]]) -> torch.BoolTensor:
    mask = []
    previous_word_id = None
    for word_id in word_ids:
        is_valid_token = word_id is not None and word_id != previous_word_id
        mask.append(is_valid_token)
        previous_word_id = word_id
    return torch.tensor(mask)


def defrag(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        *,
        pad: Optional[Any] = None,
) -> torch.Tensor:
    return defrag_(tensor=tensor.clone(), mask=mask, pad=pad)


def defrag_(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        *,
        pad: Optional[Any] = None,
) -> torch.Tensor:
    # only one- and two-dimensional masks are supported
    if len(mask.shape) not in (1, 2):
        raise ValueError('mask must be one- or two-dimensional')
    # ensure shapes are compatible
    if tensor.shape[:len(mask.shape)] != mask.shape:
        raise ValueError('array shapes are incompatible')
    # ensure two-dimensional mask
    one_dimensional_mask = len(mask.shape) == 1
    if one_dimensional_mask:
        mask = mask.unsqueeze(dim=0)
        tensor.unsqueeze_(dim=0)
    # mask must be boolean
    mask = mask.type(torch.bool)
    # defrag each batch element
    for i in range(mask.shape[0]):
        n = mask[i].sum()
        tensor[i, :n] = tensor[i, mask[i]]
        if pad is not None:
            tensor[i, n:] = pad
    # revert shape if needed
    if one_dimensional_mask:
        tensor.squeeze_(dim=0)
    # return results
    return tensor
