import os
import math
import random
import dataclasses
from numbers import Number
from collections.abc import MutableMapping, MutableSequence
from typing import Any, Optional, Literal, Union
import torch
import numpy as np
from axtk.utils import is_namedtuple
from axtk.average import ExponentialMovingAverage


def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return torch.manual_seed(seed)


def enable_full_determinism(seed: int, warn_only: bool = False):
    set_seed(seed)
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
    set_seed(worker_seed)


def get_device(module: torch.nn.Module) -> torch.device:
    """Returns the device where the (first parameter of) the module is stored."""
    return next(module.parameters()).device


def move_to_device(obj: Any, device: str | torch.device):
    """Searches for tensors in containers and moves them to the specified device."""
    if isinstance(obj, torch.Tensor):
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
    """
    Creates a boolean mask tensor indicating the positions of the first subword token
    corresponding to each word of a sequence, based on the provided list of word_ids.
    """
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
        empty_value: Optional[Any] = None,
) -> torch.Tensor:
    """
    Rearranges the elements in the input tensor based on the provided mask,
    while optionally filling empty positions with a specified value.
    """
    return defrag_(tensor=tensor.clone(), mask=mask, empty_value=empty_value)


def defrag_(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        *,
        empty_value: Optional[Any] = None,
) -> torch.Tensor:
    """
    Rearranges the elements in the input tensor based on the provided mask,
    while optionally filling empty positions with a specified value.
    """
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
        if empty_value is not None:
            tensor[i, n:] = empty_value
    # revert shape if needed
    if one_dimensional_mask:
        tensor.squeeze_(dim=0)
    # return results
    return tensor


def shift_value_range(
        tensor: torch.Tensor,
        source_range: Optional[tuple[Number, Number]] = None,
        target_range: tuple[Number, Number] = (0, 1),
        clip_values: bool = True,
) -> torch.Tensor:
    """
    Shifts the values of a tensor from source_range to target_range.
    If target_range is not provided, it defaults to (0, 1).
    If source_range is not provided, it defaults to the tensor's min and max values.
    If source_range is provided and clip_values is set to True (default), the tensor's values are clipped.
    """
    tensor = tensor.float()
    if source_range is None:
        # if source_range was not provided, infer it from the input tensor values
        source_range = (tensor.min().item(), tensor.max().item())
    elif clip_values:
        # if source_range was provided and clip_values is true, clip input tensor values
        tensor = tensor.clip(*source_range)
    # shift from source_range to 0-1 range
    if source_range != (0, 1):
        from_min, from_max = source_range
        tensor -= from_min
        tensor /= from_max - from_min
    # shift from 0-1 range to target_range
    if target_range != (0, 1):
        to_min, to_max = target_range
        tensor *= to_max - to_min
        tensor += to_min
    # return shifted tensor
    return tensor


def slerp(
        start: torch.Tensor,
        end: torch.Tensor,
        t: Union[float, torch.Tensor],
        dim: Optional[bool] = None,
):
    """Spherical linear interpolation."""
    # https://en.wikipedia.org/wiki/Slerp#Geometric_Slerp
    keepdim = dim is not None
    start_norm = start / torch.linalg.vector_norm(start, dim=dim, keepdim=keepdim)
    end_norm = end / torch.linalg.vector_norm(end, dim=dim, keepdim=keepdim)
    omega = torch.acos(torch.sum(start_norm * end_norm, dim=dim))
    sin_omega = torch.sin(omega)
    t0 = torch.where(sin_omega == 0, 1 - t, torch.sin((1 - t) * omega) / sin_omega)
    t1 = torch.where(sin_omega == 0, t, torch.sin(t * omega) / sin_omega)
    if dim is not None:
        t0.unsqueeze_(dim=dim)
        t1.unsqueeze_(dim=dim)
    return t0 * start + t1 * end


def smooth_values(
        input: torch.Tensor,
        dim: int = -1,
        dtype: Optional[torch.dtype] = None,
        beta: float = 0.98,
) -> torch.Tensor:
    # if dtype was not provided, use same as input tensor
    if dtype is None:
        dtype = input.dtype
    # move smooth dimension to the end
    smooth_last_dimension = dim == -1 or dim == len(input.size()) - 1
    if not smooth_last_dimension:
        input = input.transpose(dim, -1)
    # get input tensor shape
    size = input.size()
    # reshape input tensor into a batch of tensors to smooth
    input = input.reshape(-1, size[-1])
    # make output tensor
    output = torch.empty_like(input, dtype=dtype)
    # smooth each tensor in reshaped input
    for i in range(input.size(0)):
        avg = ExponentialMovingAverage(beta)
        for j in range(input.size(1)):
            avg.add(input[i, j])
            output[i, j] = avg.value
    # reshape output tensor to original shape
    output = output.view(size=size)
    if not smooth_last_dimension:
        output = output.transpose(dim, -1)
    # return output
    return output.contiguous()
