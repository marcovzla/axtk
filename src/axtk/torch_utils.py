import math
import dataclasses
from numbers import Number
from types import FunctionType, MethodType
from collections.abc import MutableMapping, MutableSequence
from typing import Any, Optional, Literal, Union, cast
import torch
import numpy as np
from axtk import set_seed, enable_full_determinism
from axtk.utils import is_namedtuple
from axtk.typing import F, N
from axtk.average import ExponentialMovingAverage


def seed_worker(worker_id: int):
    """
    Function that can be used as DataLoader's worker_init_fn to preserve reproducibility.
    See https://pytorch.org/docs/stable/notes/randomness.html#dataloader.
    """
    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)


def to_tensor(
        x: Any,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Convert the input to a PyTorch tensor if it's not already one.

    Args:
        x (Any): Input data. Can be any type that is convertible to a PyTorch tensor.
        dtype (Optional[torch.dtype], optional): The desired data type of the returned tensor.
            Defaults to None which means infer from the input data.
        device (Optional[torch.device], optional): The desired device of the returned tensor.
            Defaults to None which means the current default device.

    Returns:
        torch.Tensor: The input data as a PyTorch tensor.
    """
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x, dtype=dtype, device=device)


def to_list(x: Union[N, list[N], np.ndarray, torch.Tensor]) -> list[N]:
    """
    Converts a given input to a list of numbers. The input can be a single
    number, a list of numbers, a NumPy ndarray, or a PyTorch tensor. If the
    input is already a list, it is returned as is.

    Args:
        x (Union[N, list[N], np.ndarray, torch.Tensor]): The input to convert.
            Can be a single number (N), a list of numbers (list[N]), a NumPy
            ndarray, or a PyTorch tensor.

    Returns:
        list[N]: A list of numbers converted from the input.

    Raises:
        TypeError: If the input type is not supported.

    Examples:
        >>> to_list(5)
        [5]
        >>> to_list([1, 2, 3])
        [1, 2, 3]
        >>> to_list(np.array([1, 2, 3]))
        [1, 2, 3]
        >>> to_list(torch.tensor([1, 2, 3]))
        [1, 2, 3]
    """
    if isinstance(x, Number):
        return [x]
    elif isinstance(x, list):
        return x
    elif isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, torch.Tensor):
        return x.tolist()
    else:
        raise TypeError(f'Invalid type: {x.__class__.__name__}')


def get_device(module: torch.nn.Module) -> torch.device:
    """
    Retrieves the device on which a PyTorch module is located.

    This function takes a PyTorch module as input and returns the device on which the module
    is located by inspecting the device of the first parameter.

    Args:
        module (torch.nn.Module): The PyTorch module for which to determine the device.

    Returns:
        torch.device: The device on which the specified module is located.
    """
    return next(module.parameters()).device


def move_to_device(obj: Any, device: Union[str, torch.device]):
    """
    Recursively moves tensors within containers to the specified device.

    This function takes an object and moves tensors contained within it to the specified
    device. It traverses the object recursively, identifying tensors and moving them
    to the specified device. Containers such as data classes, namedtuples, tuples,
    sequences, and mappings are also traversed, and their contained tensors are moved.

    Args:
        obj (Any): The object containing tensors or other containers to be moved.
        device (str or torch.device): The target device to which tensors will be moved.

    Returns:
        Any: An object with tensors moved to the specified device.
    """
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
    """
    Converts a flat index or array of flat indices into a tuple of coordinate tensors.

    Given a flat index or an array of flat indices and a shape, this function returns a
    tuple of tensors containing the coordinates that would be needed to index into a
    multi-dimensional array of the specified shape. The order of unraveling can be either
    row-major (C) or column-major (F).

    Args:
        index (int, torch.Tensor, or array-like): A flat index or an array of flat indices.
        shape (tuple[int, ...]): The shape of the target multi-dimensional array.
        order (Literal['C', 'F'], optional): The order in which the indices should be
            unraveled. 'C' stands for row-major (C-style) order, and 'F' stands for
            column-major (Fortran-style) order. Defaults to 'C'.

    Returns:
        tuple[torch.Tensor, ...]: A tuple of tensors containing the unraveled indices.
        Each tensor corresponds to a dimension in the input shape.

    Raises:
        ValueError: If the provided index or indices are out of bounds for the given shape.

    Example:
        >>> flat_indices = torch.tensor([8, 12])
        >>> shape = (3, 4, 5)
        >>> unravel_indices(flat_indices, shape)
        (tensor([0, 0]), tensor([1, 2]), tensor([3, 2]))
    """
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


def create_first_subword_mask(word_ids: list[Optional[int]]) -> torch.BoolTensor:
    """
    Creates a boolean mask tensor indicating the positions of the first subword token
    corresponding to each word of a sequence, based on the provided list of word_ids.

    Args:
        word_ids (list[Optional[int]]): A list of word IDs where each element
            indicates the word corresponding to a token. Special tokens added by the
            tokenizer are mapped to `None`, while other tokens are mapped to the index
            of their corresponding word. Several tokens might be mapped to the same
            word index if they are parts of that word.

    Returns:
        torch.BoolTensor: A boolean tensor with `True` at positions that correspond
        to the first subword token within each word, and `False` otherwise.

    Example:
        >>> word_ids = [None, 0, 1, 1, 2, None]
        >>> make_first_subword_mask(word_ids)
        tensor([False,  True,  True, False,  True, False])
    """
    mask = []
    previous_word_id = None
    for word_id in word_ids:
        is_first_subword = word_id is not None and word_id != previous_word_id
        mask.append(is_first_subword)
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
    moving elements corresponding to `True` values to the beginning of the tensor,
    and optionally filling the remaining positions with a specified value.

    This function takes an input tensor and a mask tensor of the same shape. It reorders
    the elements in the input tensor such that elements corresponding to `True` values
    in the mask are moved to the beginning of the tensor while maintaining their relative
    order. If the optional `empty_value` is provided, the remaining positions in the tensor
    will be filled with the specified value.

    Args:
        tensor (torch.Tensor): The input tensor to be defragmented.
        mask (torch.Tensor): A tensor of the same shape as `tensor`, where each `True`
            value indicates the positions to move to the beginning of the tensor.
        empty_value (Optional[Any], optional): Value to fill in remaining positions in
            the tensor. Defaults to None.

    Returns:
        torch.Tensor: A new tensor containing the rearranged elements from the input
        tensor based on the mask. Elements corresponding to `True` values in the mask
        are moved to the beginning, and remaining positions may be filled with the
        `empty_value` if provided.

    Example:
        >>> original_tensor = torch.tensor([10, 20, 30, 40, 50])
        >>> mask = torch.tensor([True, False, True, False, True])
        >>> defrag(original_tensor, mask, empty_value=0)
        tensor([10, 30, 50,  0,  0])
    """
    return defrag_(tensor=tensor.clone(), mask=mask, empty_value=empty_value)


def defrag_(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        *,
        empty_value: Optional[Any] = None,
) -> torch.Tensor:
    """
    Rearranges the elements in the input tensor in place based on the provided mask,
    moving elements corresponding to `True` values to the beginning of the tensor,
    and optionally filling the remaining positions with a specified value.

    This function takes an input tensor and a mask tensor of the same shape. It reorders
    the elements in the input tensor such that elements corresponding to `True` values
    in the mask are moved to the beginning of the tensor while maintaining their relative
    order. If the optional `empty_value` is provided, the remaining positions in the tensor
    will be filled with the specified value.

    Args:
        tensor (torch.Tensor): The input tensor to be defragmented.
        mask (torch.Tensor): A tensor of the same shape as `tensor`, where each `True`
            value indicates the positions to move to the beginning of the tensor.
        empty_value (Optional[Any], optional): Value to fill in remaining positions in
            the tensor. Defaults to None.

    Returns:
        torch.Tensor: The input tensor with rearranged elements based on the mask.
        Elements corresponding to `True` values in the mask are moved to the beginning,
        and remaining positions may be filled with the `empty_value` if provided.

    Example:
        >>> original_tensor = torch.tensor([10, 20, 30, 40, 50])
        >>> mask = torch.tensor([True, False, True, False, True])
        >>> defrag_(original_tensor, mask, empty_value=0)
        tensor([10, 30, 50,  0,  0])
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


def scale_and_shift_values(
        tensor: torch.Tensor,
        source_range: Optional[tuple[Number, Number]] = None,
        target_range: tuple[Number, Number] = (0, 1),
        clip_values: bool = True,
) -> torch.Tensor:
    """
    Scales and shifts the values of a tensor to a specified target value range.

    This function takes an input tensor and performs a transformation that scales and shifts
    its values from a specified source range to a desired target range. If the source range
    is not provided, the function automatically determines it using the minimum and maximum
    values of the input tensor. If the `clip_values` flag is set to True (default behavior)
    and a source range is provided, the tensor's values are clipped to the source range before
    performing the transformation.

    Args:
        tensor (torch.Tensor): The input tensor to be transformed.
        source_range (Optional[tuple[Number, Number]], optional): The source value range
            from which the tensor's values will be scaled and shifted. Defaults to None.
        target_range (tuple[Number, Number], optional): The target value range to which
            the tensor's values will be transformed. Defaults to (0, 1).
        clip_values (bool, optional): A flag indicating whether to clip the tensor's
            values to the source range if provided. Defaults to True.

    Returns:
        torch.Tensor: A new tensor with values scaled and shifted to the target value range.

    Example:
        >>> input_tensor = torch.tensor([0.1, 0.5, 0.9])
        >>> scaled_shifted_tensor = scale_and_shift_values(input_tensor, source_range=(0, 1), target_range=(-1, 1))
        >>> scaled_shifted_tensor
        tensor([-0.8,  0.0,  0.8])
    """
    tensor = tensor.float()
    if source_range is None:
        # if source_range was not provided, infer it from the input tensor values
        source_range = (tensor.min().item(), tensor.max().item())
    elif clip_values:
        # if source_range was provided and clip_values is true, clip input tensor values
        tensor = tensor.clip(*source_range)
    # scale and shift the tensor values to the target range
    if source_range != target_range:
        from_min, from_max = source_range
        to_min, to_max = target_range
        tensor -= from_min
        tensor /= from_max - from_min
        tensor *= to_max - to_min
        tensor += to_min
    # return the transformed tensor
    return tensor


def slerp(
        start: torch.Tensor,
        end: torch.Tensor,
        t: Union[float, torch.Tensor],
        interpolation_dim: Optional[bool] = None,
) -> torch.Tensor:
    """
    Performs Spherical Linear Interpolation (SLERP) between two vectors.

    Given two input vectors `start` and `end`, this function computes the result of
    Spherical Linear Interpolation (SLERP) based on the interpolation parameter `t`.
    The optional parameter `interpolation_dim` specifies the dimension along which
    the vectors are treated for the interpolation.

    Args:
        start (torch.Tensor): The starting vector for the interpolation.
        end (torch.Tensor): The ending vector for the interpolation.
        t (Union[float, torch.Tensor]): Interpolation parameter ranging between 0 and 1.
        interpolation_dim (Optional[int], optional): The dimension along which to interpolate
            the vectors. If provided, the interpolation will be performed along this dimension.
            Defaults to None.

    Returns:
        torch.Tensor: The interpolated vector between `start` and `end` based on the value of `t`.

    Note:
        - The vectors `start` and `end` are not required to be unit vectors.
        - This function performs SLERP, which ensures constant angular velocity during
          the interpolation on the unit hypersphere.
        - If `interpolation_dim` is provided, the interpolation is performed along that dimension.
        - The resulting tensor will have the same shape as the input tensors.

    References:
        - https://en.wikipedia.org/wiki/Slerp#Geometric_Slerp
    """
    keepdim = interpolation_dim is not None
    start_norm = torch.linalg.vector_norm(start, dim=interpolation_dim, keepdim=keepdim)
    end_norm = torch.linalg.vector_norm(end, dim=interpolation_dim, keepdim=keepdim)

    # Normalize vectors, handle zero norm vectors to avoid division by zero
    start_unit = torch.where(start_norm == 0, start, start / start_norm)
    end_unit = torch.where(end_norm == 0, end, end / end_norm)

    omega = torch.acos(torch.sum(start_unit * end_unit, dim=interpolation_dim))
    sin_omega = torch.sin(omega)

    t0 = torch.where(sin_omega == 0, 1 - t, torch.sin((1 - t) * omega) / sin_omega)
    t1 = torch.where(sin_omega == 0, t, torch.sin(t * omega) / sin_omega)

    if interpolation_dim is not None:
        t0.unsqueeze_(interpolation_dim)
        t1.unsqueeze_(interpolation_dim)

    return t0 * start + t1 * end


def smooth_values(
        input: torch.Tensor,
        dim: int = -1,
        dtype: Optional[torch.dtype] = None,
        beta: float = 0.98,
) -> torch.Tensor:
    """
    Applies exponential moving average smoothing to values along a specified dimension.

    Given an input tensor and a dimension along which to perform smoothing, this function
    applies exponential moving average smoothing to the values along the specified dimension.
    The smoothing factor is determined by the parameter `beta`. The result is a tensor with
    the same shape as the input tensor, containing smoothed values.

    Args:
        input (torch.Tensor): The input tensor containing values to be smoothed.
        dim (int, optional): The dimension along which to apply smoothing. Defaults to -1,
            which indicates the last dimension.
        dtype (Optional[torch.dtype], optional): The data type for the output tensor.
            If not provided, the same data type as the input tensor is used.
        beta (float, optional): The smoothing factor, ranging between 0 and 1. Smaller
            values make the smoothing respond faster to changes. Defaults to 0.98.

    Returns:
        torch.Tensor: A tensor with smoothed values along the specified dimension.

    Note:
        - The smoothed values are computed using exponential moving average.
        - If `dtype` is not provided, the output tensor will have the same data type
          as the input tensor.
        - The `dim` parameter specifies the dimension along which smoothing is applied.
          If `dim` is -1, smoothing is applied along the last dimension.
    """
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


def relative_positions(
        size: int,
        max_distance: Optional[int] = None,
        apply_offset: bool = False,
) -> torch.Tensor:
    """
    Computes a matrix of relative positions.

    Args:
        size (int): Size of the matrix (size x size).
        max_distance (Optional[int]): Maximum absolute value for the relative positions. 
            If provided, all values in the resulting matrix will be clipped between [-max_distance, max_distance].
        apply_offset (bool): If True, offsets the values in the matrix by `max_distance` so the range becomes [0, 2*max_distance].
            This is valid only if `max_distance` is provided.

    Returns:
        torch.Tensor: A matrix representing the relative positions.

    Example:
        If size=5, without any clipping or offset, the matrix will be:
        [[ 0,  1,  2,  3,  4],
         [-1,  0,  1,  2,  3],
         [-2, -1,  0,  1,  2],
         [-3, -2, -1,  0,  1],
         [-4, -3, -2, -1,  0]]

        With max_distance=2, the matrix will be:
        [[ 0,  1,  2,  2,  2],
         [-1,  0,  1,  2,  2],
         [-2, -1,  0,  1,  2],
         [-2, -2, -1,  0,  1],
         [-2, -2, -2, -1,  0]]

        With max_distance=2 and apply_offset=True, the matrix will be:
        [[2, 3, 4, 4, 4],
         [1, 2, 3, 4, 4],
         [0, 1, 2, 3, 4],
         [0, 0, 1, 2, 3],
         [0, 0, 0, 1, 2]]
    """
    position = torch.arange(size).expand(size, size)
    relative_matrix = position - position.T

    if max_distance is not None:
        relative_matrix = relative_matrix.clip(-max_distance, max_distance)

    if apply_offset:
        if max_distance is None:
            raise ValueError('Max distance needs to be provided for offset to be applied.')
        relative_matrix = relative_matrix + max_distance

    return relative_matrix


def remove_no_grad_decorator(f: F) -> F:
    """
    Removes 'torch.no_grad()' decorator from a given function or method.
    
    Args:
        f (F): The function or method decorated with `torch.no_grad()`.

    Returns:
        F: The original undecorated function or method.

    Raises:
        TypeError: If the closure is None, not structured as expected, or the argument type is invalid.
    """

    # Check if closure is None, which usually indicates that the function is not decorated
    if f.__closure__ is None:
        raise TypeError('Closure is None. This usually means the function is not decorated with `torch.no_grad()`.')

    # Check if closure is structured as expected for a `torch.no_grad()` decorated function
    elif len(f.__closure__) != 2:
        raise TypeError('Closure is not structured as expected. Check if the function is correctly decorated.')

    # Handling the case of a function
    elif isinstance(f, FunctionType):
        func = f.__closure__[1].cell_contents
        return cast(F, func)

    # Handling the case of a method
    elif isinstance(f, MethodType):
        obj = f.__self__
        func = f.__closure__[1].cell_contents
        return cast(F, MethodType(func, obj))

    else:
        raise TypeError('Invalid argument type. Expected a function or method decorated with `torch.no_grad()`.')
