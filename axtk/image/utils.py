from typing import Optional, Union, Literal, Any
from numbers import Number
from collections.abc import Sequence, Iterable
import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from axtk.typing import PathLike
from axtk.utils import is_pathlike


# imagenet stats
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ImageLike = Union[Image.Image, torch.Tensor, np.ndarray]


def normalize(
        tensor: torch.Tensor,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        inplace: bool = False,
) -> torch.Tensor:
    """
    Normalize a float tensor image with mean and standard deviation.
    Defaults to ImageNet statistics.
    """
    return F.normalize(tensor=tensor, mean=mean, std=std, inplace=inplace)


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


def to_pt_image(obj: Union[ImageLike, PathLike]) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj
    elif Image.isImageType(obj) or isinstance(obj, np.ndarray):
        return F.to_tensor(obj)
    elif is_pathlike(obj):
        return read_image(obj, return_type='pt')
    else:
        raise TypeError(f'unknown type: {type(obj).__qualname__}')


def to_np_image(obj: Union[ImageLike, PathLike]) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        return obj
    elif Image.isImageType(obj):
        return F.to_tensor(obj).numpy()
    elif isinstance(obj, torch.Tensor):
        return obj.numpy(force=True)
    elif is_pathlike(obj):
        return read_image(obj, return_type='np')
    else:
        raise TypeError(f'unknown type: {type(obj).__qualname__}')


def to_pil_image(obj: Union[ImageLike, PathLike]) -> np.ndarray:
    if Image.isImageType(obj):
        return obj
    elif isinstance(obj, np.ndarray):
        return Image.fromarray(obj)
    elif isinstance(obj, torch.Tensor):
        return F.to_pil_image(obj)
    elif is_pathlike(obj):
        return read_image(obj, return_type='pil')
    else:
        raise TypeError(f'unknown type: {type(obj).__qualname__}')


def read_image(
        path: PathLike,
        size: Optional[Union[int, tuple[int, int]]] = None,
        return_type: Literal['pt', 'np', 'pil'] = 'pil',
) -> ImageLike:
    img = Image.open(path)
    if size is not None:
        if isinstance(size, int):
            size = scale_size(img.size, new_max=size)
        img = img.resize(size)
    if return_type == 'pil':
        return img
    elif return_type == 'pt':
        return F.to_tensor(img)
    elif return_type == 'np':
        return F.to_tensor(img).numpy()
    else:
        raise ValueError(f'invalid return_type: {return_type}')


def scale_size(
        size: tuple[int, int],
        *,
        new_min: Optional[int] = None,
        new_max: Optional[int] = None,
) -> tuple[int, int]:
    if new_min is None and new_max is None:
        raise ValueError('either new_min or new_max must be provided')
    elif new_min is not None and new_max is not None:
        raise ValueError('either new_min or new_max must be provided, but not both')
    elif new_min is not None:
        if size[0] < size[1]:
            return new_min, round(new_min * size[1] / size[0])
        elif size[0] > size[1]:
            return round(new_min * size[0] / size[1]), new_min
        else:
            return new_min, new_min
    elif new_max is not None:
        if size[0] < size[1]:
            return round(new_max * size[0] / size[1]), new_max
        elif size[0] > size[1]:
            return new_max, round(new_max * size[1] / size[0])
        else:
            return new_max, new_max


def show_image(img: ImageLike, figsize: Optional[Sequence[float]] = None):
    img = to_np_image(img)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.xticks([], [])
    plt.yticks([], [])


def show_grid(
        images: list[ImageLike],
        nrow: int = 8,
        padding: int = 2,
        pad_value: float = 0,
        figsize: Optional[Sequence[float]] = None,
):
    """Displays a grid of images."""
    images = pad_to_max(map(to_pt_image, images))
    grid = make_grid(images, nrow=nrow, padding=padding, pad_value=pad_value)
    show_image(grid, figsize=figsize)


def pad_to_max(images: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    images = list(images)
    max_width = max(img.shape[-1] for img in images)
    max_height = max(img.shape[-2] for img in images)
    shape = (max_height, max_width)
    return [
        im if im.shape[-2:] == shape else F.center_crop(im, shape)
        for im in images
    ]


def crop_to_min(images: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    images = list(images)
    min_width = min(img.shape[-1] for img in images)
    min_height = min(img.shape[-2] for img in images)
    shape = (min_height, min_width)
    return [
        im if im.shape[-2:] == shape else F.center_crop(im, shape)
        for im in images
    ]
