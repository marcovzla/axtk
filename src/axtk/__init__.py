import os
import sys
import random
import numpy as np
import torch


__all__ = [
    'running_in_notebook',
    'running_in_colab',
    'set_seed',
    'enable_full_determinism',
]


def running_in_notebook() -> bool:
    """
    Determine if the current environment is a Jupyter Notebook.

    Returns:
        bool: True if running in a Jupyter Notebook, False otherwise.

    Notes:
        The function checks for the existence of the 'IPython' module 
        and the presence of 'IPKernelApp' in its configuration to infer
        if the code is being executed within a Jupyter Notebook.
    """
    try:
        get_ipython = sys.modules['IPython'].get_ipython
        return 'IPKernelApp' in get_ipython().config
    except KeyError:
        # 'IPython' not in sys.modules
        return False
    except AttributeError:
        # get_ipython or config not available
        return False


def running_in_colab() -> bool:
    """
    Determine if the current environment is Google Colab.

    Returns:
        bool: True if running in Google Colab, False otherwise.
    """
    return 'google.colab' in sys.modules


def set_seed(seed: int) -> torch.Generator:
    """
    Sets the random number generator seeds for Python, NumPy, and PyTorch.

    This function takes an integer seed value and sets the random number generator seeds
    for Python's built-in `random` module, NumPy's random module, and PyTorch's random module.
    The provided seed value ensures reproducibility of random number generation across
    different libraries and functions.

    Args:
        seed (int): The seed value to initialize the random number generators.

    Returns:
        torch.Generator: A PyTorch random number generator with the specified seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    return torch.manual_seed(seed)


def enable_full_determinism(seed: int, warn_only: bool = False):
    """
    Enables full determinism in PyTorch operations for reproducible results.

    This function configures various settings within the PyTorch environment to ensure
    full determinism in computations. By setting a common seed and modifying relevant
    environment variables, it aims to make PyTorch operations consistent and reproducible.
    This is especially useful for debugging and achieving consistent results across runs.

    Args:
        seed (int): The seed value to initialize the random number generators.
        warn_only (bool, optional): If True, warnings about non-deterministic operations
            will be displayed, but the operations will not be disabled. Defaults to False.

    Note:
        - Enabling full determinism might impact performance due to certain optimizations
          being disabled.
        - CUDA-based operations and libraries are also configured for determinism.
    """
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
