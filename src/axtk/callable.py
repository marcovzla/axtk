import inspect
from copy import copy
from typing import Any, Optional, Union
from collections.abc import Callable


KEYWORD_PARAMETERS = {
    inspect.Parameter.KEYWORD_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
}


def get_caller_name(stack_level: int = 0) -> str:
    """
    Get the calling function's name at a specified stack level.

    Args:
        stack_level (int): The stack level to inspect, where 0 represents the calling function.

    Returns:
        str: The calling function's name.
    """
    return inspect.stack()[stack_level+1].function


def get_caller(stack_level: int = 0):
    """
    Get the calling function at a specified stack level.

    Args:
        stack_level (int): The stack level to inspect, where 0 represents the calling function.

    Returns:
        Callable: The calling function.

    Raises:
        Exception: If the caller function cannot be found in the call stack.
    """
    info = inspect.stack()[stack_level+1]
    frame = info.frame
    name = info.function
    while frame:
        if f := frame.f_globals.get(name):
            if f.__code__ == info.frame.f_code:
                return f
        for f in frame.f_locals.values():
            if getattr(f, '__code__', None) == info.frame.f_code:
                return f
            if f := getattr(f, name, None):
                if f.__code__ == info.frame.f_code:
                    return f
        frame = frame.f_back
    raise Exception('caller not found')


def get_caller_arguments(
        stack_level: int = 0,
        caller: Optional[Callable] = None,
        return_kwargs_only: bool = False,
) -> Union[dict[str, Any], tuple[tuple[Any, ...], dict[str, Any]]]:
    """
    Get the arguments passed to the calling function at a specified stack level.

    Args:
        stack_level (int): The stack level to inspect, where 0 represents the calling function.
        caller (Optional[Callable]): The calling function. If not provided, it will be determined automatically.
        return_kwargs_only (bool): If True, all arguments are returned as keyword arguments; otherwise, positional
            and keyword arguments are returned separately. Defaults to False.

    Returns:
        Union[dict[str, Any], tuple[tuple[Any, ...], dict[str, Any]]]: The collected arguments. If return_kwargs_only
        is True, returns a dictionary of keyword arguments. Otherwise, returns a tuple containing a tuple of
        positional arguments and a dictionary of keyword arguments.
    """
    # Get frame information
    stack = inspect.stack()
    frame_info = stack[stack_level+1]
    frame = frame_info.frame

    # Get the calling function
    if caller is None:
        caller = get_caller(stack_level+1)

    # Get the signature of the calling function
    signature = inspect.signature(caller)

    # Collect positional and keyword arguments
    args, kwargs = [], {}
    for param in signature.parameters.values():
        # Get the argument value
        value = frame.f_locals[param.name]

        # Store keyword argument unless it *must* be positional
        if param.kind == inspect.Parameter.POSITIONAL_ONLY and not return_kwargs_only:
            args.append(value)
        else:
            kwargs[param.name] = value

    # Return collected arguments
    return kwargs if return_kwargs_only else (tuple(args), kwargs)


def construct_kwargs(
        prefix: str,
        arguments: dict[str, Any],
        sep: str = '_',
) -> dict[str, Any]:
    """Builds a kwargs dictionary with all the arguments that match a given prefix."""
    # either get kwargs from the arguments, or make a new one
    kwargs = copy(arguments.get(prefix, {}))
    if not isinstance(kwargs, dict):
        raise TypeError(f'{prefix} is not a dict')
    # append separator to prefix
    prefix += sep
    # add arguments that start with prefix
    for key, value in arguments.items():
        if key.startswith(prefix):
            name = key[len(prefix):]
            if not name: continue
            kwargs[name] = value
    # return results
    return kwargs


def requires_positional_arguments(f: Callable) -> bool:
    """
    Returns True if the callable requires at least one positional argument,
    i.e., there is at least one positional parameter with no default value.
    """
    signature = inspect.signature(f)
    for param in signature.parameters.values():
        if (
            param.kind == inspect.Parameter.POSITIONAL_ONLY
            and param.default == inspect.Parameter.empty
        ):
            return True
    return False


def accepts_any_keyword_argument(f: Callable) -> bool:
    """Returns True if callable accepts any keyword argument, i.e. **kwargs."""
    signature = inspect.signature(f)
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def keyword_argument_names(f: Callable) -> list[str]:
    """Returns a list with the callable's keyword argument names."""
    names = []
    signature = inspect.signature(f)
    for param in signature.parameters.values():
        if param.kind in KEYWORD_PARAMETERS:
            names.append(param.name)
    return names


def keyword_argument_is_required(name: str, f: Callable) -> bool:
    """
    Returns True if the keyword argument is required,
    i.e., it is a keyword argument and it does not have a default value.
    """
    signature = inspect.signature(f)
    param = signature.parameters.get(name)
    return (
        param is not None
        and param.kind in KEYWORD_PARAMETERS
        and param.default == inspect.Parameter.empty
    )
