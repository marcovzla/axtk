from types import FunctionType, MethodType
from typing import Any, Optional, cast
from functools import update_wrapper
from axtk.typing import F


def patch_object(
        obj: Any,
        f: F,
        f_name: Optional[str] = None,
):
    """
    Patch an object by adding or replacing a method.

    Args:
        obj (Any): The object to patch.
        f (F): The function or method to add or replace.
        f_name (str, optional): The name to use for the patched method. Defaults to None (uses the original name).
    """
    if f_name is None:
        f_name = f.__name__
    c_name = type(obj).__name__
    f_copy = copy_or_wrap(f)
    f_copy.__qualname__ = f'{c_name}.{f_name}'
    setattr(obj, f_name, MethodType(f_copy, obj))


def patch_class(
        cls: type,
        f: F,
        f_name: Optional[str] = None,
        as_classmethod: bool = False,
        as_property: bool = False,
):
    """
    Patch a class by adding or replacing a method.

    Args:
        cls (type): The class to patch.
        f (F): The function or method to add or replace.
        f_name (str, optional): The name to use for the patched method. Defaults to None (uses the original name).
        as_classmethod (bool, optional): If True, add the method as a class method. Defaults to False.
        as_property (bool, optional): If True, add the method as a property. Defaults to False.

    Raises:
        ValueError: If both as_classmethod and as_property are set to True.
    """
    if as_classmethod and as_property:
        raise ValueError('as_classmethod and as_property cannot be both set to True')
    if f_name is None:
        f_name = f.__name__
    c_name = cls.__name__
    f_copy = copy_or_wrap(f)
    f_copy.__qualname__ = f'{c_name}.{f_name}'
    if as_classmethod:
        setattr(cls, f_name, MethodType(f_copy, cls))
    elif as_property:
        setattr(cls, f_name, property(f_copy))
    else:
        setattr(cls, f_name, f_copy)


def patch_dunder_methods(obj: Any, **kwargs: F):
    """
    Patch the dunder methods of an object with provided functions.

    This function creates a new class that inherits from the original class of the object
    and MonkeyPatchedDunderMethods. It then assigns the provided functions to the specified
    dunder method names for the new class and sets the object's class to the new patched class.

    Parameters:
        obj (Any): The object to patch.
        **kwargs (Callable): Keyword arguments where keys are dunder method names and values
            are the functions to patch with.
    """
    cls = type(obj)
    patched_cls = type(f'Patched_{cls.__name__}', (cls, MonkeyPatchedDunderMethods), {})
    for name, f in kwargs.items():
        patch_class(patched_cls, f, name)
    obj.__class__ = patched_cls


def restore_dunder_methods(obj: Any, recursive_restore: bool = False):
    """
    Restore the original class of an object by removing monkey-patched dunder methods.

    This function restores the object's class to its immediate base class or repeatedly
    to its ancestors until it no longer has monkey-patched dunder methods.

    Parameters:
        obj (Any): The object whose dunder methods need to be restored.
        recursive_restore (bool, optional): If True, recursively restores the object's class
            to its base classes until no monkey-patched dunder methods are found.
            Otherwise, only the immediate base class is used. Default is False.
    """
    while has_patched_dunder_methods(obj):
        obj.__class__ = obj.__class__.__base__
        if not recursive_restore:
            break


class MonkeyPatchedDunderMethods:
    """A base class for objects with monkey-patched dunder methods."""


def has_patched_dunder_methods(obj: Any) -> bool:
    """
    Check if an object has monkey-patched dunder methods.

    Parameters:
        obj (Any): The object to check.

    Returns:
        bool: True if the object has monkey-patched dunder methods, False otherwise.
    """
    return isinstance(obj, MonkeyPatchedDunderMethods)


def copy_or_wrap(f: F) -> F:
    """
    Copy or wrap a function or method depending on its type.

    Args:
        f (F): The function or method to copy or wrap.

    Returns:
        F: A copy of the input function if it's a function, or a wrapped function if it's a method.
    
    Raises:
        TypeError: If the input is not a function or method.
    """
    if isinstance(f, FunctionType):
        return cast(F, copy_function(f))
    elif isinstance(f, MethodType):
        return cast(F, wrap_method(f))
    else:
        raise TypeError('Input must be a function or a method')


def copy_function(f: FunctionType) -> FunctionType:
    """
    Create a copy of a function.

    Args:
        f (FunctionType): The function to copy.

    Returns:
        FunctionType: A copy of the input function.
    """
    f_copy = FunctionType(f.__code__, f.__globals__, f.__name__, f.__defaults__, f.__closure__)
    f_copy.__kwdefaults__ = f.__kwdefaults__
    return update_wrapper(f_copy, f)


def wrap_method(m: MethodType) -> FunctionType:
    """
    Wrap a bound method in a function that can be used for monkeypatching.

    Args:
        m (MethodType): The bound method to wrap.

    Returns:
        FunctionType: A wrapped function that can be used for monkeypatching.
    """
    def f(obj, *args, **kwargs):
        return m(obj, *args, **kwargs)
    return f
