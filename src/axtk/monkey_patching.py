from types import FunctionType, MethodType
from typing import Any, Optional, Union
from functools import update_wrapper


FunctionOrMethod = Union[FunctionType, MethodType]


def copy_function(f: FunctionType) -> FunctionType:
    f_copy = FunctionType(f.__code__, f.__globals__, f.__name__, f.__defaults__, f.__closure__)
    f_copy.__kwdefaults__ = f.__kwdefaults__
    return update_wrapper(f_copy, f)


def wrap_method(m: MethodType) -> FunctionType:
    """Wraps a bound method in a function that can be used for monkeypatching."""
    def f(obj, *args, **kwargs):
        return m(obj, *args, **kwargs)
    return f


def copy_or_wrap(f: FunctionOrMethod) -> FunctionType:
    if isinstance(f, FunctionType):
        return copy_function(f)
    elif isinstance(f, MethodType):
        return wrap_method(f)
    else:
        raise TypeError


def patch_class(
        cls: type,
        f: FunctionOrMethod,
        f_name: Optional[str] = None,
        as_classmethod: bool = False,
        as_property: bool = False,
):
    if as_classmethod and as_property:
        raise ValueError('as_classmethod and as_property cannot be both set to true')
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


def patch_object(
        obj: Any,
        f: FunctionOrMethod,
        f_name: Optional[str] = None,
):
    if f_name is None:
        f_name = f.__name__
    c_name = type(obj).__name__
    f_copy = copy_or_wrap(f)
    f_copy.__qualname__ = f'{c_name}.{f_name}'
    setattr(obj, f_name, MethodType(f_copy, obj))


class MonkeyPatchedDunderMethods:
    pass


def has_patched_dunder_methods(obj: Any) -> bool:
    return isinstance(obj, MonkeyPatchedDunderMethods)


def patch_dunder_methods(obj: Any, **kwargs: FunctionOrMethod):
    cls = type(obj)
    patched_cls = type(f'Patched_{cls.__name__}', (cls, MonkeyPatchedDunderMethods), {})
    for f_name, f in kwargs.items():
        patch_class(patched_cls, f, f_name)
    obj.__class__ = patched_cls


def restore_dunder_methods(obj: Any):
    while has_patched_dunder_methods(obj):
        obj.__class__ = obj.__class__.__base__
