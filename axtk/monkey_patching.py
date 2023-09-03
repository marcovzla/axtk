from types import FunctionType, MethodType
from typing import Any, Optional
from functools import update_wrapper


def copy_function(f: FunctionType) -> FunctionType:
    f_copy = FunctionType(f.__code__, f.__globals__, f.__name__, f.__defaults__, f.__closure__)
    f_copy.__kwdefaults__ = f.__kwdefaults__
    return update_wrapper(f_copy, f)


def patch_class(
        cls: type,
        f: FunctionType,
        f_name: Optional[str] = None,
        as_classmethod: bool = False,
        as_property: bool = False,
):
    if as_classmethod and as_property:
        raise ValueError('as_classmethod and as_property cannot be both set to true')
    if f_name is None:
        f_name = f.__name__
    c_name = cls.__name__
    f_copy = copy_function(f)
    f_copy.__qualname__ = f'{c_name}.{f_name}'
    if as_classmethod:
        setattr(cls, f_name, MethodType(f_copy, cls))
    elif as_property:
        setattr(cls, f_name, property(f_copy))
    else:
        setattr(cls, f_name, f_copy)


def patch_object(
        obj: Any,
        f: FunctionType,
        f_name: Optional[str] = None,
):
    if f_name is None:
        f_name = f.__name__
    c_name = type(obj).__name__
    f_copy = copy_function(f)
    f_copy.__qualname__ = f'{c_name}.{f_name}'
    setattr(obj, f_name, MethodType(f_copy, obj))


def patch_dunder(obj: Any, **kwargs: FunctionType):
    cls = type(obj)
    new_cls = type(f'Patched_{cls.__name__}', (cls,), {})
    for f_name, f in kwargs.items():
        patch_class(new_cls, f, f_name)
    obj.__class__ = new_cls
