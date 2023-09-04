from typing import Any, Optional
from collections.abc import Callable, Iterable
from functools import partial
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle


Args = tuple[Any, ...]
KwArgs = dict[str, Any]
Grads = tuple[Optional[Tensor], ...]

TensorHookCallable = Callable[['TensorHook', Tensor], Optional[Tensor]]

ModuleForwardHookCallable = Callable[['ModuleForwardHook', Module, Args, KwArgs, Any], Optional[Any]]
ModulePreForwardHookCallable = Callable[['ModulePreForwardHook', Module, Args, KwArgs], Optional[tuple[Args, KwArgs]]]
ModuleBackwardHookCallable = Callable[['ModuleBackwardHook', Module, Grads, Grads], Optional[Grads]]
ModulePreBackwardHookCallable = Callable[['ModulePreBackwardHook', Module, Grads], Optional[Tensor]]


class Hook:
    def __enter__(self):
        self.register()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.unregister()

    def __del__(self):
        self.unregister()

    def register(self):
        raise NotImplementedError

    def unregister(self):
        raise NotImplementedError


class HookManager(Hook):
    def __init__(self, hooks: Iterable[Hook]):
        super().__init__()
        self.hooks = list(hooks)

    def __len__(self):
        return len(self.hooks)

    def __iter__(self):
        return iter(self.hooks)

    def register(self):
        for hook in self.hooks:
            hook.register()

    def unregister(self):
        for hook in self.hooks:
            hook.unregister()


class TorchHook(Hook):
    def __init__(self):
        super().__init__()
        self.handle: Optional[RemovableHandle] = None

    def unregister(self):
        self.handle.remove()
        self.handle = None


class TensorHook(TorchHook):
    def __init__(
            self,
            tensor: Tensor,
            hook: TensorHookCallable,
    ):
        super().__init__()
        self.tensor = tensor
        self.hook = partial(hook, self)

    def register(self):
        self.handle = self.tensor.register_hook(self.hook)


class ModuleForwardHook(TorchHook):
    def __init__(
            self,
            module: Module,
            hook: ModuleForwardHookCallable,
            prepend: bool = False,
    ):
        super().__init__()
        self.module = module
        self.hook = partial(hook, self)
        self.prepend = prepend

    def register(self):
        self.handle = self.module.register_forward_hook(
            hook=self.hook,
            prepend=self.prepend,
            with_kwargs=True,
        )


class ModulePreForwardHook(TorchHook):
    def __init__(
            self,
            module: Module,
            hook: ModulePreForwardHookCallable,
            prepend: bool = False,
    ):
        super().__init__()
        self.module = module
        self.hook = partial(hook, self)
        self.prepend = prepend

    def register(self):
        self.handle = self.module.register_forward_pre_hook(
            hook=self.hook,
            prepend=self.prepend,
            with_kwargs=True,
        )


class ModuleBackwardHook(TorchHook):
    def __init__(
            self,
            module: Module,
            hook: ModuleBackwardHookCallable,
            prepend: bool = False,
    ):
        super().__init__()
        self.module = module
        self.hook = partial(hook, self)
        self.prepend = prepend

    def register(self):
        self.handle = self.module.register_full_backward_hook(
            hook=self.hook,
            prepend=self.prepend,
        )


class ModulePreBackwardHook(TorchHook):
    def __init__(
            self,
            module: Module,
            hook: ModulePreBackwardHookCallable,
            prepend: bool = False,
    ):
        super().__init__()
        self.module = module
        self.hook = partial(hook, self)
        self.prepend = prepend

    def register(self):
        self.handle = self.module.register_full_backward_pre_hook(
            hook=self.hook,
            prepend=self.prepend,
        )
