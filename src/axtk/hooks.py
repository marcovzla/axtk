from typing import Any, Optional
from collections.abc import Callable, Iterable
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle


Args = tuple[Any, ...]
KwArgs = dict[str, Any]
Grads = tuple[Optional[Tensor], ...]

TensorHookCallable = Callable[['TensorHook', Tensor], Optional[Tensor]]

ModuleForwardHookCallable = Callable[['ModuleForwardHook', Module, Args, KwArgs, Any], Optional[Any]]
ModulePreForwardHookCallable = Callable[['ModulePreForwardHook', Module, Args, KwArgs], Optional[tuple[Any, KwArgs]]]
ModuleBackwardHookCallable = Callable[['ModuleBackwardHook', Module, Grads, Grads], Optional[Grads]]
ModulePreBackwardHookCallable = Callable[['ModulePreBackwardHook', Module, Grads], Optional[Tensor]]


class Hook:
    def __enter__(self):
        self.register_hook()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.unregister_hook()

    def __del__(self):
        if self.is_registered:
            self.unregister_hook()

    @property
    def is_registered(self) -> bool:
        """Returns True if hook is currently registered."""
        return False

    def register_hook(self):
        """Register hook."""
        if self.is_registered:
            raise Exception('hook is already registered')

    def unregister_hook(self):
        """Unregister hook."""
        if not self.is_registered:
            raise Exception('hook is not currently registered')


class HookManager(Hook):
    def __init__(self, hooks: Optional[Iterable[Hook]] = None):
        super().__init__()
        self._hooks: list[Hook] = []
        self._registered: bool = False
        self.hooks = hooks

    def __len__(self):
        return len(self.hooks)

    def __iter__(self):
        return iter(self.hooks)

    @property
    def hooks(self) -> list[Hook]:
        return self._hooks

    @hooks.setter
    def hooks(self, hs: Optional[Iterable[Hook]]):
        if self.is_registered:
            raise Exception('cannot swap hooks while manager is registered')
        self._hooks = [] if hs is None else list(hs)

    @property
    def is_registered(self) -> bool:
        return self._registered

    def register_hook(self):
        super().register_hook()
        self._registered = True
        for hook in self.hooks:
            hook.register_hook()

    def unregister_hook(self):
        super().unregister_hook()
        self._registered = False
        for hook in self.hooks:
            hook.unregister_hook()


class TorchHook(Hook):
    def __init__(self):
        super().__init__()
        self.handle: Optional[RemovableHandle] = None

    @property
    def is_registered(self) -> bool:
        return self.handle is not None

    def unregister_hook(self):
        super().unregister_hook()
        self.handle.remove()
        self.handle = None


class TensorHook(TorchHook):
    def __init__(
            self,
            tensor: Tensor,
            hook_function: Optional[TensorHookCallable] = None,
    ):
        super().__init__()
        self.tensor = tensor
        self._hook_function = hook_function

    def register_hook(self):
        super().register_hook()
        self.handle = self.tensor.register_hook(self.hook_function)

    def hook_function(self, tensor: Tensor) -> Optional[Tensor]:
        if self._hook_function is None:
            raise NotImplementedError
        return self._hook_function(self, tensor)


class ModuleForwardHook(TorchHook):
    def __init__(
            self,
            module: Module,
            hook_function: Optional[ModuleForwardHookCallable] = None,
            prepend: bool = False,
    ):
        super().__init__()
        self.module = module
        self._hook_function = hook_function
        self.prepend = prepend

    def register_hook(self):
        super().register_hook()
        self.handle = self.module.register_forward_hook(
            hook=self.hook_function,
            prepend=self.prepend,
            with_kwargs=True,
        )

    def hook_function(
            self,
            module: Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
    ) -> Optional[Any]:
        if self._hook_function is None:
            raise NotImplementedError
        return self._hook_function(self, module, args, kwargs, output)


class ModulePreForwardHook(TorchHook):
    def __init__(
            self,
            module: Module,
            hook_function: Optional[ModulePreForwardHookCallable] = None,
            prepend: bool = False,
    ):
        super().__init__()
        self.module = module
        self._hook_function = hook_function
        self.prepend = prepend

    def register_hook(self):
        super().register_hook()
        self.handle = self.module.register_forward_pre_hook(
            hook=self.hook_function,
            prepend=self.prepend,
            with_kwargs=True,
        )

    def hook_function(
            self,
            module: Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
    ) -> Optional[tuple[Any, dict[str, Any]]]:
        if self._hook_function is None:
            raise NotImplementedError
        return self._hook_function(self, module, args, kwargs)


class ModuleBackwardHook(TorchHook):
    def __init__(
            self,
            module: Module,
            hook_function: Optional[ModuleBackwardHookCallable] = None,
            prepend: bool = False,
    ):
        super().__init__()
        self.module = module
        self._hook_function = hook_function
        self.prepend = prepend

    def register_hook(self):
        super().register_hook()
        self.handle = self.module.register_full_backward_hook(
            hook=self.hook_function,
            prepend=self.prepend,
        )

    def hook_function(
            self,
            module: Module,
            grad_input: tuple[Optional[Tensor], ...],
            grad_output: tuple[Optional[Tensor], ...],
    ) -> Optional[tuple[Tensor]]:
        if self._hook_function is None:
            raise NotImplementedError
        return self._hook_function(self, module, grad_input, grad_output)


class ModulePreBackwardHook(TorchHook):
    def __init__(
            self,
            module: Module,
            hook_function: Optional[ModulePreBackwardHookCallable] = None,
            prepend: bool = False,
    ):
        super().__init__()
        self.module = module
        self._hook_function = hook_function
        self.prepend = prepend

    def register_hook(self):
        super().register_hook()
        self.handle = self.module.register_full_backward_pre_hook(
            hook=self.hook_function,
            prepend=self.prepend,
        )

    def hook_function(
            self,
            module: Module,
            grad_output: tuple[Optional[Tensor], ...],
    ) -> Optional[Tensor]:
        if self._hook_function is None:
            raise NotImplementedError
        return self._hook_function(self, module, grad_output)
