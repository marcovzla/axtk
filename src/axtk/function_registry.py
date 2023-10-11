from collections.abc import Callable
from typing import Any, Optional
import json
import inspect
from inspect import Parameter
import docstring_parser
from pydantic import create_model
from pydantic.fields import FieldInfo



class FunctionRegistry:
    """A registry to store and manage Python functions."""

    def __init__(self, functions: Optional[list[Callable]] = None) -> None:
        """Initialize an empty registry."""
        self._functions: dict[str, Callable] = {}

        # register functions if provided
        if functions is not None:
            for f in functions:
                self.register(f)

    def __len__(self):
        return len(self._functions)

    def register(self, f: Optional[Callable] = None, name: Optional[str] = None) -> Callable:
        """
        Register a function with an optional name. This method can be used as a decorator.

        Args:
            f (Optional[Callable], default=None): The function to register.
            name (Optional[str], default=None): The name to register the callable under.
                Uses the function's name if None.

        Returns:
            Callable: The registered function.
        """
        def inner_register(func: Callable) -> Callable:
            nonlocal name
            name = name if name is not None else func.__name__
            self._functions[name] = func
            return func

        # Works as both a decorator and a function call
        return inner_register if f is None else inner_register(f)

    def json_schema(self) -> list[dict[str, Any]]:
        """
        Generate JSON schemas for all registered functions.

        Returns:
            list[dict[str, Any]]: List of JSON schemas for the registered functions.
        """
        return [
            generate_json_schema(f, name=name)
            for name, f in self._functions.items()
        ]

    def call(self, name: str, **kwargs) -> Any:
        if f := self._functions.get(name):
            return f(**kwargs)
        raise ValueError(f'unknown function: {name}')

    def gen_response(self, msg: dict[str, str]) -> dict[str, str]:
        name = msg['name']
        kwargs = json.loads(msg['arguments'])
        return {
            'role': 'function',
            'name': name,
            'content': self.call(name, **kwargs),
        }



# https://platform.openai.com/docs/guides/gpt/function-calling
# https://json-schema.org/understanding-json-schema/

def generate_json_schema(f: Callable, name: Optional[str] = None) -> dict[str, Any]:
    """
    Generate a JSON schema for a Python callable.

    Args:
        f (Callable): The Python callable for which the schema is generated.
        name (Optional[str], default=None): The name to be used for the schema.
            Defaults to the callable's name if None.

    Returns:
        dict[str, Any]: A dictionary containing the JSON schema for the callable,
        including its name, description, and parameters.
    """
    name = name if name is not None else f.__name__

    doc = docstring_parser.parse(f.__doc__)
    description = doc.short_description or doc.long_description
    param_descriptions = {p.arg_name: p.description for p in doc.params}

    field_definitions = {
        p.name: (
            p.annotation,
            FieldInfo(
                annotation=p.annotation,
                default=p.default if p.default != Parameter.empty else ...,
                description=param_descriptions.get(p.name, ...),
            )
        ) for p in inspect.signature(f).parameters.values()
    }

    model = create_model(f'Input for `{name}`', **field_definitions)

    return {
        'name': name,
        'description': description,
        'parameters': model.model_json_schema(),
    }
