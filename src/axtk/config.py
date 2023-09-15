from __future__ import annotations
from typing import Any, Union, Optional
from collections.abc import Mapping, MutableMapping, Iterator
import json
import copy
from dotenv import dotenv_values
from axtk.typing import PathLike


CONFIG_FIELDS_NAME = '_config_fields'
SEPARATOR = '.'


class Config(MutableMapping):
    """A Config class that acts like a dictionary but provides additional features."""

    def __init__(self, *args, **kwargs):
        self._config_fields: dict[str, Any] = {}
        # initialize with class attributes
        for name in vars(type(self)):
            if not name.startswith('_'):
                self[name] = getattr(type(self), name)
        # add provided entries
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def __repr__(self):
        return f'{self.__class__.__name__}({self._config_fields})'

    def __len__(self):
        return len(self._config_fields)

    def __iter__(self):
        return iter(self._config_fields)

    def __getitem__(self, key: str):
        try:
            keys = key.split(SEPARATOR, maxsplit=1)
            item = self._config_fields[keys[0]]
            return item[keys[1]] if len(keys) == 2 else item
        except:
            raise KeyError(key)

    def __setitem__(self, key: str, value: Any):
        if isinstance(value, Mapping):
            value = self.new_config(value)
        keys = key.split(SEPARATOR, maxsplit=1)
        if len(keys) == 1:
            value = self.enforce_annotation(keys[0], value)
            self._config_fields[keys[0]] = value
        elif keys[0] in self:
            self._config_fields[keys[0]][keys[1]] = value
        else:
            self._config_fields[keys[0]] = self.new_config({keys[1]: value})

    def __delitem__(self, key: str):
        try:
            keys = key.split(SEPARATOR, maxsplit=1)
            if len(keys) == 1:
                del self._config_fields[keys[0]]
            else:
                config = self._config_fields[keys[0]]
                del config[keys[1]]
        except:
            raise KeyError(key)
    
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any):
        if name == CONFIG_FIELDS_NAME:
            super().__setattr__(CONFIG_FIELDS_NAME, value)
        else:
            self[name] = value

    def __delattr__(self, name: str):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __copy__(self):
        # Create a new instance without calling __init__
        new_copy = self.__class__.__new__(self.__class__)
        # Manually copying _config_fields (shallow copy)
        new_copy._config_fields = dict(self._config_fields)
        return new_copy

    def __deepcopy__(self, memo: dict[int, Any]):
        # Avoiding recursive deep copies
        if id(self) in memo:
            return memo[id(self)]
        # Creating a new instance without calling __init__
        new_copy = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_copy
        # Manually copying _config_fields (deep copy)
        new_copy._config_fields = copy.deepcopy(self._config_fields, memo)
        return new_copy

    def __or__(self, other: Mapping):
        new_config = copy.deepcopy(self)
        new_config.update(other)
        return new_config

    def __ror__(self, other: Mapping):
        new_config = self.new_config(copy.deepcopy(other))
        new_config.update(self)
        return new_config

    def __ior__(self, other: Mapping):
        self.update(other)
        return self

    def new_config(self, *args, **kwargs):
        return self.__class__(*args, **kwargs)

    def copy(self, *, shallow: bool = False):
        return copy.copy(self) if shallow else copy.deepcopy(self)

    def update(self, other: Mapping):
        other = self.new_config(other)
        for key, value in other.items():
            if key in self and isinstance(self[key], Config) and isinstance(value, Config):
                self[key].update(value)
            else:
                self[key] = value

    def fields(self) -> Iterator[str]:
        for key, value in self.items():
            if isinstance(value, Config):
                for field in value.fields():
                    yield f'{key}{SEPARATOR}{field}'
            else:
                yield key

    def annotation(self, key: str) -> Optional[Any]:
        if hasattr(self, '__annotations__'):
            return self.__annotations__.get(key)

    def enforce_annotation(self, key: str, value: Any) -> Any:
        if annotation := self.annotation(key):
            if isinstance(annotation, type):
                return annotation(value)
        return value

    @classmethod
    def from_dotenv(cls, *paths: PathLike, keepcase: bool = False):
        if paths:
            values = dict()
            for path in paths:
                values.update(dotenv_values(path))
        else:
            values = dotenv_values()
        if not keepcase:
            values = {k.lower(): v for k, v in values.items()}
        return cls(values)

    @classmethod
    def from_json(cls, s: str):
        """Load a Config object from a JSON string."""
        return json.loads(s, cls=ConfigJSONDecoder, config_cls=cls)

    def to_json(self, *, ensure_ascii: bool = False, indent: Optional[Union[int, str]] = None):
        """Convert the Config object to a JSON string."""
        return json.dumps(self, ensure_ascii=ensure_ascii, indent=indent, cls=ConfigJSONEncoder)


class ConfigJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Config):
            return dict(obj)
        return super().default(obj)


class ConfigJSONDecoder(json.JSONDecoder):
    def __init__(self, config_cls=Config, **kwargs):
        super().__init__(object_hook=self.object_hook, **kwargs)
        self.config_cls = config_cls
    
    def object_hook(self, dct):
        return self.config_cls(dct)
