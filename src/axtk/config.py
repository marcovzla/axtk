from __future__ import annotations
from typing import Any, Union, Optional
from collections.abc import Mapping, Iterator, Sequence
import json
import copy
from dotenv import dotenv_values
from axtk.typing import PathLike


CONFIG_FIELDS_NAME = '_config_fields'
SEPARATOR = '.'


class Config(Mapping):
    """A Config class that acts like a dictionary but provides additional features."""

    def __init__(self, *args, **kwargs):
        self._config_fields: dict[str, Any] = {}
        self.update(self.defaults() | dict(*args, **kwargs))

    def __repr__(self):
        return f'{self.__class__.__name__}({self._config_fields})'

    def __len__(self):
        return len(self._config_fields)

    def __iter__(self):
        return iter(self._config_fields)

    def __getitem__(self, key: str):
        try:
            return self._find_item(key)
        except:
            raise KeyError(key)

    def __setitem__(self, key: str, value: Any):
        if isinstance(value, Mapping):
            value = Config(value)
        keys = key.split(SEPARATOR, maxsplit=1)
        if len(keys) == 1:
            value = self.enforce_annotation(keys[0], value)
            self._config_fields[keys[0]] = value
        elif keys[0] in self:
            self._config_fields[keys[0]][keys[1]] = value
        else:
            self._config_fields[keys[0]] = Config({keys[1]: value})

    def __delitem__(self, key: str):
        try:
            keys = key.split(SEPARATOR)
            item = self._find_item(keys[:-1])
            del item[int(keys[-1]) if isinstance(item, Sequence) else keys[-1]]
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
        new_config = Config(copy.deepcopy(other))
        new_config.update(self)
        return new_config

    def __ior__(self, other: Mapping):
        self.update(other)
        return self

    @classmethod
    def annotation(cls, key: str) -> Optional[Any]:
        for c in cls.mro():
            if issubclass(c, Config) and hasattr(c, '__annotations__'):
                if key in c.__annotations__:
                    return c.__annotations__[key]

    @classmethod
    def enforce_annotation(cls, key: str, value: Any) -> Any:
        if annotation := cls.annotation(key):
            if isinstance(annotation, type) and not isinstance(value, annotation):
                return annotation(value)
        return value

    @classmethod
    def defaults(cls) -> dict[str, Any]:
        return {
            key: value
            for c in reversed(cls.mro())
            if issubclass(c, Config)
            for key, value in vars(c).items()
            if not key.startswith('_')
            if not callable(value) and not isinstance(value, (classmethod, staticmethod))
        }

    @classmethod
    def from_dotenv(cls, *paths: PathLike, keepcase: bool = False):
        # read values from dotenv files
        if paths:
            values = {}
            for path in paths:
                values.update(dotenv_values(path))
        else:
            values = dotenv_values()
        # convert dotenv values to config fields
        fields = {}
        for k, v in values.items():
            if not keepcase:
                k = k.lower()
            fields[k] = cls.enforce_annotation(k, v)
        # return config
        return cls(fields)

    @classmethod
    def from_json(cls, s: str):
        """Load a Config object from a JSON string."""
        return json.loads(s, cls=ConfigJSONDecoder, config_cls=cls)

    def to_json(self, *, ensure_ascii: bool = False, indent: Optional[Union[int, str]] = None):
        """Convert the Config object to a JSON string."""
        return json.dumps(self, ensure_ascii=ensure_ascii, indent=indent, cls=ConfigJSONEncoder)

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            if key in self and isinstance(self[key], Config) and isinstance(value, Mapping):
                self[key].update(value)
            else:
                self[key] = value

    def field_names(self) -> Iterator[str]:
        for key, value in self.items():
            if isinstance(value, Config):
                for name in value.field_names():
                    yield f'{key}{SEPARATOR}{name}'
            else:
                yield key

    def fields(self) -> Iterator[tuple[str, Any]]:
        for key, value in self.items():
            if isinstance(value, Config):
                for name, val in value.fields():
                    yield f'{key}{SEPARATOR}{name}', val
            else:
                yield key, value

    def _find_item(self, keys: Union[str, list[str]]) -> Any:
        if not keys:
            return self
        if isinstance(keys, str):
            keys = keys.split(SEPARATOR)
        item = self._config_fields[keys[0]]
        for k in keys[1:]:
            if isinstance(item, Sequence):
                item = item[int(k)]
            else:
                item = item[k]
        return item


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
