"""Name -> class registries, and building an object from a config entry.

Every config slot that makes an object names a registered class by ``type:``
and passes the rest of the entry as its arguments, the way mmengine's
``build_from_cfg`` does::

    prepare:
      - {type: put_on_ground, mode: mesh}

Each slot has its own registry (`PREPARE`, `REDUCE`, `WINDOWS`, `MODELS`, ...),
so short names only have to be unique within one. A user's own class joins by
registering itself and being imported first, which a config asks for with
``custom_imports: [my_package.my_module]`` (`import_custom`).
"""

from __future__ import annotations

import difflib
import importlib
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def decorate(cls: type[T]) -> type[T]:
            if name in self._items:
                raise ValueError(f"{self.kind} `{name}` is already registered")
            self._items[name] = cls
            return cls

        return decorate

    def get(self, name: str) -> type[T]:
        try:
            return self._items[name]
        except KeyError:
            close = difflib.get_close_matches(name, self.names(), n=1)
            hint = f", did you mean `{close[0]}`?" if close else ""
            raise KeyError(
                f"unknown {self.kind} `{name}`{hint}. Available: {', '.join(self.names())}"
            ) from None

    def name_of(self, cls: type) -> str:
        """The name ``cls`` is registered under."""
        for name, item in self._items.items():
            if item is cls:
                return name
        raise KeyError(f"{cls.__name__} is not a registered {self.kind}")

    def build(self, entry: Mapping[str, Any] | str, **extra: Any) -> T:
        """The object a config entry describes: ``type`` picks the class, every
        other key is passed to it, then ``extra`` (what the caller supplies, such
        as a model's input width). A bare string is ``{type: <string>}``."""
        if isinstance(entry, str):
            entry = {"type": entry}
        entry = _plain(entry)
        if "type" not in entry:
            raise KeyError(f"a {self.kind} entry needs a `type:` naming one of: {', '.join(self.names())}")
        kwargs = {k: v for k, v in entry.items() if k != "type"}
        return self.get(str(entry["type"]))(**kwargs, **extra)

    def build_all(self, entries: Iterable[Mapping[str, Any] | str] | None) -> list[T]:
        return [self.build(entry) for entry in entries or ()]

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)


def _plain(value: Any) -> Any:
    """A config node as plain dicts and lists, so constructors get Python values."""
    try:
        from omegaconf import OmegaConf
    except ImportError:  # pragma: no cover - omegaconf ships with the project
        OmegaConf = None
    if OmegaConf is not None and OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return dict(value)


def import_custom(modules: Iterable[str] | None) -> None:
    """Import a config's ``custom_imports``, so the classes they register are
    known before anything is built. A module that fails to import is an error:
    a silently missing registration would surface later as an unknown name."""
    for module in modules or ():
        importlib.import_module(str(module))
