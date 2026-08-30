"""A tiny name -> class registry.

Used only for list-valued config slots (features, conditioners, losses), where a
full Hydra ``_target_`` block per entry would be noise. Singular choices --
model, process, data -- stay plain Hydra config groups with no registry.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from typing import Generic, TypeVar

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

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)
