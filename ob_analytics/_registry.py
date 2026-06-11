"""Generic registry — the single plumbing primitive for pluggable surfaces.

One :class:`Registry` instance per pluggable surface (formats, writers,
capturers, renderers), so every surface shares the same lookup rules and
error messages.
"""

from __future__ import annotations

import builtins
from typing import Any, Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class Registry(Generic[K, V]):
    """A named mapping of keys to values with a uniform lookup error.

    String keys are matched case-insensitively; non-string keys (e.g.
    ``(plot, backend)`` tuples) are matched as-is.

    Parameters
    ----------
    kind : str
        Human-readable noun used in error messages, e.g. ``"format"``.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[K, V] = {}

    def register(self, key: K, value: V) -> None:
        """Register *value* under *key*. Overwrites silently (handy in tests)."""
        self._items[self._normalize(key)] = value

    def get(self, key: K) -> V:
        """Return the value for *key*, or raise ``KeyError`` listing registrations."""
        nk = self._normalize(key)
        if nk not in self._items:
            raise KeyError(f"Unknown {self._kind} {key!r}. Registered: {self.list()}")
        return self._items[nk]

    # ``builtins.list`` because the method name ``list`` shadows the builtin
    # within the class namespace where this return annotation is resolved.
    def list(self) -> builtins.list[K]:
        """Return registered keys, sorted when orderable."""
        try:
            return sorted(self._items)
        except TypeError:
            return list(self._items)

    def __contains__(self, key: object) -> bool:
        try:
            return self._normalize(key) in self._items
        except Exception:
            return False

    @staticmethod
    def _normalize(key: Any) -> Any:
        return key.lower() if isinstance(key, str) else key
