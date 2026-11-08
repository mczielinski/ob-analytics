"""Generic registry — the single plumbing primitive for pluggable surfaces.

One :class:`Registry` instance per pluggable surface (sources, metrics, bar
rules, features, writers, renderers), so every surface shares the same lookup
rules and error messages.  :func:`load_plugins` is the one way any of them
discovers plug-ins that other packages advertise through entry points.
"""

from __future__ import annotations

import builtins
import warnings
from collections.abc import Callable
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Generic, TypeVar

from loguru import logger

from ob_analytics.exceptions import UnknownNameError

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
        """Return the value for *key*.

        Raises
        ------
        UnknownNameError
            If nothing is registered under *key*.  The message names the
            registered keys closest to it.
        """
        nk = self._normalize(key)
        if nk not in self._items:
            raise UnknownNameError(self._kind, key, self.list())
        return self._items[nk]

    # ``builtins.list`` because the method name ``list`` shadows the builtin
    # within the class namespace where this return annotation is resolved.
    def list(self) -> builtins.list[K]:
        """Return registered keys, sorted when orderable."""
        try:
            return sorted(self._items)  # ty: ignore[invalid-argument-type] -- unorderable keys fall through to the except
        except TypeError:
            return list(self._items)

    def __contains__(self, key: object) -> bool:
        try:
            return self._normalize(key) in self._items
        except Exception:  # noqa: BLE001 - membership must never raise; unknown key is simply absent
            return False

    @staticmethod
    def _normalize(key: Any) -> Any:
        return key.lower() if isinstance(key, str) else key


#: The entry-point groups already scanned in this process.
_loaded_groups: set[str] = set()


def load_plugins(
    group: str, register: Callable[[EntryPoint], str], *, force: bool = False
) -> list[str]:
    """Register every plug-in advertised under the entry-point *group*.

    *register* is called once per entry point.  It loads what the entry point
    names, registers it, and returns the name it registered.  Each registry
    supplies its own *register*, because what an entry point names differs: a
    class, an instance to construct, or a module to import later.

    The scan runs once per group per process unless *force* is set.  A
    plug-in that fails to load or register is skipped with a warning, so one
    broken package cannot stop the rest from loading.

    Returns
    -------
    list of str
        The names this call registered.
    """
    if group in _loaded_groups and not force:
        return []

    registered: list[str] = []
    for ep in entry_points(group=group):
        try:
            name = register(ep)
        except Exception as exc:  # noqa: BLE001 - one bad plug-in must not break the rest
            warnings.warn(
                f"Skipped the plug-in {ep.name!r} ({ep.value}) in the entry-point "
                f"group {group!r}: it failed to load: {exc!r}",
                stacklevel=2,
            )
            continue
        registered.append(name)
        logger.debug("Registered plug-in {!r} from entry-point group {!r}", name, group)

    _loaded_groups.add(group)
    return registered
