"""Custom exceptions for the ob-analytics package.

Every error the package raises for a mistake in what it was given is an
:class:`ObAnalyticsError`, so callers can catch them all with one ``except``
clause.  Each kind of mistake has one type:

* :class:`ConfigError` -- a value is invalid or inconsistent: a bad setting
  or argument, or input data that is empty or missing required columns.
  It is also a :class:`ValueError`.
* :class:`UnknownNameError` -- a name (a source, format, metric, plot,
  style, ...) is not one the package knows.  It is also a
  :class:`KeyError`, and its message names the known names closest to the
  one given.

A path that does not exist raises :class:`FileNotFoundError`, a missing
optional dependency raises :class:`ImportError`, and an argument of the
wrong type raises :class:`TypeError`, as in the standard library.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable

__all__ = [
    "ConfigError",
    "ObAnalyticsError",
    "UnknownNameError",
]

#: Show every known name when there are at most this many and none is close.
_SHORT_LIST = 12


class ObAnalyticsError(Exception):
    """Base exception for all ob-analytics errors."""


class ConfigError(ObAnalyticsError, ValueError):
    """Configuration values or input data are invalid or inconsistent."""


class UnknownNameError(ObAnalyticsError, KeyError):
    """A name is not one the package knows.

    The message names the known names closest to *name*.  When none is
    close, it lists the known names if there are only a few.

    Parameters
    ----------
    kind : str
        What the name names, for the message, e.g. ``"source"``.
    name : object
        The name that was given.
    known : iterable
        The names that would have been accepted.
    hint : str, optional
        A sentence to end the message with, e.g. which backends draw a plot.

    Attributes
    ----------
    kind, name : as given
    known : list of str
        The accepted names, as text.
    """

    def __init__(
        self,
        kind: str,
        name: object,
        known: Iterable[object],
        hint: str | None = None,
    ) -> None:
        self.kind = kind
        self.name = name
        self.known = [str(k) for k in known]
        self.hint = hint
        super().__init__(self._message())

    def _message(self) -> str:
        text = f"Unknown {self.kind} {self.name!r}."
        close = difflib.get_close_matches(
            str(self.name).lower(),
            [k.lower() for k in self.known],
            n=3,
            cutoff=0.6,
        )
        by_lower = {k.lower(): k for k in self.known}
        if close:
            text += " Did you mean " + _or_list([by_lower[c] for c in close]) + "?"
        elif not self.known and self.hint is None:
            text += f" No {self.kind} is registered."
        elif self.known and len(self.known) <= _SHORT_LIST:
            text += " Known: " + ", ".join(repr(k) for k in sorted(self.known)) + "."
        if self.hint:
            text += " " + self.hint
        return text

    def __str__(self) -> str:
        # KeyError's own __str__ quotes the message like a dict key.
        return str(self.args[0])

    def __reduce__(self) -> tuple[object, ...]:
        return (type(self), (self.kind, self.name, self.known, self.hint))


def _or_list(names: list[str]) -> str:
    quoted = [repr(n) for n in names]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + " or " + quoted[-1]
