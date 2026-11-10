"""The unified source registry and plug-in discovery.

Every data source — file or live — registers itself here under a name, the
way plot backends register with ``RENDERERS`` and writers with ``WRITERS``.
One registry keyed by name replaces the previous split between the offline
``FORMATS`` registry and the live ``CAPTURERS`` registry.

Built-in sources self-register at import time (see the bottom of
:mod:`ob_analytics.bitstamp`, :mod:`ob_analytics.lobster`,
:mod:`ob_analytics.depth_l2`, and :mod:`ob_analytics.live.ccxt_source`).
Third-party sources ship in their own package and are found through the
``ob_analytics.sources`` entry-point group — :func:`load_source_plugins`
discovers and registers them with no edit to this core.

A registered value is a :class:`~ob_analytics.protocols.Source` class; the
caller constructs it (``get_source("ccxt")(settings=CcxtSettings(...))``) and
uses the capability it needs — :class:`~ob_analytics.protocols.OfflineSource`
for file replay, :class:`~ob_analytics.live._base.LiveSource` for live capture.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import EntryPoint
from typing import Any

from ob_analytics._registry import Registry, load_plugins
from ob_analytics.protocols import Source


@dataclass(frozen=True)
class CliOption:
    """How a :class:`~ob_analytics.config.SourceSettings` field shows on the command line.

    ``ob-analytics capture <source>`` has one flag for each field of the
    source's settings class, so a plug-in source can be set up from the
    command line as from Python.  Put this marker on a field with
    ``typing.Annotated`` to give the flag its help text, another name, or a
    way to read its value::

        class VenueSettings(SourceSettings):
            venue: Annotated[str, CliOption(help="The venue id")] = ""
            depth: Annotated[int, CliOption(help="Levels a side", flag="levels")] = 10

    A field without the marker still gets a flag named after it, when its
    type is one the command line can read: ``str``, ``int``, ``float``,
    ``bool``, a ``Literal``, an ``Enum`` or ``Path``, or ``None`` or one of
    those.  A field of any other type gets no flag unless the marker gives a
    *type*.  A :class:`~ob_analytics.config.Credential` field never gets one.

    Attributes
    ----------
    help : str
        The flag's help text.
    flag : str, optional
        The flag's name without the leading ``--``.  ``None`` (the default)
        names it after the field, with ``-`` for ``_``.
    type : callable, optional
        Turns the text given on the command line into the field's value.
        ``None`` (the default) reads it from the field's type.
    """

    help: str
    flag: str | None = None
    type: Callable[[str], Any] | None = None


#: The entry-point group a third-party package advertises a source under, e.g.
#: ``[project.entry-points."ob_analytics.sources"]`` with ``coinbase =
#: my_pkg.coinbase:CoinbaseSource``.
ENTRY_POINT_GROUP = "ob_analytics.sources"

#: Registry of source name → :class:`~ob_analytics.protocols.Source` class.
SOURCES: Registry[str, type[Source]] = Registry("source")


def register_source(name: str, source_cls: type[Source]) -> None:
    """Register a :class:`~ob_analytics.protocols.Source` class under *name*.

    Case-insensitive; overwriting an existing registration is allowed (handy
    for tests and for a plug-in that intentionally shadows a built-in).
    """
    SOURCES.register(name, source_cls)


def list_sources() -> list[str]:
    """Return a sorted list of registered source names."""
    return SOURCES.list()


def get_source(name: str) -> type[Source]:
    """Return the source class registered under *name* (case-insensitive).

    Raises
    ------
    UnknownNameError
        If no source is registered under *name*; the message names
        the registered names closest to it.
    """
    return SOURCES.get(name)


def _register_entry_point(ep: EntryPoint) -> str:
    register_source(ep.name, ep.load())
    return ep.name


def load_source_plugins(*, force: bool = False) -> list[str]:
    """Discover and register sources advertised via entry points.

    Scans the :data:`ENTRY_POINT_GROUP` entry-point group; each entry's value
    is loaded to a :class:`~ob_analytics.protocols.Source` class and registered
    under the entry-point name.  This is what lets a source live in a separate
    installable package without editing ob-analytics.

    The scan runs once per process unless *force* is set.  A plug-in that
    fails to load is skipped with a warning, so one broken package cannot stop
    the rest from loading.

    Returns
    -------
    list of str
        The names newly registered by this call.
    """
    return load_plugins(ENTRY_POINT_GROUP, _register_entry_point, force=force)
