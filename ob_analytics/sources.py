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

from importlib.metadata import entry_points

from loguru import logger

from ob_analytics._registry import Registry
from ob_analytics.protocols import Source

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
    KeyError
        If no source is registered under *name*; the message lists the
        registered names.
    """
    return SOURCES.get(name)


_plugins_loaded = False


def load_source_plugins(*, force: bool = False) -> list[str]:
    """Discover and register sources advertised via entry points.

    Scans the :data:`ENTRY_POINT_GROUP` entry-point group; each entry's value
    is loaded to a :class:`~ob_analytics.protocols.Source` class and registered
    under the entry-point name.  This is what lets a source live in a separate
    installable package without editing ob-analytics.

    Idempotent: the scan runs once per process unless *force* is set (the test
    suite forces a re-scan after monkeypatching the entry points).  A plug-in
    that fails to import is logged and skipped, so one broken package cannot
    stop the rest from loading.

    Returns
    -------
    list of str
        The names newly registered by this call.
    """
    global _plugins_loaded
    if _plugins_loaded and not force:
        return []

    registered: list[str] = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            source_cls = ep.load()
        except Exception as exc:  # noqa: BLE001 - one bad plug-in must not break the rest
            logger.warning(
                "Source plug-in {!r} ({}) failed to load: {!r}",
                ep.name,
                getattr(ep, "value", "?"),
                exc,
            )
            continue
        register_source(ep.name, source_cls)
        registered.append(ep.name)
        logger.debug("Registered source plug-in {!r} from entry point", ep.name)

    _plugins_loaded = True
    return registered
