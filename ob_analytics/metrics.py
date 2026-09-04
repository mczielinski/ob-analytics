"""The metric registry and its plug-in discovery.

A metric measures a finished run and draws as a level-less plot.  Every metric
registers itself here under a name, the way sources register with ``SOURCES``
and plot backends with ``RENDERERS``.

Third-party metrics ship in their own package and are found through the
``ob_analytics.metrics`` entry-point group — :func:`load_metric_plugins`
discovers and registers them with no edit to this core.

A registered value is a :class:`~ob_analytics.protocols.Metric` *instance*, not
a class: a metric carries no per-run construction, so the object registered is
the object called.
"""

from __future__ import annotations

from importlib.metadata import entry_points

from loguru import logger

from ob_analytics._registry import Registry
from ob_analytics.protocols import Metric

#: The entry-point group a third-party package advertises a metric under, e.g.
#: ``[project.entry-points."ob_analytics.metrics"]`` with ``amihud =
#: my_pkg.amihud:AmihudMetric``.
ENTRY_POINT_GROUP = "ob_analytics.metrics"

#: Registry of metric name → :class:`~ob_analytics.protocols.Metric` instance.
METRICS: Registry[str, Metric] = Registry("metric")


def register_metric(metric: Metric) -> None:
    """Register *metric* under its own :attr:`~ob_analytics.protocols.Metric.name`.

    Case-insensitive; overwriting an existing registration is allowed (handy
    for tests and for a plug-in that intentionally shadows a built-in).
    """
    METRICS.register(metric.name, metric)


def list_metrics() -> list[str]:
    """Return a sorted list of registered metric names."""
    return METRICS.list()


def get_metric(name: str) -> Metric:
    """Return the metric registered under *name* (case-insensitive).

    Raises
    ------
    KeyError
        If no metric is registered under *name*; the message lists the
        registered names.
    """
    return METRICS.get(name)


_plugins_loaded = False


def load_metric_plugins(*, force: bool = False) -> list[str]:
    """Discover and register metrics advertised via entry points.

    Scans the :data:`ENTRY_POINT_GROUP` entry-point group; each entry's value
    is loaded to a :class:`~ob_analytics.protocols.Metric` class, instantiated
    with no arguments, and registered under its own ``name``.  This is what
    lets a metric live in a separate installable package without editing
    ob-analytics.

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
            metric = ep.load()()
        except Exception as exc:  # noqa: BLE001 - one bad plug-in must not break the rest
            logger.warning(
                "Metric plug-in {!r} ({}) failed to load: {!r}",
                ep.name,
                getattr(ep, "value", "?"),
                exc,
            )
            continue
        register_metric(metric)
        registered.append(metric.name)
        logger.debug("Registered metric plug-in {!r} from entry point", metric.name)

    _plugins_loaded = True
    return registered
