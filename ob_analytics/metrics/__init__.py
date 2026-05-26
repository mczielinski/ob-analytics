"""Pluggable flow-toxicity and liquidity metrics.

Public API:
    ToxicityMetric             — protocol
    register_metric(name, cls) — add a custom metric
    list_metrics()             — names of registered built-ins
    Vpin, KyleLambda, Ofi      — built-in implementations
"""

from __future__ import annotations

from ob_analytics.metrics._base import ToxicityMetric
from ob_analytics.metrics.kyle_lambda import KyleLambda
from ob_analytics.metrics.ofi import Ofi
from ob_analytics.metrics.vpin import Vpin

_METRICS: dict[str, type[ToxicityMetric]] = {}


def register_metric(name: str, metric_cls: type[ToxicityMetric]) -> None:
    """Register a :class:`ToxicityMetric` implementation under *name*."""
    _METRICS[name.lower()] = metric_cls


def list_metrics() -> list[str]:
    """Return a sorted list of registered metric names."""
    return sorted(_METRICS)


# Built-ins
register_metric("vpin", Vpin)
register_metric("kyle_lambda", KyleLambda)
register_metric("ofi", Ofi)


__all__ = [
    "KyleLambda",
    "Ofi",
    "ToxicityMetric",
    "Vpin",
    "list_metrics",
    "register_metric",
]
