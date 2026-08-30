"""Live order-book capture.

Live sources register into the one unified registry with every other source
(:mod:`ob_analytics.sources`) — look them up there with ``get_source`` /
``list_sources``, and drive one with :func:`ob_analytics.live._runner.run_capturer`.

Public API:
    CaptureConfig, CaptureResult, CaptureSink, EventDict
    LiveSource, SupportsDiagnostics

Importing this package registers the built-in ccxt live source.  The bitstamp
live capability rides on :class:`ob_analytics.bitstamp.BitstampSource`, so it is
registered when that module is imported (at ``import ob_analytics``).
"""

from __future__ import annotations

# Register the ccxt live source.  Importing the module is cheap — ccxt itself is
# imported lazily only when a capture starts — so it registers unconditionally;
# a capture without the ``[ccxt]`` extra raises a clear install hint at that
# point.
from ob_analytics.live import ccxt_source  # noqa: F401 - fires register_source
from ob_analytics.live._base import (
    CaptureConfig,
    CaptureResult,
    CaptureSink,
    EventDict,
    LiveSource,
    SupportsDiagnostics,
)

__all__ = [
    "CaptureConfig",
    "CaptureResult",
    "CaptureSink",
    "EventDict",
    "LiveSource",
    "SupportsDiagnostics",
]
