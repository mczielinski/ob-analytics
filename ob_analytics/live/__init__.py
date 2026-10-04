"""Live order-book capture.

Live sources register into the one unified registry with every other source
(:mod:`ob_analytics.sources`) — look them up there with ``get_source`` /
``list_sources``.  :func:`run_capture` drives one for as long as asked, into a
directory of segments recorded in ``manifest.json`` (read it back with
:func:`read_manifest`); :func:`ob_analytics.live._runner.run_capturer` is one
segment.

Public API:
    CaptureConfig, CaptureResult, CaptureSink, EventDict
    LiveSource, SupportsDiagnostics, SupportsPreflight
    run_capture, CaptureRun
    read_manifest, CaptureManifest, Segment, Gap, EndReason

Importing this package registers the built-in ccxt and cryptofeed live sources.  The bitstamp
live capability rides on :class:`ob_analytics.bitstamp.BitstampSource`, so it is
registered when that module is imported (at ``import ob_analytics``).
"""

from __future__ import annotations

# Register the ccxt and cryptofeed live sources.  Importing either module is
# cheap — the venue library itself is imported lazily only when a capture
# starts — so they register unconditionally; a capture without the matching
# extra raises a clear install hint at that point.
from ob_analytics.live import (
    ccxt_source,  # noqa: F401 - fires register_source
    cryptofeed_source,  # noqa: F401 - fires register_source
)
from ob_analytics.live._base import (
    CaptureConfig,
    CaptureResult,
    CaptureSink,
    EventDict,
    LiveSource,
    SupportsDiagnostics,
    SupportsPreflight,
)
from ob_analytics.live._manifest import (
    CaptureManifest,
    EndReason,
    Gap,
    Segment,
    read_manifest,
)
from ob_analytics.live._supervisor import CaptureRun, run_capture

__all__ = [
    "CaptureConfig",
    "CaptureManifest",
    "CaptureResult",
    "CaptureRun",
    "CaptureSink",
    "EndReason",
    "EventDict",
    "Gap",
    "LiveSource",
    "Segment",
    "SupportsDiagnostics",
    "SupportsPreflight",
    "read_manifest",
    "run_capture",
]
