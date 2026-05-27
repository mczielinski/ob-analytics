"""Live order-book capture.

Public API:
    register_capturer(name, capturer_cls)
    list_capturers() -> list[str]
    get_capturer(name) -> type[LiveCapturer]
    LiveCapturer, CaptureConfig, CaptureResult
"""

from __future__ import annotations

from ob_analytics.live._base import (
    CaptureConfig,
    CaptureResult,
    CaptureSink,
    EventDict,
    LiveCapturer,
)

_CAPTURERS: dict[str, type[LiveCapturer]] = {}


def register_capturer(name: str, capturer_cls: type[LiveCapturer]) -> None:
    """Register a :class:`LiveCapturer` under *name* (case-insensitive).

    Idempotent: overwriting an existing registration is allowed (useful for
    monkey-patching in tests).
    """
    _CAPTURERS[name.lower()] = capturer_cls


def list_capturers() -> list[str]:
    """Return the sorted list of registered capturer names."""
    return sorted(_CAPTURERS)


def get_capturer(name: str) -> type[LiveCapturer]:
    """Return the capturer class registered under *name* (case-insensitive)."""
    key = name.lower()
    if key not in _CAPTURERS:
        available = ", ".join(list_capturers()) or "(none)"
        raise ValueError(f"Unknown capturer {name!r}. Registered: {available}")
    return _CAPTURERS[key]


# -- Register built-ins -----------------------------------------------------
# Done at import time. Import is local so the absence of ``websockets`` (an
# optional ``[live]`` extra) does not break ``import ob_analytics.live``.
try:
    from ob_analytics.live.bitstamp import BitstampCapturer

    register_capturer("bitstamp", BitstampCapturer)
except ImportError:
    # websockets not installed; the registry stays empty until the user
    # registers their own capturer or installs ob-analytics[live].
    pass


__all__ = [
    "CaptureConfig",
    "CaptureResult",
    "CaptureSink",
    "EventDict",
    "LiveCapturer",
    "get_capturer",
    "list_capturers",
    "register_capturer",
]
