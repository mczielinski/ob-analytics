"""Tests for the protocol contracts in ob_analytics.protocols."""

from __future__ import annotations

from ob_analytics.protocols import Format


def test_formats_are_structural_without_inheritance():
    from ob_analytics.bitstamp import BitstampFormat
    from ob_analytics.lobster import LobsterFormat

    # The concrete formats do NOT inherit from Format ...
    assert Format not in BitstampFormat.__mro__
    assert Format not in LobsterFormat.__mro__

    # ... yet both satisfy the runtime-checkable Protocol structurally.
    assert isinstance(BitstampFormat(), Format)
    assert isinstance(LobsterFormat(), Format)
