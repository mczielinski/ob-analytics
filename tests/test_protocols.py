"""Tests for the source protocol contracts in ob_analytics.protocols."""

from __future__ import annotations

from ob_analytics.protocols import OfflineSource, Source


def test_sources_are_structural_without_inheritance():
    from ob_analytics.bitstamp import BitstampSource
    from ob_analytics.live import LiveSource
    from ob_analytics.lobster import LobsterSource

    # The concrete sources do NOT inherit from the protocols ...
    assert Source not in BitstampSource.__mro__
    assert OfflineSource not in BitstampSource.__mro__
    assert OfflineSource not in LobsterSource.__mro__

    # ... yet they satisfy the runtime-checkable protocols structurally.
    assert isinstance(BitstampSource(), Source)
    assert isinstance(BitstampSource(), OfflineSource)
    # Bitstamp is one source with both capabilities (offline replay + live).
    assert isinstance(BitstampSource(), LiveSource)

    # LOBSTER ships as files: offline-capable, not live.
    assert isinstance(LobsterSource(), OfflineSource)
    assert not isinstance(LobsterSource(), LiveSource)
