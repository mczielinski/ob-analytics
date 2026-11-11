"""Tests for the source protocol contracts in ob_analytics.protocols."""

from __future__ import annotations

import pytest

from ob_analytics.protocols import (
    Clocks,
    FeedDeclarations,
    FeedType,
    Level,
    OfflineSource,
    SequenceKind,
    Source,
    SupportsComputeDepth,
    SupportsRequiredContext,
    SupportsWriter,
    TradeAttribution,
    declarations_of,
    required_context_of,
)


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


class _TwoMethodSource:
    """An offline source with only the two required factories."""

    name = "two_methods"
    level = Level.L3
    feed_type = FeedType.DIFF_FEED
    settings = None

    def create_loader(self, config, ctx):
        raise NotImplementedError

    def create_trade_reader(self, config, ctx, loader):
        raise NotImplementedError


def test_an_offline_source_needs_only_the_two_factories():
    source = _TwoMethodSource()
    assert isinstance(source, OfflineSource)
    assert not isinstance(source, SupportsWriter)
    assert not isinstance(source, SupportsComputeDepth)
    assert not isinstance(source, SupportsRequiredContext)
    assert required_context_of(source) == []


def test_built_in_sources_declare_only_the_capabilities_they_use():
    from ob_analytics.bitstamp import BitstampSource
    from ob_analytics.depth_csv import DepthCsvSource
    from ob_analytics.lobster import LobsterSource

    assert isinstance(BitstampSource(), SupportsWriter)
    assert not isinstance(BitstampSource(), SupportsComputeDepth)
    assert not isinstance(BitstampSource(), SupportsRequiredContext)
    assert not isinstance(DepthCsvSource(), SupportsComputeDepth)
    # LOBSTER reads its orderbook file for depth and needs the trading date.
    assert isinstance(LobsterSource(), SupportsComputeDepth)
    assert required_context_of(LobsterSource()) == ["trading_date"]


class TestFeedDeclarations:
    def test_an_undeclared_source_gets_the_defaults(self):
        assert declarations_of(None) == FeedDeclarations()
        assert declarations_of(object()) == FeedDeclarations()
        assert FeedDeclarations().feed_type is FeedType.UNKNOWN

    def test_a_source_declares_its_feed(self):
        from ob_analytics.depth_csv import DepthCsvSource

        declared = declarations_of(DepthCsvSource())
        assert declared.feed_type is FeedType.PRICE_LEVELS
        assert declared.trade_attribution is TradeAttribution.NONE

    def test_round_trips_through_a_dict(self):
        declared = FeedDeclarations(
            feed_type=FeedType.MATCHED_BOOK,
            trade_attribution=TradeAttribution.MAKER_ONLY,
            sequence_kind=SequenceKind.MONOTONIC,
            clocks=Clocks.RECEIVE_ONLY,
            sequence_restarts=3,
        )
        assert FeedDeclarations.from_dict(declared.to_dict()) == declared
        assert FeedDeclarations.from_dict({}) == FeedDeclarations()

    def test_an_unknown_value_is_an_error(self):
        with pytest.raises(ValueError):
            FeedDeclarations.from_dict({"feed_type": "teleport"})
