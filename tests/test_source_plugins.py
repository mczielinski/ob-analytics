"""Entry-point plug-in discovery for data sources (issue #137).

The headline acceptance: a source that lives outside the core package is
discovered and registered through the ``ob_analytics.sources`` entry-point
group, with **no edit to ob-analytics**, and then drives the pipeline like any
built-in.  The "third-party package" is stood up here by monkeypatching
``importlib.metadata.entry_points`` so the test stays offline and needs no real
install.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ob_analytics import sources
from ob_analytics.config import PipelineConfig, SourceSettings
from ob_analytics.pipeline import Pipeline
from ob_analytics.protocols import FeedType, Level, OfflineSource, RunContext, Source


class FakeSource:
    """A stand-in third-party L2 source, defined entirely outside the core.

    Reuses the shared L2 loader / trade reader (issue #98) — exactly how a real
    plug-in would — so a full ``Pipeline`` run proves the source is usable, not
    just registered.
    """

    name = "faketest"
    level = Level.L2
    feed_type = FeedType.MATCHED_BOOK
    settings = SourceSettings()

    def create_loader(self, config: PipelineConfig, ctx: RunContext):
        from ob_analytics.depth_l2 import L2DepthLoader

        return L2DepthLoader(config)

    def create_trade_source(self, config: PipelineConfig, ctx: RunContext):
        from ob_analytics.depth_l2 import L2TradeReader

        return L2TradeReader(config)

    def create_writer(self, config: PipelineConfig, ctx: RunContext):
        return None

    def compute_depth(self, events, config, source, ctx):
        return None

    def config_defaults(self) -> dict:
        return {}

    def required_context(self) -> list[str]:
        return []


class _FakeEntryPoint:
    """Minimal stand-in for ``importlib.metadata.EntryPoint``."""

    name = "faketest"
    value = "tests.test_source_plugins:FakeSource"

    def load(self):
        return FakeSource


def _fake_entry_points(*, group: str | None = None):
    if group == sources.ENTRY_POINT_GROUP:
        return [_FakeEntryPoint()]
    return []


@pytest.fixture
def _restore_registry():
    """Snapshot / restore the global source registry around the test."""
    before = dict(sources.SOURCES._items)
    yield
    sources.SOURCES._items.clear()
    sources.SOURCES._items.update(before)


@pytest.fixture
def discovered_plugin(monkeypatch, _restore_registry):
    """Discover the fake plug-in through the (monkeypatched) entry points."""
    monkeypatch.setattr(sources, "entry_points", _fake_entry_points)
    registered = sources.load_source_plugins(force=True)
    assert "faketest" in registered
    return registered


class TestEntryPointDiscovery:
    def test_registers_without_editing_core(self, discovered_plugin):
        # The plug-in is now in the one shared registry, found by name.
        assert "faketest" in sources.list_sources()
        assert sources.get_source("faketest") is FakeSource

    def test_registered_source_satisfies_the_protocols(self, discovered_plugin):
        src = sources.get_source("faketest")()
        assert isinstance(src, Source)
        assert isinstance(src, OfflineSource)
        assert src.level is Level.L2
        assert src.feed_type is FeedType.MATCHED_BOOK

    def test_plugin_source_drives_the_pipeline(self, discovered_plugin, tmp_path):
        # Prove usability end-to-end: a Pipeline built from the plug-in name
        # runs the L2 path on a tiny depth CSV.
        pd.DataFrame(
            {
                "timestamp": [1_700_000_000_000, 1_700_000_000_001],
                "side": ["bid", "ask"],
                "price": [100.0, 101.0],
                "volume": [5.0, 3.0],
            }
        ).to_csv(tmp_path / "depth.csv", index=False)

        result = Pipeline.from_source("faketest").run(tmp_path)
        assert result.level is Level.L2
        assert len(result.depth) == 2

    def test_idempotent_scan_is_a_noop_without_force(self, monkeypatch):
        # After a scan, a second call without force re-registers nothing.
        monkeypatch.setattr(sources, "_plugins_loaded", True)
        monkeypatch.setattr(sources, "entry_points", _fake_entry_points)
        assert sources.load_source_plugins() == []

    def test_broken_plugin_is_skipped_not_fatal(self, monkeypatch, _restore_registry):
        class _BrokenEP:
            name = "broken"
            value = "nope:Nope"

            def load(self):
                raise ImportError("no such module")

        def _eps(*, group=None):
            return (
                [_BrokenEP(), _FakeEntryPoint()]
                if group == sources.ENTRY_POINT_GROUP
                else []
            )

        monkeypatch.setattr(sources, "entry_points", _eps)
        registered = sources.load_source_plugins(force=True)
        # The good one still loads; the broken one is logged and skipped.
        assert "faketest" in registered
        assert "broken" not in registered
