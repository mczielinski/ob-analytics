"""The result keeps the tables built from it (issue #498).

``PipelineResult`` builds its per-order tables, bars and features once and
keeps them, and the gallery keeps its display-unit tables and scene on the
result, so a second plot does not rebuild them.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ob_analytics import engine
from ob_analytics.data import load_result, save_data
from ob_analytics.exceptions import ConfigError
from ob_analytics.pipeline import Pipeline, PipelineResult
from ob_analytics.visualization import _concepts, gallery


class _Counter:
    """Wrap *function*, counting its calls."""

    def __init__(self, function: Callable[..., Any]) -> None:
        self.function = function
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self.function(*args, **kwargs)


def _count(monkeypatch: pytest.MonkeyPatch, owner: Any, name: str) -> _Counter:
    counter = _Counter(getattr(owner, name))
    monkeypatch.setattr(owner, name, counter)
    return counter


@pytest.fixture
def fresh(tiny_bitstamp_orders_csv: Path) -> PipelineResult:
    """A run with nothing kept yet."""
    return Pipeline().run(tiny_bitstamp_orders_csv)


class TestAccessorsBuildOnce:
    def test_lifecycles(self, fresh, monkeypatch) -> None:
        counter = _count(monkeypatch, engine, "order_lifecycles")
        first = fresh.lifecycles()
        assert fresh.lifecycles() is first
        assert counter.calls == 1
        assert len(first) > 0

    def test_queue_per_levels(self, fresh, monkeypatch) -> None:
        counter = _count(monkeypatch, engine, "queue_positions")
        touch = fresh.queue()
        assert fresh.queue(levels="touch") is touch
        every = fresh.queue(levels="all")
        assert fresh.queue(levels="all") is every
        assert counter.calls == 2
        assert len(every) >= len(touch)

    def test_book_per_instant(self, fresh, monkeypatch) -> None:
        counter = _count(monkeypatch, engine, "order_book")
        at = fresh.events["timestamp"].iloc[len(fresh.events) // 2]
        book = fresh.book(at)
        assert fresh.book(at) is book
        assert counter.calls == 1
        assert fresh.book() is not book
        assert counter.calls == 2

    def test_bars(self, fresh, monkeypatch) -> None:
        bars_module = importlib.import_module("ob_analytics.bars")

        counter = _count(monkeypatch, bars_module, "bars")
        first = fresh.bars("tick", 5)
        assert fresh.bars("tick", 5) is first
        assert counter.calls == 1
        fresh.bars("tick", 10)
        assert counter.calls == 2

    def test_features(self, fresh, monkeypatch) -> None:
        features_module = importlib.import_module("ob_analytics.features")

        counter = _count(monkeypatch, features_module, "features")
        first = fresh.features("tick", 5, include=["spread"])
        assert fresh.features("tick", 5, include=("spread",)) is first
        assert counter.calls == 1
        assert list(first.attrs["features"]) == ["spread"]

    def test_tables_match_the_functions(self, fresh) -> None:
        from ob_analytics import order_book, order_lifecycles, queue_positions
        from ob_analytics.bars import bars

        pd.testing.assert_frame_equal(
            fresh.lifecycles(), order_lifecycles(fresh.events)
        )
        pd.testing.assert_frame_equal(fresh.queue(), queue_positions(fresh.events))
        pd.testing.assert_frame_equal(fresh.book().bids, order_book(fresh.events).bids)
        pd.testing.assert_frame_equal(
            fresh.bars("tick", 5), bars(fresh.trades, "tick", 5)
        )

    def test_unhashable_setting_is_built_each_time(self, fresh, monkeypatch) -> None:
        bars_module = importlib.import_module("ob_analytics.bars")

        counter = _count(monkeypatch, bars_module, "bars")
        threshold = pd.Series([5]).tolist()  # a list cannot be a key
        with pytest.raises(ConfigError):
            fresh.bars("tick", threshold)
        assert counter.calls == 1

    def test_a_replaced_result_keeps_nothing(self, fresh) -> None:
        fresh.lifecycles()
        assert replace(fresh)._kept == {}

    def test_kept_tables_are_not_in_the_repr(self, fresh) -> None:
        fresh.lifecycles()
        assert "_kept" not in repr(fresh)


class TestPlotsReuseTheKeptCopies:
    def test_a_second_plot_converts_nothing(self, fresh, monkeypatch) -> None:
        convert = _count(monkeypatch, gallery, "display_result")
        scene = _count(monkeypatch, _concepts, "Scene")
        fresh.plot("trade_tape", "L2")
        fresh.plot("depth_heatmap", "L2")
        gallery.build_gallery_model(fresh)
        assert (convert.calls, scene.calls) == (1, 1)

    def test_each_model_is_new(self, fresh) -> None:
        first = gallery.build_gallery_model(fresh)
        first.analytics.append(first.analytics[0])
        second = gallery.build_gallery_model(fresh)
        assert second is not first
        assert len(second.analytics) == len(first.analytics) - 1

    def test_one_gallery_builds_lifecycles_and_queue_once(
        self, fresh, monkeypatch, tmp_path
    ) -> None:
        lifecycles = _count(monkeypatch, engine, "order_lifecycles")
        queue = _count(monkeypatch, engine, "queue_positions")
        gallery.generate_gallery(fresh, tmp_path, view="l3", backends=["matplotlib"])
        assert (lifecycles.calls, queue.calls) == (1, 1)


class TestSaveAndLoad:
    def test_round_trip(self, fresh, tmp_path) -> None:
        save_data(fresh, tmp_path, extra_tables=["lifecycles", "queue"])
        assert {p.stem for p in tmp_path.glob("*.parquet")} >= {"lifecycles", "queue"}
        loaded = load_result(tmp_path)
        assert loaded._kept.keys() == {"lifecycles", ("queue", "touch")}
        pd.testing.assert_frame_equal(loaded.lifecycles(), fresh.lifecycles())
        pd.testing.assert_frame_equal(loaded.queue(), fresh.queue())

    def test_loaded_tables_are_not_rebuilt(self, fresh, tmp_path, monkeypatch) -> None:
        save_data(fresh, tmp_path, extra_tables=["lifecycles", "queue"])
        loaded = load_result(tmp_path)
        lifecycles = _count(monkeypatch, engine, "order_lifecycles")
        queue = _count(monkeypatch, engine, "queue_positions")
        loaded.lifecycles()
        loaded.queue()
        assert (lifecycles.calls, queue.calls) == (0, 0)

    def test_default_saves_neither_and_removes_an_old_one(
        self, fresh, tmp_path
    ) -> None:
        save_data(fresh, tmp_path, extra_tables=["lifecycles"])
        save_data(fresh, tmp_path)
        assert not (tmp_path / "lifecycles.parquet").exists()
        assert load_result(tmp_path)._kept == {}

    def test_one_name_is_one_table(self, fresh, tmp_path) -> None:
        save_data(fresh, tmp_path, extra_tables="queue")
        assert (tmp_path / "queue.parquet").exists()
        assert not (tmp_path / "lifecycles.parquet").exists()

    def test_unknown_table_is_refused(self, fresh, tmp_path) -> None:
        with pytest.raises(ConfigError, match="'book'"):
            save_data(fresh, tmp_path, extra_tables=["book"])

    def test_tables_that_are_not_a_result_are_refused(self, fresh, tmp_path) -> None:
        with pytest.raises(ConfigError, match="PipelineResult"):
            save_data(fresh.tables(), tmp_path, extra_tables=["queue"])


def test_root_exports_the_per_order_functions() -> None:
    import ob_analytics
    from ob_analytics import order_book, order_lifecycles, queue_positions
    from ob_analytics.book import order_book as book_order_book
    from ob_analytics.queue import queue_positions as queue_queue_positions

    assert order_book is book_order_book
    assert queue_positions is queue_queue_positions
    assert {"order_book", "order_lifecycles", "queue_positions"} <= set(
        ob_analytics.__all__
    )
    assert callable(order_lifecycles)
