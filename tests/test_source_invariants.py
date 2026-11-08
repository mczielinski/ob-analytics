"""The same invariants, checked on a small run of every offline source.

Each source has its own tests for the details of its format.  This module
asks the questions every source must answer the same way:

* prices and sizes are ``int64`` ticks and lots;
* every clock is a UTC nanosecond clock;
* the depth summary's touch is never crossed: the best bid is never above
  the best ask;
* for a per-order (L3) source, the price levels in ``depth`` equal the
  per-order book summed by price, at every instant the book changes;
* an L3 run's ``depth``, written as a price-level file and read back, gives
  the same depth summary (an L2 replay equals the L3 run);
* a run written with the Parquet writer, and with the source's own writer,
  reads back as the same run.

A new offline source joins by adding one :class:`Case` to :data:`CASES`.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ob_analytics.analytics import order_book
from ob_analytics.bitstamp import BitstampSource
from ob_analytics.data import load_result, save_data
from ob_analytics.databento import DatabentoSource
from ob_analytics.datasets import toy_l2_depth, toy_l2_trades
from ob_analytics.depth_l2 import DepthCsvSettings, DepthCsvSource
from ob_analytics.lobster import LobsterSource
from ob_analytics.pipeline import Pipeline, PipelineResult
from ob_analytics.protocols import Level, RunContext
from ob_analytics.synth import (
    SynthConfig,
    SyntheticLoader,
    SyntheticTradeSource,
    generate_session,
)
from tests._golden import assert_same_result
from tests._mbo import mbo_frame
from tests.conftest import LOBSTER_DAY_DATE, write_lobster_day, write_tiny_bitstamp


@dataclass(frozen=True)
class Case:
    """One offline source and a small input for it.

    ``write_input`` writes the input under the folder it is given and
    returns what :meth:`~ob_analytics.pipeline.Pipeline.run` takes.  When the
    source writes a format of its own (``save_data(fmt=name)``), ``saved_as``
    is the file or folder name to save to and ``reread`` turns that path into
    what ``run`` takes; ``saved_as`` is ``None`` for a source with no writer.
    """

    name: str
    pipeline: Callable[[RunContext], Pipeline]
    write_input: Callable[[Path], object]
    ctx: RunContext = field(default_factory=RunContext)
    saved_as: str | None = None
    reread: Callable[[Path], object] = lambda path: path
    writer_needs: str | None = None


_SESSION = SynthConfig(seed=418, duration=30.0)


def _synth_pipeline(ctx: RunContext) -> Pipeline:
    session = generate_session(_SESSION)
    return Pipeline(
        loader=SyntheticLoader(session),
        trade_source=SyntheticTradeSource(session),
        ctx=ctx,
    )


def _toy_l2(folder: Path) -> Path:
    save_data(
        {"depth": toy_l2_depth(), "trades": toy_l2_trades()}, folder, fmt="depth_csv"
    )
    return folder


def _from(source_cls: type) -> Callable[[RunContext], Pipeline]:
    return lambda ctx: Pipeline(source=source_cls(), ctx=ctx)


CASES = [
    Case("synth", _synth_pipeline, lambda _: None),
    Case(
        "bitstamp",
        _from(BitstampSource),
        write_tiny_bitstamp,
        saved_as="out",
        reread=lambda path: path / "orders.csv",
    ),
    Case(
        "lobster",
        _from(LobsterSource),
        write_lobster_day,
        ctx=RunContext(trading_date=LOBSTER_DAY_DATE),
        saved_as="out",
    ),
    Case(
        "databento",
        _from(DatabentoSource),
        lambda _: mbo_frame(),
        saved_as="out.dbn",
        writer_needs="databento",
    ),
    Case("depth_csv", _from(DepthCsvSource), _toy_l2, saved_as="out"),
]


@pytest.fixture(scope="module", params=CASES, ids=[c.name for c in CASES])
def case(request) -> Case:
    return request.param


@pytest.fixture(scope="module")
def run(case: Case, tmp_path_factory) -> PipelineResult:
    source = case.write_input(tmp_path_factory.mktemp(case.name))
    return case.pipeline(case.ctx).run(source)


# ── Units ──────────────────────────────────────────────────────────────

_INTEGER_COLUMNS = {
    "events": ("price", "volume", "fill"),
    "trades": ("price", "volume"),
    "depth": ("price", "volume"),
    "depth_summary": (
        "best_bid_price",
        "best_bid_vol",
        "best_ask_price",
        "best_ask_vol",
    ),
}


def test_prices_and_sizes_are_int64(run: PipelineResult) -> None:
    for table, columns in _INTEGER_COLUMNS.items():
        frame = getattr(run, table)
        for column in columns:
            assert frame[column].dtype == "int64", f"{table}.{column}"


def test_every_clock_is_utc_nanoseconds(run: PipelineResult) -> None:
    clocks = 0
    for table, frame in run.tables().items():
        for column in frame.columns:
            if pd.api.types.is_datetime64_any_dtype(frame[column]):
                assert str(frame[column].dtype) == "datetime64[ns, UTC]", (
                    f"{table}.{column}"
                )
                clocks += 1
    assert clocks >= 3


# ── The book ────────────────────────────────────────────────────────────


def test_the_touch_is_never_crossed(run: PipelineResult) -> None:
    summary = run.depth_summary
    both = (summary["best_bid_price"] > 0) & (summary["best_ask_price"] > 0)
    assert both.any()
    crossed = summary.loc[
        both & (summary["best_bid_price"] > summary["best_ask_price"])
    ]
    assert crossed.empty, crossed[["timestamp", "best_bid_price", "best_ask_price"]]


def _levels_at(depth: pd.DataFrame, at: pd.Timestamp) -> dict[tuple[str, int], int]:
    """Each non-empty price level's size after the depth rows up to *at*."""
    rows = depth[depth["timestamp"] <= at]
    last = rows.drop_duplicates(subset=["direction", "price"], keep="last")
    last = last[last["volume"] > 0]
    return {
        (str(d), int(p)): int(v)
        for d, p, v in zip(last["direction"], last["price"], last["volume"])
    }


def _book_levels_at(
    events: pd.DataFrame, at: pd.Timestamp
) -> dict[tuple[str, int], int]:
    """The per-order book at *at*, summed by price level."""
    book = order_book(events, tp=at)
    levels: dict[tuple[str, int], int] = {}
    for side, direction in (("bids", "bid"), ("asks", "ask")):
        for price, volume in zip(book[side]["price"], book[side]["volume"]):
            key = (direction, int(price))
            levels[key] = levels.get(key, 0) + int(volume)
    return levels


def test_depth_equals_the_per_order_book(run: PipelineResult) -> None:
    if run.level is not Level.L3:
        pytest.skip("a price-level source has no per-order book")
    instants = run.depth["timestamp"].drop_duplicates()
    if len(instants) > 40:
        instants = instants.iloc[np.linspace(0, len(instants) - 1, 40).astype(int)]
    for at in instants:
        assert _levels_at(run.depth, at) == _book_levels_at(run.events, at), at


def test_an_l2_replay_equals_the_l3_run(run: PipelineResult, tmp_path) -> None:
    if run.level is not Level.L3:
        pytest.skip("the run is already price-level")
    # The CSV's default clock is in milliseconds; nanoseconds keep every
    # instant of the run, so the replay can match it exactly.
    l2 = DepthCsvSource(settings=DepthCsvSettings(timestamp_unit="ns"))
    save_data(run, tmp_path, writer=l2.create_writer(run.config, RunContext()))
    replay = Pipeline(run.config, source=l2).run(tmp_path)

    columns = ["timestamp", "price", "volume", "direction"]
    pd.testing.assert_frame_equal(
        replay.depth[columns].reset_index(drop=True),
        run.depth[columns].reset_index(drop=True),
        check_categorical=False,
    )
    touch = ["best_bid_price", "best_bid_vol", "best_ask_price", "best_ask_vol"]
    pd.testing.assert_frame_equal(
        replay.depth_summary[touch].reset_index(drop=True),
        run.depth_summary[touch].reset_index(drop=True),
    )


# ── Writers ───────────────────────────────────────────────────────────


def test_the_parquet_writer_round_trips(run: PipelineResult, tmp_path) -> None:
    save_data(run, tmp_path)
    assert_same_result(load_result(tmp_path), run)


def test_the_sources_own_writer_round_trips(
    case: Case, run: PipelineResult, tmp_path
) -> None:
    if case.saved_as is None:
        pytest.skip(f"{case.name} has no writer of its own")
    if case.writer_needs and importlib.util.find_spec(case.writer_needs) is None:
        pytest.skip(f"{case.writer_needs} is an optional extra")
    target = tmp_path / case.saved_as
    save_data(run, target, fmt=case.name, ctx=case.ctx)
    back = case.pipeline(case.ctx).run(case.reread(target))

    for name, columns in (
        ("events", ["id", "price", "volume", "action", "direction", "fill"]),
        ("trades", ["timestamp", "price", "volume", "direction"]),
        ("depth", ["timestamp", "price", "volume", "direction"]),
    ):
        pd.testing.assert_frame_equal(
            getattr(back, name)[columns].reset_index(drop=True),
            getattr(run, name)[columns].reset_index(drop=True),
            check_categorical=False,
            obj=name,
        )
