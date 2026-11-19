"""The table contracts match the frames the package returns (issue #499).

:data:`ob_analytics.schemas.TABLES` holds one contract per table.  These tests
run the bundled sample and check each table against its contract, and check
the two things built on the contracts: :func:`~ob_analytics.schemas.depth_bins`
and :func:`~ob_analytics.display.display_result`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ob_analytics.cost import transaction_costs
from ob_analytics.display import display_result
from ob_analytics.schemas import (
    PRICE,
    SIZE,
    TABLES,
    DepthBin,
    depth_bin_name,
    depth_bins,
)


@pytest.fixture(scope="module")
def tables(bundled_sample_result) -> dict[str, pd.DataFrame]:
    result = bundled_sample_result
    at = result.events["timestamp"].iloc[len(result.events) // 2]
    return {
        "events": result.events,
        "trades": result.trades,
        "depth": result.depth,
        "depth_summary": result.depth_summary,
        "lifecycles": result.lifecycles(),
        "queue": result.queue(),
        "book": result.book(at).bids,
        "bars": result.bars(),
        "costs": transaction_costs(result.trades, result.depth_summary),
    }


def test_every_table_has_a_contract(tables) -> None:
    assert set(tables) == set(TABLES)


@pytest.mark.parametrize("name", sorted(TABLES))
def test_table_matches_its_contract(tables, name: str) -> None:
    table = TABLES[name]
    frame = tables[name]
    unknown = [c for c in frame.columns if table.column(c) is None]
    assert not unknown, f"{name}: columns with no contract: {unknown}"
    missing = set(table.required) - set(frame.columns)
    assert not missing, f"{name}: required columns missing: {missing}"
    for column_name in frame.columns:
        column = table.column(column_name)
        assert column is not None
        if column.dtype != "object":
            assert frame[column_name].dtype == column.dtype, f"{name}.{column_name}"


@pytest.mark.parametrize("name", ["bars", "costs", "queue", "book"])
def test_fixed_tables_keep_the_contract_order(tables, name: str) -> None:
    assert tuple(tables[name].columns) == TABLES[name].names


def test_depth_bins_reads_side_and_distance() -> None:
    names = ["timestamp", "ask_vol50bps", "bid_vol100bps", "bid_vol25bps", "x_vol5bps"]
    assert depth_bins(names) == [
        DepthBin("bid_vol25bps", "bid", 25),
        DepthBin("bid_vol100bps", "bid", 100),
        DepthBin("ask_vol50bps", "ask", 50),
    ]
    assert depth_bins(names, "ask") == [DepthBin("ask_vol50bps", "ask", 50)]


def test_depth_bins_reads_back_the_engine_names(bundled_sample_result) -> None:
    summary = bundled_sample_result.depth_summary
    bins = depth_bins(summary.columns)
    assert [depth_bin_name(b.side, b.bps) for b in bins] == [b.column for b in bins]
    bid = [b.bps for b in bins if b.side == "bid"]
    assert bid == [b.bps for b in bins if b.side == "ask"]
    assert bid == sorted(bid) and len(bid) > 0


def test_display_result_converts_every_tagged_column(bundled_sample_result) -> None:
    raw = bundled_sample_result
    shown = display_result(raw)
    tick, lot = raw.config.tick_size, raw.config.lot_size
    for name in ("events", "trades", "depth", "depth_summary"):
        table = TABLES[name]
        before, after = getattr(raw, name), getattr(shown, name)
        for kind, step in ((PRICE, tick), (SIZE, lot)):
            for column in table.columns_of(before.columns, kind):
                assert after[column].dtype == np.float64, f"{name}.{column}"
                np.testing.assert_allclose(
                    after[column].to_numpy(), before[column].to_numpy() * step
                )
        others = [
            c
            for c in before.columns
            if c not in table.columns_of(before.columns, PRICE)
            and c not in table.columns_of(before.columns, SIZE)
        ]
        pd.testing.assert_frame_equal(after[others], before[others])


def test_display_result_is_idempotent(bundled_sample_result) -> None:
    shown = display_result(bundled_sample_result)
    again = display_result(shown)
    pd.testing.assert_frame_equal(again.depth_summary, shown.depth_summary)
    pd.testing.assert_frame_equal(again.events, shown.events)


def test_display_result_refuses_a_plain_object() -> None:
    with pytest.raises(TypeError, match="pipeline result"):
        display_result(object())
