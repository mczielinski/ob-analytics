"""The finders return a DataFrame of findings with their summary in ``attrs``."""

from __future__ import annotations

import pandas as pd
import pytest

from ob_analytics import (
    ConfigError,
    classify_trade_sign,
    detect_icebergs,
    detect_sequence_gaps,
    detect_stale_orders,
    hidden_trades,
    iceberg_slices,
    toy_trades,
)
from ob_analytics.analytics import set_order_types
from ob_analytics.datasets import toy_events
from ob_analytics.depth import price_level_volume, summarize_depth


@pytest.fixture(scope="module")
def toy():
    events = toy_events()
    trades = toy_trades()
    events = set_order_types(events, trades)
    summary = summarize_depth(price_level_volume(events))
    return events, trades, summary


def test_detect_icebergs(toy) -> None:
    events, trades, _ = toy
    found = detect_icebergs(events, trades)
    assert isinstance(found, pd.DataFrame)
    assert {"timestamp_start", "timestamp_end", "confidence"} <= set(found.columns)
    assert set(found.attrs) >= {"max_delay", "n_filled_out"}


def test_iceberg_slices(toy) -> None:
    events, trades, _ = toy
    slices = iceberg_slices(events, trades)
    assert isinstance(slices, pd.DataFrame)
    assert {"iceberg", "slice", "direction", "price", "confidence"} <= set(
        slices.columns
    )
    assert "max_delay" in slices.attrs


def test_hidden_trades(toy) -> None:
    events, trades, summary = toy
    inside = hidden_trades(events, trades, summary)
    assert isinstance(inside, pd.DataFrame)
    assert inside.attrs["n_trades"] == len(trades)
    assert 0 <= inside.attrs["n_readable"] <= len(trades)


def test_detect_stale_orders(toy) -> None:
    events, trades, _ = toy
    stale = detect_stale_orders(events, trades)
    assert isinstance(stale, pd.DataFrame)
    assert list(stale.columns) == [
        "id",
        "direction",
        "price",
        "disproved_at",
        "stale_seconds",
        "touch_seconds",
    ]
    assert stale.attrs["grace"] == pd.Timedelta(seconds=1)


def _channels(sequence: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence": pd.array(sequence, dtype="Int64"),
            "venue": pd.Categorical(["a"] * len(sequence)),
        }
    )


def test_detect_sequence_gaps_rows_and_summary() -> None:
    gaps = detect_sequence_gaps(_channels([1, 2, 5, 4]))
    assert gaps[["previous_sequence", "sequence", "missing", "out_of_order"]].to_dict(
        "records"
    ) == [
        {"previous_sequence": 2, "sequence": 5, "missing": 2, "out_of_order": False},
        {"previous_sequence": 5, "sequence": 4, "missing": 0, "out_of_order": True},
    ]
    assert gaps.attrs["n_missing"] == 2
    assert gaps.attrs["first_break_sequence"] == 2


def test_detect_sequence_gaps_keeps_the_channel_dtype() -> None:
    """A clean and a broken feed give the same column dtypes."""
    clean = detect_sequence_gaps(_channels([1, 2, 3]))
    broken = detect_sequence_gaps(_channels([1, 3]))
    assert clean.empty and not broken.empty
    assert clean.dtypes.to_dict() == broken.dtypes.to_dict()


def test_classify_trade_sign_refuses_a_missing_sign_method() -> None:
    with pytest.raises(ConfigError, match="sign_method"):
        classify_trade_sign(toy_trades(), sign_method=None)  # ty: ignore[invalid-argument-type]
