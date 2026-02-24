"""Tests for ob_analytics._chart_data – backend-agnostic data preparation."""

import numpy as np
import pandas as pd
import pytest

from ob_analytics._chart_data import (
    _default_start_end,
    _price_axis_breaks,
    prepare_current_depth_data,
    prepare_event_map_data,
    prepare_events_histogram_data,
    prepare_kyle_lambda_data,
    prepare_ofi_data,
    prepare_price_levels_data,
    prepare_time_series_data,
    prepare_trades_data,
    prepare_volume_map_data,
    prepare_volume_percentiles_data,
    prepare_vpin_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_trades() -> pd.DataFrame:
    ts = pd.Timestamp("2015-05-01 01:00:00")
    return pd.DataFrame(
        {
            "timestamp": [ts + pd.Timedelta(seconds=i) for i in range(5)],
            "price": [236.50, 236.55, 236.45, 236.60, 236.50],
            "volume": [100, 200, 150, 300, 250],
            "direction": pd.Categorical(
                ["buy", "sell", "buy", "sell", "buy"],
                categories=["buy", "sell"],
            ),
        }
    )


@pytest.fixture
def sample_events() -> pd.DataFrame:
    ts = pd.Timestamp("2015-05-01 01:00:00")
    n = 20
    return pd.DataFrame(
        {
            "timestamp": [ts + pd.Timedelta(seconds=i) for i in range(n)],
            "price": np.linspace(236.0, 237.0, n),
            "volume": np.random.default_rng(42).uniform(100, 1000, n),
            "direction": pd.Categorical(
                ["bid", "ask"] * (n // 2), categories=["bid", "ask"]
            ),
            "action": pd.Categorical(
                ["created", "deleted"] * (n // 2),
                categories=["created", "changed", "deleted"],
                ordered=True,
            ),
            "type": ["flashed-limit"] * 10 + ["resting-limit"] * 10,
        }
    )


@pytest.fixture
def sample_depth_summary() -> pd.DataFrame:
    ts = pd.Timestamp("2015-05-01 01:00:00")
    n = 30
    rng = np.random.default_rng(42)
    data: dict = {"timestamp": [ts + pd.Timedelta(seconds=i) for i in range(n)]}
    for side in ("bid", "ask"):
        for bps in range(25, 501, 25):
            data[f"{side}_vol{bps}bps"] = rng.uniform(10, 500, n)
    data["best_bid_price"] = np.full(n, 236.50)
    data["best_ask_price"] = np.full(n, 237.00)
    data["best_bid_vol"] = rng.uniform(100, 1000, n)
    data["best_ask_vol"] = rng.uniform(100, 1000, n)
    return pd.DataFrame(data)


@pytest.fixture
def sample_order_book() -> dict:
    return {
        "timestamp": 1430445600,
        "bids": np.array([[236.50, 100, 100], [236.00, 200, 300]]),
        "asks": np.array([[237.00, 150, 150], [237.50, 250, 400]]),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestDefaultStartEnd:
    def test_fills_none_from_df(self, sample_trades: pd.DataFrame) -> None:
        start, end = _default_start_end(sample_trades, None, None)
        assert start == sample_trades["timestamp"].min()
        assert end == sample_trades["timestamp"].max()

    def test_preserves_explicit_values(self, sample_trades: pd.DataFrame) -> None:
        explicit_start = pd.Timestamp("2015-05-01 00:00:00")
        explicit_end = pd.Timestamp("2015-05-01 02:00:00")
        start, end = _default_start_end(sample_trades, explicit_start, explicit_end)
        assert start == explicit_start
        assert end == explicit_end


class TestPriceAxisBreaks:
    def test_positive_range(self) -> None:
        step, breaks = _price_axis_breaks(236.0, 237.0)
        assert step > 0
        assert len(breaks) > 0

    def test_zero_range(self) -> None:
        step, breaks = _price_axis_breaks(236.0, 236.0)
        assert step == 1.0
        assert len(breaks) == 1


# ---------------------------------------------------------------------------
# Prepare functions
# ---------------------------------------------------------------------------


class TestPrepareTimeSeries:
    def test_returns_dict(self) -> None:
        ts = pd.Series(pd.date_range("2015-01-01", periods=5, freq="s"))
        vals = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        data = prepare_time_series_data(ts, vals, title="test", y_label="val")
        assert isinstance(data, dict)
        assert "df" in data
        assert data["title"] == "test"
        assert len(data["df"]) == 5

    def test_filters_by_time(self) -> None:
        ts = pd.Series(pd.date_range("2015-01-01", periods=10, freq="s"))
        vals = pd.Series(range(10), dtype=float)
        start = ts.iloc[3]
        end = ts.iloc[7]
        data = prepare_time_series_data(ts, vals, start_time=start, end_time=end)
        assert len(data["df"]) == 5  # indices 3..7

    def test_length_mismatch_raises(self) -> None:
        ts = pd.Series(pd.date_range("2015-01-01", periods=5, freq="s"))
        vals = pd.Series([1.0, 2.0])
        with pytest.raises(ValueError, match="Length"):
            prepare_time_series_data(ts, vals)


class TestPrepareTrades:
    def test_returns_dict_with_required_keys(self, sample_trades: pd.DataFrame) -> None:
        data = prepare_trades_data(sample_trades)
        assert "filtered_trades" in data
        assert "y_breaks" in data
        assert len(data["filtered_trades"]) == 5

    def test_filters_by_time(self, sample_trades: pd.DataFrame) -> None:
        start = sample_trades["timestamp"].iloc[1]
        end = sample_trades["timestamp"].iloc[3]
        data = prepare_trades_data(sample_trades, start_time=start, end_time=end)
        assert len(data["filtered_trades"]) == 3


class TestPrepareEventMap:
    def test_returns_created_and_deleted(self, sample_events: pd.DataFrame) -> None:
        data = prepare_event_map_data(sample_events)
        assert "created" in data
        assert "deleted" in data
        assert "events" in data
        assert "price_by" in data
        assert len(data["created"]) + len(data["deleted"]) <= len(data["events"])


class TestPrepareVolumeMap:
    def test_default_action_deleted(self, sample_events: pd.DataFrame) -> None:
        data = prepare_volume_map_data(sample_events)
        assert "events" in data
        assert "log_scale" in data
        assert data["log_scale"] is False

    def test_invalid_action_raises(self, sample_events: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="action must be"):
            prepare_volume_map_data(sample_events, action="invalid")

    def test_log_scale_passed_through(self, sample_events: pd.DataFrame) -> None:
        data = prepare_volume_map_data(sample_events, log_scale=True)
        assert data["log_scale"] is True


class TestPrepareCurrentDepth:
    def test_returns_depth_df(self, sample_order_book: dict) -> None:
        data = prepare_current_depth_data(sample_order_book)
        assert "depth_df" in data
        assert "bids" in data
        assert "asks" in data
        assert "timestamp" in data
        assert isinstance(data["depth_df"], pd.DataFrame)

    def test_volume_scale_applied(self, sample_order_book: dict) -> None:
        data = prepare_current_depth_data(sample_order_book, volume_scale=0.5)
        # Liquidity and volume should be scaled
        assert data["depth_df"]["liquidity"].max() <= 200  # 400 * 0.5


class TestPrepareVolumePercentiles:
    def test_returns_cumsum_data(self, sample_depth_summary: pd.DataFrame) -> None:
        data = prepare_volume_percentiles_data(sample_depth_summary)
        assert "asks_cumsum" in data
        assert "bids_cumsum_neg" in data
        assert "colors_dict" in data
        assert "legend_names" in data
        assert len(data["asks_cols"]) == 20
        assert len(data["bids_cols"]) == 20


class TestPrepareEventsHistogram:
    def test_returns_filtered_events(self, sample_events: pd.DataFrame) -> None:
        data = prepare_events_histogram_data(sample_events, val="price")
        assert "events" in data
        assert data["val"] == "price"

    def test_invalid_val_raises(self, sample_events: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="val must be"):
            prepare_events_histogram_data(sample_events, val="invalid")


class TestPrepareVpin:
    def test_returns_vpin_data(self) -> None:
        ts = pd.date_range("2015-01-01", periods=5, freq="min")
        vpin_df = pd.DataFrame({
            "timestamp_end": ts,
            "vpin": [0.3, 0.5, 0.7, 0.4, 0.6],
            "vpin_avg": [0.3, 0.4, 0.5, 0.45, 0.5],
        })
        data = prepare_vpin_data(vpin_df, threshold=0.7)
        assert data["threshold"] == 0.7
        assert "bar_width" in data
        assert len(data["vpin_df"]) == 5


class TestPrepareOfi:
    def test_returns_colors_and_bar_width(self) -> None:
        ts = pd.date_range("2015-01-01", periods=5, freq="min")
        ofi_df = pd.DataFrame({
            "timestamp": ts,
            "ofi": [0.3, -0.5, 0.7, -0.4, 0.6],
        })
        data = prepare_ofi_data(ofi_df)
        assert "colors" in data
        assert len(data["colors"]) == 5
        assert data["colors"][0] == "#27ae60"  # positive → green
        assert data["colors"][1] == "#e74c3c"  # negative → red


class TestPrepareKyleLambda:
    def test_extracts_fields(self) -> None:
        class FakeResult:
            regression_df = pd.DataFrame({
                "signed_volume": [1.0, -2.0, 3.0],
                "delta_price": [0.01, -0.02, 0.03],
            })
            lambda_ = 0.01
            r_squared = 0.5
            t_stat = 2.1

        data = prepare_kyle_lambda_data(FakeResult())
        assert data["lambda_"] == 0.01
        assert data["r_squared"] == 0.5
        assert len(data["reg_df"]) == 3
