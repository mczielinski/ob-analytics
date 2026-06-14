"""Tests for ob_analytics._chart_data – backend-agnostic data preparation."""

import numpy as np
import pandas as pd
import pytest

from ob_analytics.visualization._data import (
    _default_start_end,
    _price_axis_breaks,
    infer_volume_scale,
    normalized_marker_areas,
    prepare_book_snapshot_data,
    prepare_cancellations_l3_data,
    prepare_event_map_data,
    prepare_events_histogram_data,
    prepare_kyle_lambda_data,
    prepare_liquidity_at_touch_data,
    prepare_ofi_data,
    prepare_order_activity_l3_data,
    prepare_order_outcome_l3_data,
    prepare_time_series_data,
    prepare_trade_tape_l3_data,
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


class TestInferVolumeScale:
    def test_already_in_target_range_returns_one(self) -> None:
        assert infer_volume_scale(np.array([1.0, 2.0, 3.0, 5.0])) == 1.0

    def test_satoshi_scale_volumes(self) -> None:
        # Bitstamp-style volumes (≈1e8 satoshis): scale should bring
        # the median into [0.1, 100).
        scale = infer_volume_scale(np.array([1e8, 2e8, 5e8]))
        scaled_median = float(np.median(np.array([1e8, 2e8, 5e8]) * scale))
        assert 0.1 <= scaled_median < 100
        assert scale == 1e-8

    def test_fractional_volumes_scaled_up(self) -> None:
        scale = infer_volume_scale(np.array([1e-4, 5e-4, 9e-4]))
        scaled_median = float(np.median(np.array([1e-4, 5e-4, 9e-4]) * scale))
        assert 0.1 <= scaled_median < 100
        assert scale == 1e4

    def test_empty_input_returns_one(self) -> None:
        assert infer_volume_scale(np.array([])) == 1.0

    def test_nonpositive_median_returns_one(self) -> None:
        assert infer_volume_scale(np.array([0.0, 0.0, 0.0, 1.0])) == 1.0

    def test_nan_handled(self) -> None:
        # nanmedian ignores NaNs.
        scale = infer_volume_scale(np.array([np.nan, 100.0, 100.0]))
        assert scale == 0.01

    def test_accepts_pandas_series(self) -> None:
        s = pd.Series([1e6, 2e6, 3e6])
        assert infer_volume_scale(s) == 1e-6


class TestNormalizedMarkerAreas:
    def test_empty_input_returns_empty(self) -> None:
        out = normalized_marker_areas(np.array([]))
        assert out.shape == (0,)

    def test_bounded_within_lo_hi(self) -> None:
        vols = np.array([0.0, 1.0, 5.0, 50.0, 1000.0])
        out = normalized_marker_areas(vols, lo=10.0, hi=120.0)
        assert out.min() >= 10.0
        assert out.max() <= 120.0

    def test_zero_maps_to_lo(self) -> None:
        out = normalized_marker_areas(np.array([0.0, 100.0]), lo=8.0, hi=90.0)
        assert out[0] == pytest.approx(8.0)

    def test_saturates_above_reference(self) -> None:
        # Values at/above the reference quantile clamp to ``hi``.
        vols = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        out = normalized_marker_areas(vols, lo=10.0, hi=120.0, ref_quantile=0.8)
        assert out[-1] == pytest.approx(120.0)

    def test_degenerate_all_zero_returns_lo(self) -> None:
        out = normalized_marker_areas(np.zeros(5), lo=12.0, hi=180.0)
        assert np.all(out == 12.0)

    def test_monotonic_in_volume(self) -> None:
        vols = np.array([1.0, 2.0, 4.0, 8.0])
        out = normalized_marker_areas(vols)
        assert np.all(np.diff(out) >= 0)


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


class TestPrepareCancellationsL3:
    def test_returns_sided_frames(
        self, sample_cancellation_events: pd.DataFrame
    ) -> None:
        data = prepare_cancellations_l3_data(sample_cancellation_events)
        assert set(data) == {"bids", "asks", "volume_scale"}
        for side in (data["bids"], data["asks"]):
            assert isinstance(side, pd.DataFrame)
            assert {"age_s", "distance_from_touch"} <= set(side.columns)

    def test_age_and_distance_floored_for_log(
        self, sample_cancellation_events: pd.DataFrame
    ) -> None:
        data = prepare_cancellations_l3_data(sample_cancellation_events)
        both = pd.concat([data["bids"], data["asks"]])
        assert not both.empty
        # §3.3: floored onto the log scale (no zeros), distance is positive bps.
        assert (both["age_s"] >= 1e-3).all()
        assert (both["distance_from_touch"] >= 0.05).all()

    def test_drops_unmatched_and_uncreated(
        self, sample_cancellation_events: pd.DataFrame
    ) -> None:
        # A deleted order whose creation we never saw has no distance_bps and
        # must be dropped (the inner merge + dropna handles it).
        extra = sample_cancellation_events.copy()
        orphan = extra.iloc[[1]].copy()  # a 'deleted' row...
        orphan["id"] = 999  # ...with no matching 'created'
        extra = pd.concat([extra, orphan], ignore_index=True)
        data = prepare_cancellations_l3_data(extra)
        both = pd.concat([data["bids"], data["asks"]])
        assert (both["age_s"].notna()).all()
        assert 999 not in both["id"].to_numpy()


class TestPrepareOrderActivityL3:
    def test_returns_fate_frames(
        self, sample_order_lifecycle_events: pd.DataFrame
    ) -> None:
        data = prepare_order_activity_l3_data(sample_order_lifecycle_events)
        assert set(data) == {
            "filled",
            "cancelled",
            "resting",
            "volume_scale",
            "price_by",
            "y_range",
            "shown_of",
            "show_markers",
        }
        for fate in (data["filled"], data["cancelled"], data["resting"]):
            assert isinstance(fate, pd.DataFrame)
            assert {"start_ts", "end_ts", "price", "linewidth"} <= set(fate.columns)
        # Fixture exercises all three outcomes.
        assert not data["filled"].empty
        assert not data["cancelled"].empty
        assert not data["resting"].empty

    def test_spans_split_by_outcome(
        self, sample_order_lifecycle_events: pd.DataFrame
    ) -> None:
        data = prepare_order_activity_l3_data(sample_order_lifecycle_events)
        assert (data["cancelled"]["outcome"] == "cancelled").all()
        assert data["filled"]["outcome"].isin(["filled", "partial"]).all()
        assert (data["resting"]["outcome"] == "resting").all()
        # One span per surviving order id, never per raw event.
        for fate in (data["filled"], data["cancelled"], data["resting"]):
            assert fate["id"].is_unique

    def test_cancelled_span_ends_at_delete(
        self, sample_order_lifecycle_events: pd.DataFrame
    ) -> None:
        data = prepare_order_activity_l3_data(sample_order_lifecycle_events)
        cancelled = data["cancelled"]
        # Each cancelled order ran 3s from create to delete (< the window), so
        # its span must terminate before the window end.
        assert (cancelled["end_ts"] > cancelled["start_ts"]).all()
        assert (
            cancelled["end_ts"] < sample_order_lifecycle_events["timestamp"].max()
        ).all()

    def test_forever_resting_span_extends_to_end(
        self, sample_order_lifecycle_events: pd.DataFrame
    ) -> None:
        # The created-only order (id 300) never terminates, so its span runs to
        # the window end rather than collapsing to its single event.
        end_time = sample_order_lifecycle_events["timestamp"].max()
        data = prepare_order_activity_l3_data(sample_order_lifecycle_events)
        forever = data["resting"].set_index("id").loc[300]
        assert forever["end_ts"] == end_time
        assert forever["end_ts"] > forever["start_ts"]

    def test_density_degradation_keeps_fills_samples_flood(
        self, sample_order_lifecycle_events: pd.DataFrame
    ) -> None:
        # Every fill survives; the cancelled/resting flood is sampled down to
        # the remaining budget, with the ratio reported.  A wide price window
        # disables the percentile clip so the counts are exact.
        data = prepare_order_activity_l3_data(
            sample_order_lifecycle_events, max_spans=5, price_from=0.0, price_to=1e9
        )
        assert data["shown_of"] is not None
        shown, total = data["shown_of"]
        assert total == 7  # 3 cancelled + 3 filled + 1 resting
        assert len(data["filled"]) == 3  # fills never sampled away
        # flood (cancelled + resting) sampled to fill the remaining budget (2)
        assert len(data["cancelled"]) + len(data["resting"]) == 2
        assert shown == 5

    def test_linewidth_encodes_size(
        self, sample_order_lifecycle_events: pd.DataFrame
    ) -> None:
        data = prepare_order_activity_l3_data(sample_order_lifecycle_events)
        allspans = pd.concat(
            [data["filled"], data["cancelled"], data["resting"]], ignore_index=True
        )
        assert allspans["linewidth"].min() >= 0.5
        assert allspans["linewidth"].max() <= 4.5 + 1e-9
        # The largest order draws the widest line.
        widest = allspans.loc[allspans["volume"].idxmax(), "linewidth"]
        assert widest == allspans["linewidth"].max()


class TestPrepareLiquidityAtTouch:
    def test_returns_paired_series(self, sample_depth_summary: pd.DataFrame) -> None:
        data = prepare_liquidity_at_touch_data(sample_depth_summary)
        assert set(data) == {"timestamp", "bid_vol", "ask_vol", "volume_scale"}
        assert len(data["timestamp"]) == len(sample_depth_summary)
        assert len(data["bid_vol"]) == len(data["ask_vol"]) == len(data["timestamp"])

    def test_scales_volume(self, sample_depth_summary: pd.DataFrame) -> None:
        raw = prepare_liquidity_at_touch_data(sample_depth_summary, volume_scale=1.0)
        scaled = prepare_liquidity_at_touch_data(sample_depth_summary, volume_scale=2.0)
        assert np.allclose(
            scaled["bid_vol"].to_numpy(), raw["bid_vol"].to_numpy() * 2.0
        )

    def test_window_filters_by_time(self, sample_depth_summary: pd.DataFrame) -> None:
        mid = sample_depth_summary["timestamp"].iloc[15]
        data = prepare_liquidity_at_touch_data(sample_depth_summary, start_time=mid)
        assert (data["timestamp"] >= mid).all()


class TestPrepareOrderOutcomeL3:
    def test_returns_outcome_frames(
        self, sample_executed_orders: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        events, trades = sample_executed_orders
        data = prepare_order_outcome_l3_data(events, bps_quantiles=(0.0, 1.0))
        assert set(data) == {"filled", "partial", "cancelled", "volume_scale"}
        for frame in (data["filled"], data["partial"], data["cancelled"]):
            assert {"distance_bps", "placed", "marker_area"} <= set(frame.columns)

    def test_competing_risks_classification(
        self, sample_executed_orders: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        events, trades = sample_executed_orders
        data = prepare_order_outcome_l3_data(events, bps_quantiles=(0.0, 1.0))
        # filled via maker (1, 6) + via taker (5); partial=2; cancelled=3.
        assert set(data["filled"]["id"]) == {1, 5, 6}
        assert set(data["partial"]["id"]) == {2}
        assert set(data["cancelled"]["id"]) == {3}

    def test_censored_orders_dropped(
        self, sample_executed_orders: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        events, trades = sample_executed_orders
        data = prepare_order_outcome_l3_data(events, bps_quantiles=(0.0, 1.0))
        all_ids = pd.concat([data["filled"], data["partial"], data["cancelled"]])["id"]
        # id 4 never filled and never deleted -> censored -> absent.
        assert 4 not in all_ids.to_numpy()

    def test_bounded_markers(
        self, sample_executed_orders: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        events, trades = sample_executed_orders
        data = prepare_order_outcome_l3_data(events, bps_quantiles=(0.0, 1.0))
        both = pd.concat([data["filled"], data["partial"], data["cancelled"]])
        assert both["marker_area"].between(10.0, 120.0).all()


class TestPrepareTradeTapeL3:
    def test_returns_sided_frames(
        self, sample_executed_orders: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        events, trades = sample_executed_orders
        data = prepare_trade_tape_l3_data(events, trades, price_from=0.0, price_to=1e9)
        assert set(data) == {"buys", "sells", "volume_scale", "y_range"}
        for side in (data["buys"], data["sells"]):
            assert {"created_ts", "timestamp", "price", "marker_area"} <= set(
                side.columns
            )

    def test_maker_bars_precede_executions(
        self, sample_executed_orders: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        events, trades = sample_executed_orders
        data = prepare_trade_tape_l3_data(events, trades, price_from=0.0, price_to=1e9)
        both = pd.concat([data["buys"], data["sells"]])
        # Each maker order was created before the trade that consumed it.
        assert (both["created_ts"] <= both["timestamp"]).all()

    def test_split_by_aggressor_side(
        self, sample_executed_orders: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        events, trades = sample_executed_orders
        data = prepare_trade_tape_l3_data(events, trades, price_from=0.0, price_to=1e9)
        # T2 (buy) -> maker id 2; T1+T4 (sell) -> maker ids 1, 6. The un-tracked
        # maker in T3 (sentinel event) is dropped by the inner merge.
        assert set(data["buys"]["id"]) == {2}
        assert set(data["sells"]["id"]) == {1, 6}

    def test_y_range_spans_prices(
        self, sample_executed_orders: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        events, trades = sample_executed_orders
        data = prepare_trade_tape_l3_data(events, trades, price_from=0.0, price_to=1e9)
        lo, hi = data["y_range"]
        assert lo <= hi


class TestPrepareBookSnapshot:
    def test_returns_book_sides(self, sample_order_book: dict) -> None:
        data = prepare_book_snapshot_data(sample_order_book)
        assert "bids" in data
        assert "asks" in data
        assert "timestamp" in data
        assert data["per_order"] is False
        for side in (data["bids"], data["asks"]):
            assert isinstance(side, pd.DataFrame)
            assert {"price", "volume", "liquidity", "seg_lo", "seg_hi"} <= set(
                side.columns
            )

    def test_volume_scale_applied(self, sample_order_book: dict) -> None:
        data = prepare_book_snapshot_data(sample_order_book, volume_scale=0.5)
        # Cumulative liquidity is scaled: ask cumsum 400 * 0.5 == 200.
        assert data["asks"]["liquidity"].max() <= 200

    def test_aggregate_zeroes_segment_floor(self, sample_order_book: dict) -> None:
        # L2 bars sit on the axis: each level is one segment from 0.
        data = prepare_book_snapshot_data(sample_order_book, per_order=False)
        assert (data["bids"]["seg_lo"] == 0).all()
        assert (data["asks"]["seg_lo"] == 0).all()

    def test_per_order_stacks_within_level(self) -> None:
        # Two orders at one price stack: seg_lo of the second == seg_hi of first.
        book = {
            "timestamp": 1430445600,
            "bids": np.array([[236.0, 100, 100], [236.0, 50, 150]]),
            "asks": np.empty((0, 3)),
        }
        data = prepare_book_snapshot_data(book, per_order=True, volume_scale=1.0)
        bids = data["bids"]
        assert data["per_order"] is True
        assert len(bids) == 2
        assert bids["seg_lo"].tolist() == [0.0, 100.0]
        assert bids["seg_hi"].tolist() == [100.0, 150.0]


class TestPrepareVolumePercentiles:
    def test_returns_cumsum_data(self, sample_depth_summary: pd.DataFrame) -> None:
        data = prepare_volume_percentiles_data(sample_depth_summary)
        assert "asks_cumsum" in data
        assert "bids_cumsum_neg" in data
        assert "colors_dict" in data
        assert "legend_names" in data
        assert len(data["asks_cols"]) == 20
        assert len(data["bids_cols"]) == 20

    def test_palette_is_sequential_luminance(self) -> None:
        from ob_analytics.visualization._data import _volume_percentile_palette

        for n in (2, 10, 20, 40):
            pal = np.array([c[:3] for c in _volume_percentile_palette(n)])
            lum = pal @ np.array([0.2126, 0.7152, 0.0722])
            diffs = np.diff(lum)
            # Monotonic luminance => ordered ramp, not a rainbow (jet zig-zags).
            assert np.all(diffs < 0) or np.all(diffs > 0)

    def test_any_depth_bps_configuration_works(self) -> None:
        # Regression: the prepare hardcoded 25-500bps columns and raised
        # KeyError for any other depth_bps/depth_bins (the documented
        # PipelineConfig(depth_bps=50) example broke the face).
        ts = pd.date_range("2024-01-01", periods=10, freq="s")
        cols: dict = {"timestamp": ts}
        for b in range(50, 1001, 50):  # depth_bps=50, depth_bins=20
            cols[f"bid_vol{b}bps"] = np.ones(10)
            cols[f"ask_vol{b}bps"] = np.ones(10)
        data = prepare_volume_percentiles_data(pd.DataFrame(cols))
        assert len(data["asks_cols"]) == 20
        assert len(data["colors_dict"]) == 40
        # §3.7: legend collapsed to 3 representative depths per side
        # (touch / mid / far), ordered touch-first.
        assert len(data["legend_entries"]) == 6
        assert len(data["legend_names"]) == 6
        assert data["legend_names"][0] == "+0050bps"

    def test_ramp_is_dark_at_touch(self) -> None:
        # §3.7 importance ↦ salience: the near-touch band (index 0) is the
        # darkest and fades outward to the far-depth band.
        from ob_analytics.visualization._data import _volume_percentile_palette

        w = np.array([0.2126, 0.7152, 0.0722])
        lum = np.array([c[:3] for c in _volume_percentile_palette(10, hue="blue")]) @ w
        assert lum[0] < lum[-1]

    def test_two_hue_families_share_luminance(self) -> None:
        # §3.7: asks (blue) and bids (orange) share a luminance ramp so they
        # stay distinguishable in grayscale, but differ in hue in colour.
        from ob_analytics.visualization._data import _volume_percentile_palette

        w = np.array([0.2126, 0.7152, 0.0722])
        blue = np.array([c[:3] for c in _volume_percentile_palette(10, "blue")]) @ w
        orange = np.array([c[:3] for c in _volume_percentile_palette(10, "orange")]) @ w
        assert np.allclose(blue, orange, atol=0.08)  # grayscale-equivalent
        b0 = _volume_percentile_palette(2, "blue")[0]
        o0 = _volume_percentile_palette(2, "orange")[0]
        assert b0[2] > o0[2]  # blue family is bluer
        assert o0[0] > b0[0]  # orange family is redder


class TestPrepareEventsHistogram:
    def test_returns_filtered_events(self, sample_events: pd.DataFrame) -> None:
        data = prepare_events_histogram_data(sample_events, val="price")
        assert "events" in data
        assert data["val"] == "price"

    def test_invalid_val_raises(self, sample_events: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="val must be"):
            prepare_events_histogram_data(sample_events, val="invalid")

    def test_clips_price_to_focus_window(self, sample_events: pd.DataFrame) -> None:
        # The price face was a single 1px spike because q01-q99 of a heavy-tailed
        # book still spans far-from-touch flashed orders.  A caller-supplied
        # focus window must clip the events to the near-touch band.
        data = prepare_events_histogram_data(
            sample_events, val="price", price_from=236.4, price_to=236.7
        )
        prices = data["events"]["price"]
        assert prices.min() >= 236.4
        assert prices.max() <= 236.7
        assert len(prices) < len(sample_events)  # outliers were clipped


class TestPrepareVpin:
    def test_returns_vpin_data(self) -> None:
        ts = pd.date_range("2015-01-01", periods=5, freq="min")
        vpin_df = pd.DataFrame(
            {
                "timestamp_end": ts,
                "vpin": [0.3, 0.5, 0.7, 0.4, 0.6],
                "vpin_avg": [0.3, 0.4, 0.5, 0.45, 0.5],
            }
        )
        data = prepare_vpin_data(vpin_df, threshold=0.7)
        assert data["threshold"] == 0.7
        assert "bar_width" in data
        assert len(data["vpin_df"]) == 5


class TestPrepareOfi:
    def test_returns_colors_and_bar_width(self) -> None:
        ts = pd.date_range("2015-01-01", periods=5, freq="min")
        ofi_df = pd.DataFrame(
            {
                "timestamp": ts,
                "ofi": [0.3, -0.5, 0.7, -0.4, 0.6],
            }
        )
        data = prepare_ofi_data(ofi_df)
        assert "colors" in data
        assert len(data["colors"]) == 5
        assert data["colors"][0] == "#27ae60"  # positive → green
        assert data["colors"][1] == "#e74c3c"  # negative → red


class TestPrepareKyleLambda:
    def test_extracts_fields(self) -> None:
        class FakeResult:
            regression_df = pd.DataFrame(
                {
                    "signed_volume": [1.0, -2.0, 3.0],
                    "delta_price": [0.01, -0.02, 0.03],
                }
            )
            lambda_ = 0.01
            r_squared = 0.5
            t_stat = 2.1

        data = prepare_kyle_lambda_data(FakeResult())
        assert data["lambda_"] == 0.01
        assert data["r_squared"] == 0.5
        assert len(data["reg_df"]) == 3
