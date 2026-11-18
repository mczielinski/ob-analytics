"""Long series are thinned to what a screen can show (#491).

The Plotly gallery of the bundled sample was 231 MiB, because every step line
carried every row of ``depth_summary``.  The data functions now thin each
series, keeping the lowest and highest value in each time bin, and the Plotly
heatmap is one image.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ob_analytics.visualization._data import (
    Thinned,
    depth_raster,
    prepare_order_activity_l2_data,
    prepare_order_outcome_data,
    prepare_queue_position_data,
    thin_order_lines,
    thin_payload,
    thin_points,
    thin_positions,
)


def _times(n: int) -> pd.Series:
    rng = np.random.default_rng(0)
    gaps = rng.exponential(1.0, n).cumsum()
    return pd.Series(pd.to_datetime(gaps, unit="s", utc=True))


def _bins(ts: pd.Series, bins: int) -> np.ndarray:
    t = pd.DatetimeIndex(ts).asi8
    return np.minimum(((t - t[0]) / (t[-1] - t[0]) * bins).astype(int), bins - 1)


class TestThinPositions:
    def test_keeps_the_extremes_of_every_bin(self) -> None:
        n, bins = 50_000, 200
        ts = _times(n)
        rng = np.random.default_rng(1)
        walk = rng.normal(size=n).cumsum()
        walk[rng.integers(0, n, 30)] += 500  # single-row spikes
        noise = rng.normal(size=n)

        keep = thin_positions(ts, walk, noise, bins=bins)

        assert keep is not None
        assert len(keep) <= 6 * bins  # first, last and two extremes per series
        full = pd.DataFrame({"bin": _bins(ts, bins), "walk": walk, "noise": noise})
        thin = full.iloc[keep]
        for column in ("walk", "noise"):
            pd.testing.assert_frame_equal(
                thin.groupby("bin")[column].agg(["min", "max", "first", "last"]),
                full.groupby("bin")[column].agg(["min", "max", "first", "last"]),
            )

    def test_keeps_the_edges_of_a_gap(self) -> None:
        n = 10_000
        ts = _times(n)
        values = np.arange(n, dtype=float)
        values[4_000:4_500] = np.nan
        keep = thin_positions(ts, values, bins=10)
        assert keep is not None
        assert {3_999, 4_000, 4_499, 4_500} <= set(keep)

    def test_a_short_series_is_not_thinned(self) -> None:
        ts = _times(400)
        assert thin_positions(ts, np.ones(400), bins=100) is None
        assert thin_positions(_times(5_000), np.ones(5_000), bins=None) is None

    def test_times_out_of_order_are_not_thinned(self) -> None:
        ts = _times(5_000).iloc[::-1].reset_index(drop=True)
        assert thin_positions(ts, np.ones(5_000), bins=100) is None


class TestThinPayload:
    def test_series_stay_aligned_and_say_so(self) -> None:
        n = 20_000
        ts = _times(n)
        a = np.sin(np.arange(n) / 50.0)
        payload = {
            "timestamp": ts,
            "a": a,
            "b": pd.Series(np.arange(n, dtype=float)),
            "y_range": (0.0, 1.0),
            "levels": 5,
        }
        out = thin_payload(payload, 100)
        thinned = out["thinned"]
        assert isinstance(thinned, Thinned)
        assert (thinned.total, thinned.bins) == (n, 100)
        assert len(out["timestamp"]) == len(out["a"]) == len(out["b"])
        assert len(out["a"]) == thinned.shown < n
        # b is the row number, so it says which rows a kept.
        np.testing.assert_array_equal(out["a"], a[out["b"].to_numpy(dtype=int)])
        assert out["y_range"] == (0.0, 1.0) and out["levels"] == 5
        assert f"{n:,}" in thinned.note

    def test_nothing_thinned_is_none(self) -> None:
        payload = {"timestamp": _times(10), "a": np.zeros(10)}
        assert thin_payload(payload, 100)["thinned"] is None
        assert thin_payload(payload, None)["thinned"] is None


class TestThinPoints:
    def test_keeps_the_largest_event_on_each_spot(self) -> None:
        n = 5_000
        df = pd.DataFrame(
            {
                "timestamp": _times(n),
                "price": np.tile([100.0, 101.0], n // 2),
                "volume": np.arange(n, dtype=float),
                "direction": "bid",
            }
        )
        out, thinned = thin_points(
            df, y="price", by=["direction"], size="volume", bins=10, rows=2
        )
        assert thinned is not None and thinned.points
        assert len(out) == thinned.shown == 20  # 10 columns x 2 rows
        cols = _bins(df["timestamp"], 10)
        expected = df.groupby([cols, df["price"]])["volume"].max().sort_values()
        np.testing.assert_array_equal(out["volume"].sort_values(), expected)

    def test_a_repeated_index_label_keeps_one_point(self) -> None:
        n = 2_000
        df = pd.DataFrame(
            {
                "timestamp": _times(n).to_numpy(),
                "price": 100.0,
                "volume": np.arange(n, dtype=float),
            },
            index=np.zeros(n, dtype=int),  # every row has label 0
        )
        out, thinned = thin_points(df, y="price", by=[], size="volume", bins=10)
        assert thinned is not None
        assert len(out) == thinned.shown == 10

    def test_any_x_column_and_a_log_y_axis(self) -> None:
        # Volumes 1, 2, ..., 9 and 10, 20, ..., 90 on two log rows: a linear
        # cut would put all of 1..9 into the first of two rows with 10..40.
        size = np.r_[np.arange(1.0, 10.0), np.arange(10.0, 100.0, 10.0)]
        n = 40 * len(size)
        df = pd.DataFrame(
            {
                "distance": np.repeat(np.linspace(-1.0, 1.0, 40), len(size)),
                "size": np.tile(size, 40),
            }
        )
        out, thinned = thin_points(
            df, x="distance", y="size", by=[], size="size", bins=4, rows=2, log_y=True
        )
        assert thinned is not None and thinned.total == n
        cols = _cells(df["distance"].to_numpy(), 4)
        rows = _cells(np.log10(df["size"].to_numpy()), 2)
        expected = df.groupby([cols, rows])["size"].max().sort_values()
        np.testing.assert_array_equal(out["size"].sort_values(), expected)
        assert sorted(out["size"].unique()) == [9.0, 90.0]


class TestThinOrderLines:
    def test_each_order_keeps_its_extremes_in_every_bin(self) -> None:
        n, bins = 20_000, 50
        rng = np.random.default_rng(2)
        df = pd.DataFrame(
            {
                "timestamp": _times(n),
                "id": rng.integers(0, 40, n),
                "rank": rng.integers(1, 30, n),
            }
        )
        out, thinned = thin_order_lines(df, y="rank", bins=bins)
        assert thinned is not None and thinned.orders
        assert len(out) == thinned.shown < n
        assert out.index.is_monotonic_increasing  # still in time order
        cols = pd.Series(_bins(df["timestamp"], bins), index=df.index)
        _assert_same_extremes(df, out, cols)

    def test_a_point_drawn_twice_keeps_one_row(self) -> None:
        # Each order has two rows on the same spot of the screen, as most
        # short-lived orders at the touch do.
        n = 4_000
        df = pd.DataFrame(
            {
                "timestamp": np.repeat(_times(n // 2).to_numpy(), 2),
                "id": np.arange(n) // 2,
                "rank": 3,
            }
        )
        out, thinned = thin_order_lines(df, y="rank", bins=10)
        assert thinned is not None
        assert len(out) == n // 2
        np.testing.assert_array_equal(out.index, np.arange(1, n, 2))  # the later

    def test_a_short_frame_is_not_thinned(self) -> None:
        df = pd.DataFrame({"timestamp": _times(50), "id": 1, "rank": 1})
        assert thin_order_lines(df, y="rank", bins=100) == (df, None)
        assert thin_order_lines(df, y="rank", bins=None) == (df, None)


def _cells(values: np.ndarray, count: int) -> np.ndarray:
    lo, hi = values.min(), values.max()
    return np.minimum(((values - lo) / (hi - lo) * count).astype(int), count - 1)


def _assert_same_extremes(full: pd.DataFrame, thin: pd.DataFrame, cols) -> None:
    """*thin* has each order's lowest, highest and last rank in every bin."""
    for frame in (full, thin):
        assert frame.index.is_unique
    pd.testing.assert_frame_equal(
        thin.groupby(["id", cols.loc[thin.index]])["rank"].agg(["min", "max"]),
        full.groupby(["id", cols])["rank"].agg(["min", "max"]),
    )
    # The last row of each bin, where the line leaves it: the row itself.
    pd.testing.assert_frame_equal(
        thin.groupby(["id", cols.loc[thin.index]]).tail(1).sort_index(),
        full.groupby(["id", cols]).tail(1).sort_index(),
    )


class TestFacesKeepTheLargestPoints:
    """Each face keeps the largest point, or each order's extremes, in every cell."""

    def test_order_activity_l2(self, bundled_sample_result) -> None:
        depth = bundled_sample_result.depth
        full = prepare_order_activity_l2_data(depth, time_bins=None)
        thin = prepare_order_activity_l2_data(depth)
        assert full["thinned"] is None
        assert isinstance(thin["thinned"], Thinned) and thin["thinned"].points
        e = full["events"]
        assert thin["thinned"].total == len(e) > len(thin["events"])
        cols = _cells(pd.DatetimeIndex(e["timestamp"]).asi8.astype(float), 1000)
        rows = _cells(e["price"].to_numpy(dtype=float), 300)
        expected = e.groupby(["action", "direction", cols, rows])["volume"].max()
        np.testing.assert_array_equal(
            np.sort(thin["events"]["volume"].to_numpy(dtype=float)),
            np.sort(expected.to_numpy(dtype=float)),
        )

    def test_order_outcome(self, bundled_sample_result) -> None:
        events = bundled_sample_result.events
        full = prepare_order_outcome_data(events, bins=None)
        thin = prepare_order_outcome_data(events)
        assert full["thinned"] is None
        assert isinstance(thin["thinned"], Thinned) and thin["thinned"].points
        outcomes = ("filled", "partial", "cancelled")
        f = pd.concat([full[k] for k in outcomes])
        t = pd.concat([thin[k] for k in outcomes])
        assert thin["thinned"].total == len(f) > len(t)
        cols = _cells(f["distance_bps"].to_numpy(dtype=float), 1000)
        rows = _cells(f["placed"].to_numpy(dtype=float), 300)
        expected = f.groupby(["outcome", cols, rows])["placed"].max()
        np.testing.assert_array_equal(
            np.sort(t["placed"].to_numpy(dtype=float)),
            np.sort(expected.to_numpy(dtype=float)),
        )
        # The marker sizes still scale against every order, not the kept ones.
        pd.testing.assert_series_equal(t["marker_area"], f.loc[t.index, "marker_area"])

    def test_queue_position(self, bundled_sample_result) -> None:
        events = bundled_sample_result.events
        full = prepare_queue_position_data(events, time_bins=None)
        thin = prepare_queue_position_data(events)
        assert full["thinned"] is None
        assert isinstance(thin["thinned"], Thinned) and thin["thinned"].orders
        assert (thin["max_rank"], thin["show_markers"]) == (
            full["max_rank"],
            full["show_markers"],
        )
        # The kept rows keep their index labels, so they line up with the full
        # rows.  One order has one fate, so its rows stay in time order.
        fates = ("filled", "cancelled", "resting")
        f = pd.concat([full[k] for k in fates])
        t = pd.concat([thin[k] for k in fates])
        assert thin["thinned"].total == len(f) > len(t)
        cols = pd.Series(
            _cells(pd.DatetimeIndex(f["timestamp"]).asi8.astype(float), 1000),
            index=f.index,
        )
        _assert_same_extremes(f, t, cols)


class TestDepthRaster:
    def test_a_cell_holds_the_largest_volume_in_it(self) -> None:
        t = pd.Timestamp("2026-01-01", tz="UTC")
        s = pd.Timedelta(seconds=1)
        depth = pd.DataFrame(
            {
                # Level 100 holds 1, spikes to 9 for one second, then holds 2.
                # Level 101 holds 5 for the whole span.
                "timestamp": [t, t + 10 * s, t + 11 * s, t + 100 * s, t, t + 100 * s],
                "price": [100.0, 100.0, 100.0, 100.0, 101.0, 101.0],
                "volume": [1.0, 9.0, 2.0, 2.0, 5.0, 5.0],
            }
        )
        raster = depth_raster(depth, width=10)
        assert raster is not None
        grid = raster["volume"]
        assert grid.shape == (2, 10)
        assert (raster["price0"], raster["price_step"]) == (100.0, 1.0)
        assert (raster["t0"], raster["t1"]) == (t, t + 100 * s)
        # The one-second spike sits in column 1 and is not lost.
        np.testing.assert_array_equal(grid[0], [1, 9, 2, 2, 2, 2, 2, 2, 2, 2])
        np.testing.assert_array_equal(grid[1], [5] * 10)

    def test_rows_are_capped(self) -> None:
        t = pd.Timestamp("2026-01-01", tz="UTC")
        prices = np.arange(5_000, dtype=float)
        depth = pd.DataFrame(
            {
                "timestamp": [t] * 5_000 + [t + pd.Timedelta(seconds=1)] * 5_000,
                "price": np.r_[prices, prices],
                "volume": 1.0,
            }
        )
        raster = depth_raster(depth, width=10, max_rows=100)
        assert raster is not None
        assert raster["volume"].shape[0] <= 100
        assert not np.isnan(raster["volume"]).any()


# The largest page each face may write for the bundled 30-minute sample.  The
# faces with thinned lines stay well under the default.  The L2 event map
# keeps one change per spot of the screen and has its own, larger budget
# (19 MiB before).  The cancellations map (6.7 MiB before), the order outcome
# cloud (5.3 MiB) and the queue lines (3.4 MiB) were thinned in #533.
_DEFAULT_BUDGET_MIB = 2.5
_PAGE_BUDGET_MIB = {
    "order_activity.L2": 4.0,
}


@pytest.fixture(scope="module")
def gallery(bundled_sample_result, tmp_path_factory) -> Path:
    """The Plotly gallery of the bundled sample, built once."""
    pytest.importorskip("plotly")
    from ob_analytics.visualization.gallery import generate_gallery

    out = tmp_path_factory.mktemp("gallery")
    return generate_gallery(bundled_sample_result, out, backends=["plotly"])


class TestGalleryPageSizes:
    def test_every_page_is_within_its_budget(self, gallery: Path) -> None:
        pages = sorted((gallery.parent / "plotly").glob("*.html"))
        assert pages
        over = {}
        for page in pages:
            budget = _PAGE_BUDGET_MIB.get(page.stem, _DEFAULT_BUDGET_MIB)
            mib = page.stat().st_size / 2**20
            if mib > budget:
                over[page.stem] = f"{mib:.1f} MiB > {budget} MiB"
        assert not over

    def test_cards_say_when_a_face_shows_thinned_data(self, gallery: Path) -> None:
        html = gallery.read_text(encoding="utf-8")
        # book signals, price view, liquidity, L1 lines, heatmap quote line
        # and the event map
        assert html.count("time steps keeps its first, last, lowest") >= 5
        assert "it shows the largest one." in html
        assert "each order keeps its lowest, highest and last rank." in html
