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
# faces with thinned lines stay well under the default.  The point clouds
# have their own, larger budgets: the L2 event map keeps one event per spot of
# the screen (19 MiB before), and the others are not thinned yet.
_DEFAULT_BUDGET_MIB = 2.5
_PAGE_BUDGET_MIB = {
    "order_activity.L2": 4.0,
    "cancellations.L2": 8.0,
    "order_outcome.L3": 6.0,
    "queue_position.L3": 4.0,
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
