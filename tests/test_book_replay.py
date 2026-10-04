"""Tests for the book replay (issue #117): its prepare step and Plotly faces.

The replay steps through the book at one instant per interval: an L2 face built
from the depth table and an L3 face rebuilt from the events, beside a still
trades panel.  Skipped entirely if plotly is not installed.
"""

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

go = pytest.importorskip("plotly.graph_objects", reason="plotly not installed")

from ob_analytics.exceptions import ConfigError
from ob_analytics.visualization import (
    Level,
    display_result,
    plot,
    plot_result,
    prepare,
)
from ob_analytics.visualization._plotly import BOOK_REPLAY_SCRIPT
from tests._logging import warnings_logged


@pytest.fixture(scope="module")
def tiny_result(tiny_bitstamp_orders_csv):
    from ob_analytics.bitstamp import BitstampSource
    from ob_analytics.pipeline import Pipeline

    return Pipeline(source=BitstampSource()).run(str(tiny_bitstamp_orders_csv))


@pytest.fixture(scope="module")
def sample_result(bitstamp_sample_dir):
    """Full pipeline run on the bundled Bitstamp sample (loaded once)."""
    from ob_analytics import Pipeline

    return Pipeline().run(bitstamp_sample_dir / "orders.csv.gz")


def _replay(result, per_order=False, **kwargs):
    return prepare.book_replay(
        depth=result.depth,
        events=result.events,
        trades=result.trades,
        per_order=per_order,
        **kwargs,
    )


def _touch(snapshot) -> tuple[float, float]:
    bids, asks = snapshot["bids"], snapshot["asks"]
    return (
        bids["price"].max() if len(bids) else float("nan"),
        asks["price"].min() if len(asks) else float("nan"),
    )


class TestAcceptance:
    """The issue's acceptance check, on the bundled Bitstamp sample."""

    def test_l2_touch_matches_depth_summary_at_every_frame(self, sample_result):
        summary = sample_result.depth_summary
        start = summary["timestamp"].min() + pd.Timedelta("10min")
        data = _replay(
            sample_result, start_time=start, end_time=start + pd.Timedelta("2min")
        )
        assert len(data["snapshots"]) == 121
        for snapshot in data["snapshots"]:
            row = summary[summary["timestamp"] <= snapshot["timestamp"]].iloc[-1]
            best_bid, best_ask = _touch(snapshot)
            assert best_bid == row["best_bid_price"]
            assert best_ask == row["best_ask_price"]
            bids, asks = snapshot["bids"], snapshot["asks"]
            scale = data["volume_scale"]
            touch_bid = bids.loc[bids["price"] == best_bid, "volume"].sum() / scale
            touch_ask = asks.loc[asks["price"] == best_ask, "volume"].sum() / scale
            assert touch_bid == pytest.approx(row["best_bid_vol"])
            assert touch_ask == pytest.approx(row["best_ask_vol"])

    def test_l3_face_is_uncrossed_by_default(self, sample_result):
        start = sample_result.events["timestamp"].min() + pd.Timedelta("10min")
        window = {
            "start_time": start,
            "end_time": start + pd.Timedelta("1min"),
            "interval": "10s",
        }
        shown = _replay(sample_result, per_order=True, **window)
        assert all(bid < ask for bid, ask in map(_touch, shown["snapshots"]))
        # The faithful book of this diff feed holds stale crossed orders.
        faithful = _replay(sample_result, per_order=True, uncross=False, **window)
        assert any(bid >= ask for bid, ask in map(_touch, faithful["snapshots"]))


class TestPrepare:
    def test_frames_are_one_interval_apart(self, tiny_result):
        data = _replay(tiny_result, interval="5s")
        times = pd.DatetimeIndex([f["timestamp"] for f in data["snapshots"]])
        assert (times[1:] - times[:-1] == pd.Timedelta("5s")).all()
        assert data["interval"] == pd.Timedelta("5s")
        assert data["speeds"] == (1, 5, 20)

    def test_interval_widens_to_fit_max_frames(self, tiny_result):
        with warnings_logged() as messages:
            data = _replay(tiny_result, interval="1s", max_frames=4)
        assert len(data["snapshots"]) <= 4
        assert data["interval"] > pd.Timedelta("1s")
        assert any("max_frames=4" in m for m in messages)

    def test_every_bar_sits_in_the_shared_price_range(self, tiny_result):
        for per_order in (False, True):
            data = _replay(tiny_result, per_order=per_order, interval="2s")
            low, high = data["price_range"]
            for snapshot in data["snapshots"]:
                for side in (snapshot["bids"], snapshot["asks"]):
                    assert side["price"].between(low, high).all()

    def test_trades_cover_the_replay(self, tiny_result):
        data = _replay(tiny_result, interval="2s")
        first, last = (
            data["snapshots"][0]["timestamp"],
            data["snapshots"][-1]["timestamp"],
        )
        assert data["trades"]["timestamp"].between(first, last).all()

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"interval": "0s"}, "interval must be positive"),
            ({"max_frames": 0}, "at least 1"),
            ({"price_levels": 0}, "at least 1"),
            ({"speeds": ()}, "positive speeds"),
            ({"speeds": (1, 0)}, "positive speeds"),
        ],
    )
    def test_bad_settings_are_refused(self, tiny_result, kwargs, match):
        with pytest.raises(ConfigError, match=match):
            _replay(tiny_result, **kwargs)

    def test_end_before_start_is_refused(self, tiny_result):
        t = tiny_result.depth["timestamp"]
        with pytest.raises(ConfigError, match="before it starts"):
            _replay(tiny_result, start_time=t.max(), end_time=t.min())

    @pytest.mark.parametrize(
        ("per_order", "missing"), [(False, "depth"), (True, "events")]
    )
    def test_missing_table_is_refused(self, per_order, missing):
        with pytest.raises(ConfigError, match=f"needs the {missing} table"):
            prepare.book_replay(per_order=per_order)


@pytest.fixture(scope="module")
def faces(tiny_result):
    """The L2 and L3 replay figures, in display units, with their payloads."""
    shown = display_result(tiny_result)
    out = {}
    for level, per_order in ((Level.L2, False), (Level.L3, True)):
        data = _replay(shown, per_order=per_order, interval="2s")
        out[level] = (data, plot("book_replay", level, backend="plotly", **data))
    return out


@pytest.mark.parametrize("level", [Level.L2, Level.L3])
class TestFigure:
    def test_one_frame_per_snapshot_changing_four_traces(self, faces, level):
        data, fig = faces[level]
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == len(data["snapshots"])
        for frame in fig.frames:
            assert [t.name for t in frame.data] == ["Bid", "Ask", "Mid", "Book time"]
            # No frame moves an axis: the price, size and time axes are fixed.
            assert frame.layout.to_plotly_json() == {}

    def test_bar_count_is_the_same_in_every_frame(self, faces, level):
        # Stepping without a full redraw, Plotly draws a bar a frame adds
        # without its colour; equal counts mean no bar is ever added.
        _, fig = faces[level]
        assert len({len(f.data[0].y) for f in fig.frames}) == 1
        assert len({len(f.data[1].y) for f in fig.frames}) == 1

    def test_trades_name_the_last_frame_at_or_before_them(self, faces, level):
        data, fig = faces[level]
        times = pd.DatetimeIndex([f["timestamp"] for f in data["snapshots"]])
        names = [f.name for f in fig.frames]
        trades = data["trades"]
        for direction, side in (("buy", "ask"), ("sell", "bid")):
            (trace,) = [t for t in fig.data if t.meta == f"replay-trades:{side}"]
            stamps = trades.loc[trades["direction"] == direction, "timestamp"]
            assert len(stamps) == len(trace.customdata)
            for t, (_, frame) in zip(stamps, trace.customdata, strict=True):
                k = times.searchsorted(t, side="right") - 1
                assert frame == names[max(k, 0)]

    def test_one_play_button_per_speed(self, faces, level):
        data, fig = faces[level]
        buttons = fig.layout.updatemenus[0].buttons
        assert [b.label for b in buttons] == ["Play 1×", "Play 5×", "Play 20×", "Pause"]
        interval_ms = data["interval"].total_seconds() * 1000
        for button, speed in zip(buttons, data["speeds"], strict=False):
            assert button.args[1]["frame"]["duration"] == interval_ms / speed

    def test_html_carries_the_click_script(self, faces, level, tmp_path):
        _, fig = faces[level]
        # Plotly fills in {plot_id}, so look for a line the script keeps.
        marker = "var replay = (gd.layout.meta || {}).book_replay;"
        assert marker in BOOK_REPLAY_SCRIPT
        assert marker in fig.to_html()
        html = fig.to_html(post_script="console.log('mine');")
        assert "console.log('mine');" in html and "replay-trades:" in html
        fig.write_html(tmp_path / "replay.html")
        assert "replay-trades:" in (tmp_path / "replay.html").read_text()
        assert "replay-trades:" in fig._repr_mimebundle_()["text/html"]


def test_frame_names_are_unique_below_a_second(tiny_result):
    data = _replay(tiny_result, interval="250ms")
    fig = plot("book_replay", Level.L2, backend="plotly", **data)
    names = [f.name for f in fig.frames]
    assert len(set(names)) == len(names)


def test_plot_result_draws_the_replay_on_plotly_only(tiny_result):
    fig = plot_result(tiny_result, "book_replay", level="L3", backend="plotly")
    assert "replay-trades:" in fig.to_html()
    with pytest.raises(KeyError):
        plot_result(tiny_result, "book_replay", level="L2", backend="matplotlib")


def test_gallery_shows_the_replay_without_warnings(tiny_result, tmp_path):
    from ob_analytics.visualization.gallery import generate_gallery

    with warnings_logged() as messages:
        generate_gallery(
            tiny_result, tmp_path, view="l2", backends=["plotly", "matplotlib"]
        )
    assert not [m for m in messages if "book_replay" in m]
    assert "replay-trades:" in (tmp_path / "plotly" / "book_replay.L2.html").read_text()
