"""Cards that a thin tape or a single window made unreadable (issue #497).

Book signals takes its price range from readable quotes only.  Roll and
Amihud show a single window as a number.  VPIN and order flow imbalance
draw only buckets or windows with enough trades, and say so otherwise.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.text import Text

from ob_analytics.flow_toxicity import VpinResult
from ob_analytics.metrics import get_metric
from ob_analytics.visualization import plot, prepare
from ob_analytics.visualization._data import MIN_TRADES
from ob_analytics.visualization.gallery import _metric_payload
from tests._drawn import drawn


def _card(result, name: str, **settings) -> dict:
    """The payload the gallery card *name* draws from *result*."""
    return _metric_payload(get_metric(name), result, **settings)


def _mpl_text(fig) -> str:
    """Every piece of text on a matplotlib figure, joined."""
    return " ".join(t.get_text() for t in fig.findobj(Text))


def _plotly_text(fig) -> str:
    """The title and the annotations of a plotly figure, joined."""
    parts = [fig.layout.title.text or ""]
    parts += [a.text for a in fig.layout.annotations]
    return " ".join(parts)


# ── Book signals: the range of readable quotes ───────────────────────


def _summary_with_empty_side() -> pd.DataFrame:
    """Three quotes: an empty bid side (price 0), then two good ones."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 00:00:01", "2020-01-01 00:00:02"]
            ),
            "best_bid_price": [0.0, 100.0, 100.5],
            "best_bid_vol": [0.0, 8.0, 5.0],
            "best_ask_price": [101.0, 101.0, 101.5],
            "best_ask_vol": [4.0, 2.0, 5.0],
            "bid_vol25bps": [0.0, 12.0, 5.0],
            "ask_vol25bps": [4.0, 4.0, 5.0],
        }
    )


class TestBookSignalsRange:
    def test_the_range_leaves_out_an_empty_side(self) -> None:
        payload = prepare.book_signals(_summary_with_empty_side())
        assert payload["y_range"] == (100.0, 101.5)

    def test_the_spread_has_no_edge_where_a_side_is_empty(self) -> None:
        payload = prepare.book_signals(_summary_with_empty_side())
        np.testing.assert_allclose(
            payload["best_bid_price"], [np.nan, 100.0, 100.5], equal_nan=True
        )
        np.testing.assert_allclose(
            payload["best_ask_price"], [np.nan, 101.0, 101.5], equal_nan=True
        )

    @pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
    def test_the_sample_mid_and_micro_price_sit_inside_the_axis(
        self, bundled_sample_result, backend: str
    ) -> None:
        if backend == "plotly":
            pytest.importorskip("plotly")
        payload = _card(bundled_sample_result, "book_signals")
        low, high = payload["y_range"]
        for key in ("mid", "microprice"):
            values = np.asarray(payload[key], dtype=float)
            values = values[np.isfinite(values)]
            assert low <= values.min() and values.max() <= high
        # An empty side would pull the axis to 0; the sample's prices are ~78k.
        assert low > 0.5 * high
        fig = plot("book_signals", backend=backend, **payload)
        if backend == "matplotlib":
            axis_low, axis_high = fig.axes[0].get_ylim()
        else:
            axis_low, axis_high = fig.layout.yaxis.range
        assert axis_low > 0.5 * high and axis_high >= high


# ── Roll and Amihud: one window is a number ──────────────────────────


def _amihud(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=len(values), freq="5min"),
            "amihud": values,
        }
    )


class TestOneWindowIsANumber:
    def test_one_window_gives_a_headline(self) -> None:
        payload = prepare.amihud(_amihud([3.358874e-10]))
        assert payload["headline"] == "3.359e-10"
        assert payload["notice"] == "One window holds every trade"

    def test_one_window_with_no_value_says_so(self) -> None:
        payload = prepare.amihud(_amihud([np.nan]))
        assert payload["headline"] == "No value"

    def test_several_windows_draw_a_line(self) -> None:
        payload = prepare.amihud(_amihud([1.0, np.nan, 3.0]))
        assert payload["headline"] == ""
        assert payload["notice"] == "3 windows, 1 with no value"
        assert drawn(plot("amihud", backend="matplotlib", **payload))["line"] > 0

    def test_matplotlib_draws_the_number_and_no_time_axis(self) -> None:
        fig = plot("amihud", backend="matplotlib", **prepare.amihud(_amihud([2.5])))
        assert not drawn(fig)
        assert not fig.axes[0].axison
        assert "2.5" in _mpl_text(fig)

    def test_plotly_draws_the_number_and_no_time_axis(self) -> None:
        pytest.importorskip("plotly")
        fig = plot("amihud", backend="plotly", **prepare.amihud(_amihud([2.5])))
        assert not drawn(fig)
        assert fig.layout.xaxis.visible is False
        assert "2.5" in _plotly_text(fig)

    def test_the_sample_roll_and_amihud_cards_show_a_number(
        self, bundled_sample_result
    ) -> None:
        # Both measure the whole session by default: one window each.
        roll = _card(bundled_sample_result, "roll_spread")
        assert roll["headline"] == "No value"
        assert "median autocorrelation" in roll["notice"]
        amihud = _card(bundled_sample_result, "amihud")
        assert amihud["headline"] not in ("", "No value")
        float(amihud["headline"])


# ── VPIN: enough trades per bucket ───────────────────────────────────


def _vpin(n_trades: list[int]) -> VpinResult:
    n = len(n_trades)
    table = pd.DataFrame(
        {
            "timestamp_end": pd.date_range("2020-01-01", periods=n, freq="1min"),
            "n_trades": n_trades,
            "vpin": np.linspace(0.2, 0.6, n),
            "vpin_avg": np.linspace(0.2, 0.4, n),
        }
    )
    return VpinResult(
        table=table, bucket_volume=1.0, bucket_volume_rule="given", n_buckets=n
    )


class TestVpinNeedsEnoughTrades:
    def test_few_trades_per_bucket_give_a_message(self) -> None:
        payload = prepare.vpin(_vpin([3, 2, 4, 3]))
        assert payload["too_few"].startswith("Too few trades")

    def test_enough_trades_per_bucket_draw(self) -> None:
        payload = prepare.vpin(_vpin([MIN_TRADES] * 4))
        assert payload["too_few"] == ""
        assert drawn(plot("vpin", backend="matplotlib", **payload))["rectangle"] > 0

    def test_the_floor_can_be_lowered(self) -> None:
        assert prepare.vpin(_vpin([3, 2, 4, 3]), min_trades=2)["too_few"] == ""

    @pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
    def test_the_face_writes_the_message_in_place_of_the_chart(
        self, backend: str
    ) -> None:
        if backend == "plotly":
            pytest.importorskip("plotly")
        fig = plot("vpin", backend=backend, **prepare.vpin(_vpin([3, 2, 4, 3])))
        assert not drawn(fig)
        text = _mpl_text(fig) if backend == "matplotlib" else _plotly_text(fig)
        assert "Too few trades" in text

    def test_the_sample_vpin_card_says_too_few_taker_orders(
        self, bundled_sample_result
    ) -> None:
        # The gallery fits ~100 buckets to the sample's 284 trades.  The
        # sample is L3, so the floor counts taker orders (#532).
        payload = _card(bundled_sample_result, "vpin", bucket_volume="fit")
        assert payload["vpin_df"]["n_takers"].median() < MIN_TRADES
        assert payload["too_few"].startswith("Too few taker orders")


# ── Order flow imbalance: enough trades per window ───────────────────


def _ofi(n_trades: list[int]) -> pd.DataFrame:
    n = len(n_trades)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="1min"),
            "n_trades": n_trades,
            "ofi": np.linspace(-0.5, 0.5, n),
        }
    )


class TestImbalanceNeedsEnoughTrades:
    def test_thin_windows_are_left_out_and_counted(self) -> None:
        payload = prepare.order_flow_imbalance(_ofi([2, MIN_TRADES, 30, 1]))
        assert list(payload["ofi_df"]["n_trades"]) == [MIN_TRADES, 30]
        assert payload["notice"] == (
            f"2 of 4 windows left out: fewer than {MIN_TRADES} trades each"
        )
        assert payload["too_few"] == ""
        fig = plot("order_flow_imbalance", backend="matplotlib", **payload)
        assert drawn(fig)["rectangle"] == 2

    @pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
    def test_a_bar_stays_as_wide_as_its_window(self, backend: str) -> None:
        # The kept windows are three minutes apart; each is one minute long.
        if backend == "plotly":
            pytest.importorskip("plotly")
        payload = prepare.order_flow_imbalance(_ofi([30, 1, 1, 30]))
        assert payload["bar_gap"] == pd.Timedelta("1min")
        fig = plot("order_flow_imbalance", backend=backend, **payload)
        if backend == "matplotlib":
            widths = [p.get_width() for p in fig.axes[0].patches]
            assert widths == pytest.approx([0.8 / (24 * 60)] * 2)
        else:
            assert fig.data[0].width == pytest.approx(48_000)

    def test_no_window_with_enough_trades_gives_a_message(self) -> None:
        payload = prepare.order_flow_imbalance(_ofi([2, 3, 1]))
        assert payload["ofi_df"].empty
        assert payload["too_few"].startswith("Too few trades")

    @pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
    def test_the_face_writes_the_message_in_place_of_the_bars(
        self, backend: str
    ) -> None:
        if backend == "plotly":
            pytest.importorskip("plotly")
        payload = prepare.order_flow_imbalance(_ofi([2, 3, 1]))
        fig = plot("order_flow_imbalance", backend=backend, **payload)
        assert not drawn(fig)
        text = _mpl_text(fig) if backend == "matplotlib" else _plotly_text(fig)
        assert "Too few trades" in text

    def test_the_sample_card_draws_only_full_windows(
        self, bundled_sample_result
    ) -> None:
        payload = _card(bundled_sample_result, "order_flow_imbalance")
        kept = payload["ofi_df"]
        assert not kept.empty
        assert (kept["n_takers"] >= MIN_TRADES).all()
        assert "windows left out" in payload["notice"]
