"""Tests for flow_toxicity.py — VPIN, Kyle's Lambda, and OFI."""

import warnings

import numpy as np
import pandas as pd
import pytest

from ob_analytics.exceptions import ConfigError, ObAnalyticsError
from ob_analytics.flow_toxicity import (
    KYLE_MIN_T_STAT,
    KYLE_MIN_WINDOWS,
    KyleLambdaResult,
    compute_kyle_lambda,
    compute_vpin,
    order_flow_imbalance,
    vpin_bucket_volume,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _trades(directions, volumes=None, prices=None, base_sec_offsets=None):
    """Build a minimal trades DataFrame."""
    n = len(directions)
    base = pd.Timestamp("2015-05-01 00:00:00")
    if base_sec_offsets is None:
        base_sec_offsets = list(range(n))
    if volumes is None:
        volumes = [1.0] * n
    if prices is None:
        prices = [100.0] * n
    return pd.DataFrame(
        {
            "timestamp": [base + pd.Timedelta(seconds=s) for s in base_sec_offsets],
            "price": prices,
            "volume": volumes,
            "direction": directions,
        }
    )


# ── VPIN ─────────────────────────────────────────────────────────────


class TestComputeVpin:
    def test_uniform_buys_vpin_one(self):
        """All-buy trades → VPIN = 1.0 for every bucket."""
        trades = _trades(["buy"] * 10, volumes=[1.0] * 10)
        result = compute_vpin(trades, bucket_volume=2.0)
        assert len(result) == 5
        assert all(abs(v - 1.0) < 1e-10 for v in result["vpin"])

    def test_balanced_flow_vpin_zero(self):
        """Alternating buy/sell with equal volume → VPIN ≈ 0."""
        directions = ["buy", "sell"] * 5
        trades = _trades(directions, volumes=[2.0] * 10)
        result = compute_vpin(trades, bucket_volume=4.0)
        # Each bucket gets one buy (2.0) and one sell (2.0) → perfectly balanced
        assert all(abs(v) < 1e-10 for v in result["vpin"])

    def test_bucket_boundaries(self):
        """A large trade split across two buckets is handled correctly."""
        # One 3-unit trade, bucket_volume=2.0 → first bucket full (2.0),
        # second gets 1.0 (incomplete, not in output)
        trades = _trades(["buy"], volumes=[3.0])
        result = compute_vpin(trades, bucket_volume=2.0)
        assert len(result) == 1  # only one complete bucket
        assert abs(result.iloc[0]["vpin"] - 1.0) < 1e-10

    def test_output_columns(self):
        """Returns the expected column set."""
        trades = _trades(["buy"] * 4, volumes=[1.0] * 4)
        result = compute_vpin(trades, bucket_volume=2.0)
        expected = {
            "bucket",
            "timestamp_start",
            "timestamp_end",
            "buy_volume",
            "sell_volume",
            "vpin",
            "vpin_avg",
        }
        assert set(result.columns) == expected

    def test_vpin_avg_is_rolling_mean(self):
        """vpin_avg is a rolling mean over n_buckets."""
        trades = _trades(
            ["buy"] * 6 + ["sell"] * 6,
            volumes=[1.0] * 12,
        )
        result = compute_vpin(trades, bucket_volume=2.0, n_buckets=3)
        # First 3 buckets are all-buy (vpin=1), next 3 are all-sell (vpin=1)
        # Rolling mean with window=3 and min_periods=1
        assert len(result) == 6
        assert abs(result.iloc[0]["vpin_avg"] - 1.0) < 1e-10

    def test_empty_trades_raises(self):
        """Empty DataFrame raises ObAnalyticsError."""
        empty = pd.DataFrame(columns=["timestamp", "price", "volume", "direction"])
        with pytest.raises(ObAnalyticsError):
            compute_vpin(empty, bucket_volume=1.0)

    @pytest.mark.parametrize("sign_method", [None, "bvc"])
    def test_no_full_bucket_returns_typed_empty_frame(self, sign_method):
        """Too little volume for one bucket gives zero rows, standard dtypes."""
        trades = _trades(["buy", "sell", "buy"], prices=[100.0, 101.0, 100.5])
        trades["timestamp"] = trades["timestamp"].dt.tz_localize("UTC")
        result = compute_vpin(trades, bucket_volume=100.0, sign_method=sign_method)
        assert result.empty
        assert result.dtypes.to_dict() == {
            "bucket": np.dtype("int64"),
            "timestamp_start": trades["timestamp"].dtype,
            "timestamp_end": trades["timestamp"].dtype,
            "buy_volume": np.dtype("float64"),
            "sell_volume": np.dtype("float64"),
            "vpin": np.dtype("float64"),
            "vpin_avg": np.dtype("float64"),
        }

    def test_missing_columns_raises(self):
        """Missing required columns raises ConfigError."""
        bad = pd.DataFrame({"timestamp": [1], "price": [100]})
        with pytest.raises(ConfigError):
            compute_vpin(bad, bucket_volume=1.0)

    def test_negative_bucket_volume_raises(self):
        """Non-positive bucket_volume raises ValueError."""
        trades = _trades(["buy"])
        with pytest.raises(ValueError, match="positive"):
            compute_vpin(trades, bucket_volume=-1.0)


# ── Kyle's Lambda ────────────────────────────────────────────────────


class TestComputeKyleLambda:
    def test_positive_impact(self):
        """Buys push price up → positive lambda."""
        # Window 1: all buys, price goes 100 → 105
        # Window 2: all sells, price goes 105 → 100
        trades = _trades(
            ["buy", "buy", "sell", "sell"],
            volumes=[10.0, 10.0, 10.0, 10.0],
            prices=[100.0, 105.0, 105.0, 100.0],
            base_sec_offsets=[0, 60, 300, 360],
        )
        result = compute_kyle_lambda(trades, window="5min")
        assert isinstance(result, KyleLambdaResult)
        assert result.lambda_ > 0
        assert result.n_windows == 2

    def test_single_window_returns_nan(self):
        """Only 1 data point → can't run OLS → NaN."""
        trades = _trades(["buy"], volumes=[1.0], prices=[100.0])
        result = compute_kyle_lambda(trades, window="5min")
        assert np.isnan(result.lambda_)
        assert result.n_windows == 1

    def test_lambda_matches_independent_lstsq(self):
        """λ must equal an independent lstsq fit of its own regression_df.

        Guards the OLS implementation against drift: the slope reported by
        ``compute_kyle_lambda`` is compared (rtol=1e-10) to a fresh
        ``np.linalg.lstsq`` fit of the same (signed_volume, delta_price)
        points. Run against the normal-equations code it passes too — which
        is precisely why swapping in ``lstsq`` is behaviour-preserving.
        """
        rng = np.random.default_rng(0)
        directions: list[str] = []
        volumes: list[float] = []
        prices: list[float] = []
        offsets: list[int] = []
        # 12 distinct 5-min windows, two trades each so delta_price != 0 and
        # signed_volume varies across windows (non-degenerate design matrix).
        for w in range(12):
            base = w * 300
            p0 = 100.0 + float(rng.normal(0, 5))
            p1 = p0 + float(rng.normal(0, 2))
            directions += ["buy", "sell"]
            volumes += [float(rng.uniform(1, 10)), float(rng.uniform(1, 10))]
            prices += [p0, p1]
            offsets += [base, base + 60]
        trades = _trades(
            directions, volumes=volumes, prices=prices, base_sec_offsets=offsets
        )
        result = compute_kyle_lambda(trades, window="5min")

        reg = result.regression_df
        x = reg["signed_volume"].to_numpy(dtype=float)
        y = reg["delta_price"].to_numpy(dtype=float)
        assert len(reg) >= 2
        assert np.ptp(x) > 0  # design matrix has full rank
        X = np.column_stack([np.ones(len(x)), x])
        beta_ref = np.linalg.lstsq(X, y, rcond=None)[0]
        assert np.isclose(result.lambda_, beta_ref[1], rtol=1e-10)

    def test_result_fields(self):
        """KyleLambdaResult has all expected attributes."""
        trades = _trades(
            ["buy", "sell"],
            volumes=[1.0, 1.0],
            prices=[100.0, 99.0],
            base_sec_offsets=[0, 300],
        )
        result = compute_kyle_lambda(trades, window="5min")
        assert hasattr(result, "lambda_")
        assert hasattr(result, "t_stat")
        assert hasattr(result, "r_squared")
        assert hasattr(result, "n_windows")
        assert hasattr(result, "regression_df")
        assert isinstance(result.regression_df, pd.DataFrame)

    def test_regression_df_columns(self):
        """Regression DataFrame has expected columns."""
        trades = _trades(
            ["buy", "sell"],
            volumes=[1.0, 1.0],
            prices=[100.0, 99.0],
            base_sec_offsets=[0, 300],
        )
        result = compute_kyle_lambda(trades, window="5min")
        assert set(result.regression_df.columns) == {
            "timestamp",
            "delta_price",
            "signed_volume",
        }

    def test_empty_raises(self):
        """Empty DataFrame raises ObAnalyticsError."""
        empty = pd.DataFrame(columns=["timestamp", "price", "volume", "direction"])
        with pytest.raises(ObAnalyticsError):
            compute_kyle_lambda(empty)

    def test_missing_columns_raises(self):
        """Missing columns raises ConfigError."""
        bad = pd.DataFrame({"timestamp": [1]})
        with pytest.raises(ConfigError):
            compute_kyle_lambda(bad)


# ── Robustness diagnostics (#119) ────────────────────────────────────


def _busy_tape(n_windows: int = 200, seed: int = 1) -> pd.DataFrame:
    """A tape with many 1-minute windows and a real λ = 0.5 plus noise.

    Each window holds two trades: the first sets the window's opening price,
    the second closes it at ``open + 0.5 * signed_volume + noise``.  Both
    trades share one side, so the window's signed volume is their summed size.
    """
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-01-05 00:00:00")
    rows = []
    price = 1000.0
    for w in range(n_windows):
        side = "buy" if rng.random() < 0.5 else "sell"
        sizes = rng.uniform(1.0, 5.0, size=2)
        signed = sizes.sum() * (1.0 if side == "buy" else -1.0)
        close = price + 0.5 * signed + rng.normal(0.0, 1.0)
        t0 = base + pd.Timedelta(minutes=w)
        rows.append((t0, price, sizes[0], side))
        rows.append((t0 + pd.Timedelta(seconds=30), close, sizes[1], side))
        price = close
    return pd.DataFrame(rows, columns=["timestamp", "price", "volume", "direction"])


def _thin_tape() -> pd.DataFrame:
    """Five 1-minute windows: a fit exists, but it cannot mean much."""
    return _busy_tape(n_windows=5, seed=3)


class TestKyleLambdaDiagnostics:
    def test_thin_tape_is_flagged(self):
        result = compute_kyle_lambda(_thin_tape(), window="1min")
        assert result.n_windows == 5
        assert not result.significant
        assert any("windows" in d for d in result.diagnostics)

    def test_busy_tape_is_not_flagged(self):
        result = compute_kyle_lambda(_busy_tape(), window="1min")
        assert result.n_windows == 200
        assert abs(result.t_stat) > KYLE_MIN_T_STAT
        assert result.significant
        assert result.diagnostics == ()

    def test_low_t_stat_is_flagged(self):
        """Enough windows but no relationship: flagged on |t| alone."""
        trades = _busy_tape()
        rng = np.random.default_rng(9)
        # Shuffle closes against flow so the slope carries no signal.
        closes = trades["price"].to_numpy().copy()
        closes[1::2] = closes[0::2] + rng.normal(0.0, 1.0, size=len(closes) // 2)
        trades["price"] = closes
        result = compute_kyle_lambda(trades, window="1min")
        assert result.n_windows >= KYLE_MIN_WINDOWS
        assert abs(result.t_stat) < KYLE_MIN_T_STAT
        assert not result.significant
        assert len(result.diagnostics) == 1
        assert "|t|" in result.diagnostics[0]

    def test_undefined_fit_is_flagged(self):
        result = compute_kyle_lambda(_trades(["buy"]), window="5min")
        assert np.isnan(result.lambda_)
        assert not result.significant
        assert any("undefined" in d for d in result.diagnostics)

    def test_hand_built_result_reports_diagnostics(self):
        """The flag is derived from the fields, so a hand-built result has it."""
        weak = KyleLambdaResult(lambda_=1.0, t_stat=1.2, r_squared=0.1, n_windows=5)
        assert not weak.significant
        strong = KyleLambdaResult(lambda_=1.0, t_stat=8.0, r_squared=0.6, n_windows=100)
        assert strong.significant


class TestKyleLambdaBootstrap:
    def test_ci_contains_estimate_on_busy_tape(self):
        result = compute_kyle_lambda(_busy_tape(), window="1min")
        assert result.ci_low < result.lambda_ < result.ci_high
        # The true slope is 0.5; the interval should sit around it.
        assert result.ci_low < 0.5 < result.ci_high

    def test_ci_is_deterministic_under_a_seed(self):
        a = compute_kyle_lambda(_busy_tape(), window="1min", seed=42)
        b = compute_kyle_lambda(_busy_tape(), window="1min", seed=42)
        c = compute_kyle_lambda(_busy_tape(), window="1min", seed=43)
        assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)
        assert (a.ci_low, a.ci_high) != (c.ci_low, c.ci_high)

    def test_accepts_a_generator(self):
        a = compute_kyle_lambda(
            _busy_tape(), window="1min", seed=np.random.default_rng(5)
        )
        b = compute_kyle_lambda(
            _busy_tape(), window="1min", seed=np.random.default_rng(5)
        )
        assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)

    def test_wider_level_gives_wider_interval(self):
        narrow = compute_kyle_lambda(_busy_tape(), window="1min", ci_level=0.5)
        wide = compute_kyle_lambda(_busy_tape(), window="1min", ci_level=0.99)
        assert wide.ci_low < narrow.ci_low
        assert wide.ci_high > narrow.ci_high

    def test_bootstrap_can_be_turned_off(self):
        result = compute_kyle_lambda(_busy_tape(), window="1min", n_boot=0)
        assert np.isnan(result.ci_low) and np.isnan(result.ci_high)
        assert np.isfinite(result.lambda_)

    def test_undefined_fit_has_no_interval(self):
        result = compute_kyle_lambda(_trades(["buy"]), window="5min")
        assert np.isnan(result.ci_low) and np.isnan(result.ci_high)

    def test_bad_ci_level_raises(self):
        with pytest.raises(ValueError, match="ci_level"):
            compute_kyle_lambda(_busy_tape(), window="1min", ci_level=1.5)


class TestVpinBucketVolume:
    def test_hand_built_one_day(self):
        """Exactly one day of trading: average daily volume is the total."""
        trades = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-01-05 00:00", "2026-01-05 12:00", "2026-01-06 00:00"]
                ),
                "price": [100.0, 100.0, 100.0],
                "volume": [100.0, 200.0, 200.0],
            }
        )
        assert vpin_bucket_volume(trades) == pytest.approx(500.0 / 50)

    def test_short_session_scales_to_a_day(self):
        """Six hours of trading at 60 units: 240 units a day, /50 = 4.8."""
        trades = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-01-05 00:00", "2026-01-05 06:00"]),
                "price": [100.0, 100.0],
                "volume": [30.0, 30.0],
            }
        )
        assert vpin_bucket_volume(trades) == pytest.approx(4.8)

    def test_trading_day_and_buckets_per_day(self):
        """A 6-hour session on a 6-hour trading day is one full day."""
        trades = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-01-05 09:00", "2026-01-05 15:00"]),
                "price": [100.0, 100.0],
                "volume": [30.0, 30.0],
            }
        )
        got = vpin_bucket_volume(trades, buckets_per_day=10, trading_day="6h")
        assert got == pytest.approx(6.0)

    def test_zero_span_raises(self):
        with pytest.raises(ValueError, match="span"):
            vpin_bucket_volume(_trades(["buy", "sell"], base_sec_offsets=[0, 0]))


class TestVpinDiagnostics:
    def test_explicit_bucket_is_recorded(self):
        vpin = compute_vpin(_trades(["buy"] * 10), bucket_volume=2.0, n_buckets=3)
        assert vpin.attrs["bucket_volume"] == 2.0
        assert vpin.attrs["bucket_volume_rule"] == "given"
        assert vpin.attrs["n_buckets"] == 3
        assert vpin.attrs["diagnostics"] == ()

    def test_too_few_buckets_is_flagged(self):
        vpin = compute_vpin(_trades(["buy"] * 10), bucket_volume=2.0)
        assert len(vpin) == 5
        (message,) = vpin.attrs["diagnostics"]
        assert "5 complete buckets" in message

    def test_too_few_buckets_warns_with_given_bucket_volume(self):
        with pytest.warns(UserWarning, match="5 complete buckets"):
            vpin = compute_vpin(_trades(["buy"] * 10), bucket_volume=2.0)
        assert len(vpin) == 5

    def test_too_few_buckets_warns_to_shrink_default_bucket_volume(self):
        trades = _busy_tape()
        with pytest.warns(UserWarning, match="pass a smaller bucket_volume"):
            compute_vpin(trades, n_buckets=10**9)

    def test_full_window_raises_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            vpin = compute_vpin(_trades(["buy"] * 10), bucket_volume=2.0, n_buckets=3)
        assert vpin.attrs["diagnostics"] == ()

    def test_bvc_path_is_recorded_and_flagged(self):
        vpin = compute_vpin(
            _trades(["buy"] * 10, prices=[100.0 + i for i in range(10)]),
            bucket_volume=2.0,
            sign_method="bvc",
        )
        assert vpin.attrs["bucket_volume"] == 2.0
        assert vpin.attrs["diagnostics"]

    def test_no_complete_bucket_keeps_the_columns(self):
        """A short tape under the default rule fills no bucket (#119)."""
        for sign_method in (None, "bvc"):
            trades = _trades(["buy", "sell"] * 3, prices=[100.0, 101.0] * 3)
            vpin = compute_vpin(trades, sign_method=sign_method)
            assert vpin.empty
            assert "vpin_avg" in vpin.columns
            assert vpin.attrs["diagnostics"]

    def test_bucket_volume_defaults_to_adv_rule(self):
        trades = _busy_tape()
        vpin = compute_vpin(trades)
        assert vpin.attrs["bucket_volume"] == pytest.approx(vpin_bucket_volume(trades))
        assert vpin.attrs["bucket_volume_rule"] == "adv/50"

    def test_busy_tape_is_not_flagged(self):
        trades = _busy_tape()
        vpin = compute_vpin(trades, bucket_volume=trades["volume"].sum() / 120)
        assert len(vpin) >= 50
        assert vpin.attrs["diagnostics"] == ()


@pytest.fixture(scope="module")
def sample_trades(sample_csv_path) -> pd.DataFrame:
    from ob_analytics import Pipeline

    return Pipeline().run(sample_csv_path).trades


class TestBundledSampleDiagnostics:
    def test_kyle_lambda_not_significant(self, sample_trades):
        trades = sample_trades
        result = compute_kyle_lambda(trades, window="5min")
        assert not result.significant
        assert len(result.diagnostics) == 2  # too few windows, and |t| < 2
        assert result.ci_low < result.lambda_ < result.ci_high

    def test_vpin_reports_bucket_and_diagnostic(self, sample_trades):
        trades = sample_trades
        vpin = compute_vpin(trades)
        assert vpin.attrs["bucket_volume"] == pytest.approx(vpin_bucket_volume(trades))
        assert vpin.attrs["diagnostics"]


# ── Order Flow Imbalance ─────────────────────────────────────────────


class TestOrderFlowImbalance:
    def test_output_columns(self):
        """Returns expected columns."""
        trades = _trades(["buy", "sell"], volumes=[1.0, 1.0])
        result = order_flow_imbalance(trades, window="1min")
        expected = {"timestamp", "buy_volume", "sell_volume", "net_volume", "ofi"}
        assert set(result.columns) == expected

    def test_all_buys_ofi_one(self):
        """All buys → OFI = 1.0."""
        trades = _trades(["buy"] * 5, volumes=[1.0] * 5)
        result = order_flow_imbalance(trades, window="1min")
        assert all(abs(v - 1.0) < 1e-10 for v in result["ofi"])

    def test_all_sells_ofi_negative_one(self):
        """All sells → OFI = −1.0."""
        trades = _trades(["sell"] * 5, volumes=[1.0] * 5)
        result = order_flow_imbalance(trades, window="1min")
        assert all(abs(v - (-1.0)) < 1e-10 for v in result["ofi"])

    def test_balanced_ofi_zero(self):
        """Equal buy and sell volume in the same window → OFI = 0."""
        trades = _trades(
            ["buy", "sell"],
            volumes=[5.0, 5.0],
            base_sec_offsets=[0, 1],
        )
        result = order_flow_imbalance(trades, window="1min")
        assert all(abs(v) < 1e-10 for v in result["ofi"])

    def test_empty_raises(self):
        """Empty DataFrame raises ObAnalyticsError."""
        empty = pd.DataFrame(columns=["timestamp", "volume", "direction"])
        with pytest.raises(ObAnalyticsError):
            order_flow_imbalance(empty)

    def test_missing_columns_raises(self):
        """Missing columns raises ConfigError."""
        bad = pd.DataFrame({"timestamp": [1]})
        with pytest.raises(ConfigError):
            order_flow_imbalance(bad)


# ── Visualization ────────────────────────────────────────────────────

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure

from ob_analytics.visualization import _data, plot


class TestFlowToxicityPlots:
    def test_plot_vpin_returns_figure(self):
        """plot_vpin returns a Figure."""
        trades = _trades(["buy"] * 10, volumes=[1.0] * 10)
        vpin_df = compute_vpin(trades, bucket_volume=2.0)
        fig = plot("vpin", **_data.prepare_vpin_data(vpin_df))
        assert isinstance(fig, Figure)

    @pytest.mark.parametrize(
        "vpin_df",
        [
            # What compute_vpin returns when the trades fill no bucket.
            compute_vpin(_trades(["buy"]), bucket_volume=100.0),
            # An untyped empty frame: every column is object dtype.
            pd.DataFrame(columns=["timestamp_end", "vpin", "vpin_avg"]),
        ],
        ids=["typed", "object"],
    )
    def test_plot_vpin_empty_draws_no_buckets(self, vpin_df):
        """Zero buckets draws an empty panel instead of raising."""
        ax = plot("vpin", **_data.prepare_vpin_data(vpin_df)).axes[0]
        # The theme places titles on the left; read every slot.
        title = " ".join(ax.get_title(loc=s) for s in ("left", "center", "right"))
        assert "no complete buckets" in title
        assert not ax.patches and not ax.lines

    def test_plot_ofi_returns_figure(self):
        """plot_order_flow_imbalance returns a Figure."""
        trades = _trades(
            ["buy", "sell"] * 5,
            volumes=[1.0] * 10,
        )
        ofi_df = order_flow_imbalance(trades, window="1min")
        fig = plot("order_flow_imbalance", **_data.prepare_ofi_data(ofi_df))
        assert isinstance(fig, Figure)

    def test_plot_ofi_with_price_overlay(self):
        """plot_order_flow_imbalance with trades overlay returns a Figure."""
        trades = _trades(
            ["buy", "sell"] * 5,
            volumes=[1.0] * 10,
            prices=[100.0 + i for i in range(10)],
        )
        ofi_df = order_flow_imbalance(trades, window="1min")
        fig = plot("order_flow_imbalance", **_data.prepare_ofi_data(ofi_df, trades))
        assert isinstance(fig, Figure)

    def test_plot_kyle_lambda_returns_figure(self):
        """plot_kyle_lambda returns a Figure."""
        trades = _trades(
            ["buy", "buy", "sell", "sell"],
            volumes=[10.0, 10.0, 10.0, 10.0],
            prices=[100.0, 105.0, 105.0, 100.0],
            base_sec_offsets=[0, 60, 300, 360],
        )
        result = compute_kyle_lambda(trades, window="5min")
        fig = plot("kyle_lambda", **_data.prepare_kyle_lambda_data(result))
        assert isinstance(fig, Figure)
