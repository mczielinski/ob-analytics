"""Count taker orders, not trades, toward the card floor (issue #532).

On an L3 feed one taker order that fills many resting orders makes many
trades.  The order flow imbalance and VPIN tables add ``n_takers`` when the
trades name their takers, and the cards apply ``MIN_TRADES`` to it.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from ob_analytics import bulk_volume_classification, order_flow_imbalance, vpin
from ob_analytics.visualization import prepare
from ob_analytics.visualization._data import MIN_TRADES

BASE = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")


def _tape(takers: list, seconds: list[float], directions: list[str]) -> pd.DataFrame:
    """Unit-size trades at 100.0; *takers* is the taker id per trade."""
    n = len(takers)
    return pd.DataFrame(
        {
            "timestamp": [BASE + pd.Timedelta(seconds=s) for s in seconds],
            "price": np.linspace(100.0, 101.0, n),
            "volume": [1.0] * n,
            "direction": directions,
            "maker": [f"m{i}" for i in range(n)],
            "taker": pd.array(takers, dtype="object"),
        }
    )


def _l3_tape() -> pd.DataFrame:
    """Two one-minute windows of 12 trades each.

    In the first, one buy order fills 12 resting orders.  In the second,
    12 separate takers trade, buys and sells mixed.
    """
    sweep = _tape(["big"] * 12, [1.0 + 0.01 * i for i in range(12)], ["buy"] * 12)
    many = _tape(
        [f"t{i}" for i in range(12)],
        [61.0 + i for i in range(12)],
        ["buy", "sell"] * 6,
    )
    return pd.concat([sweep, many], ignore_index=True)


def _l2_tape() -> pd.DataFrame:
    """The same trades with no maker or taker columns, like a bare L2 tape."""
    return _l3_tape().drop(columns=["maker", "taker"])


# ── Order flow imbalance ─────────────────────────────────────────────


class TestImbalanceCountsTakers:
    def test_one_taker_filling_many_makers_counts_once(self) -> None:
        ofi = order_flow_imbalance(_l3_tape(), window="1min")
        assert list(ofi["n_trades"]) == [12, 12]
        assert list(ofi["n_takers"]) == [1, 12]
        assert list(ofi.columns).index("n_takers") == (
            list(ofi.columns).index("n_trades") + 1
        )

    def test_a_trade_with_no_taker_id_counts_as_its_own(self) -> None:
        tape = _l3_tape()
        tape.loc[[0, 1, 2], "taker"] = pd.NA
        ofi = order_flow_imbalance(tape, window="1min")
        # Three unknown takers plus "big".
        assert list(ofi["n_takers"]) == [4, 12]

    def test_an_l2_tape_has_no_taker_count(self) -> None:
        ofi = order_flow_imbalance(_l2_tape(), window="1min")
        assert "n_takers" not in ofi.columns
        assert list(ofi["n_trades"]) == [12, 12]

    def test_a_taker_column_with_no_ids_is_ignored(self) -> None:
        tape = _l3_tape()
        tape["taker"] = pd.NA
        assert "n_takers" not in order_flow_imbalance(tape, window="1min").columns

    def test_a_repeated_index_does_not_change_the_count(self) -> None:
        tape = _l3_tape()
        tape.index = [0] * len(tape)
        assert list(order_flow_imbalance(tape, window="1min")["n_takers"]) == [1, 12]

    def test_the_card_leaves_out_the_window_with_one_taker(self) -> None:
        payload = prepare.order_flow_imbalance(
            order_flow_imbalance(_l3_tape(), window="1min")
        )
        assert list(payload["ofi_df"]["n_takers"]) == [12]
        assert payload["notice"] == (
            f"1 of 2 windows left out: fewer than {MIN_TRADES} taker orders each"
        )

    def test_the_card_counts_trades_on_an_l2_tape(self) -> None:
        payload = prepare.order_flow_imbalance(
            order_flow_imbalance(_l2_tape(), window="1min")
        )
        assert len(payload["ofi_df"]) == 2
        assert payload["notice"] == ""

    def test_too_few_names_taker_orders(self) -> None:
        tape = _l3_tape().iloc[:12]
        payload = prepare.order_flow_imbalance(order_flow_imbalance(tape))
        assert payload["too_few"].startswith("Too few taker orders")
        assert f"No window holds {MIN_TRADES} taker orders" in payload["too_few"]

    def test_the_sample_card_draws_only_windows_with_enough_takers(
        self, bundled_sample_result
    ) -> None:
        trades = bundled_sample_result.trades
        ofi = order_flow_imbalance(trades)
        # The sample is L3: some windows pass on trades but not on takers.
        assert ((ofi["n_trades"] >= MIN_TRADES) & (ofi["n_takers"] < MIN_TRADES)).any()
        payload = prepare.order_flow_imbalance(ofi)
        kept = payload["ofi_df"]
        assert not kept.empty
        assert (kept["n_takers"] >= MIN_TRADES).all()
        assert "taker orders each" in payload["notice"]


# ── VPIN and bulk volume classification ──────────────────────────────


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestVpinCountsTakers:
    @pytest.mark.parametrize("sign_method", [None, "bvc"])
    def test_buckets_count_takers(self, sign_method) -> None:
        got = vpin(_l3_tape(), bucket_volume=6.0, sign_method=sign_method)
        assert list(got["n_trades"]) == [6, 6, 6, 6]
        assert list(got["n_takers"]) == [1, 1, 6, 6]

    @pytest.mark.parametrize("sign_method", [None, "bvc"])
    def test_an_l2_tape_has_no_taker_count(self, sign_method) -> None:
        got = vpin(_l2_tape(), bucket_volume=6.0, sign_method=sign_method)
        assert "n_takers" not in got.columns

    def test_a_split_trade_counts_in_both_buckets(self) -> None:
        tape = _tape(["a", "b", "c"], [1.0, 2.0, 3.0], ["buy"] * 3)
        tape["volume"] = [1.0, 2.0, 1.0]
        # Trade "b" fills the end of bucket 0 and the start of bucket 1.
        got = vpin(tape, bucket_volume=2.0)
        assert list(got["n_takers"]) == [2, 2]

    @pytest.mark.parametrize("sign_method", [None, "bvc"])
    def test_no_full_bucket_still_has_the_column(self, sign_method) -> None:
        got = vpin(_l3_tape(), bucket_volume=100.0, sign_method=sign_method)
        assert got.empty
        assert got["n_takers"].dtype == np.dtype("int64")
        assert list(got.columns).index("n_takers") == (
            list(got.columns).index("n_trades") + 1
        )

    def test_bulk_volume_classification_counts_takers(self) -> None:
        got = bulk_volume_classification(_l3_tape(), bucket_volume=6.0)
        assert list(got["n_takers"]) == [1, 1, 6, 6]
        assert (
            "n_takers"
            not in bulk_volume_classification(_l2_tape(), bucket_volume=6.0).columns
        )

    def test_the_card_names_taker_orders(self) -> None:
        payload = prepare.vpin(vpin(_l3_tape(), bucket_volume=6.0))
        # The median bucket holds 3.5 takers but 6 trades.
        assert payload["too_few"] == (
            "Too few taker orders to draw.\nA bucket holds about 3.5 taker "
            f"orders; the chart needs at least {MIN_TRADES}."
        )

    def test_the_card_floor_reads_takers_over_trades(self) -> None:
        got = vpin(_l3_tape(), bucket_volume=6.0)
        assert prepare.vpin(got, min_trades=4)["too_few"].startswith("Too few taker")
        got = got.drop(columns="n_takers")
        assert prepare.vpin(got, min_trades=4)["too_few"] == ""

    def test_one_taker_per_bucket_reads_in_the_singular(self) -> None:
        tape = _tape(["a"] * 4, [1.0, 2.0, 3.0, 4.0], ["buy"] * 4)
        payload = prepare.vpin(vpin(tape, bucket_volume=2.0))
        assert "A bucket holds about 1 taker order;" in payload["too_few"]
