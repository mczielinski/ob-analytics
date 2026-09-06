"""A price level must empty when the order resting on it goes away.

Bitstamp reports a ``deleted`` whose price is not the price the order rested
at for a small share of orders.  ``price_level_volume`` adds an order's volume
at its ``created`` price, so it has to subtract it there too; subtracting at
the price the later row happens to carry leaves the volume on the created
level for the rest of the session, where every consumer reads it back as a
resting level that no order is on.

The per-order rebuild (``engine.book_state``) tracks orders by id and has
never had this problem, so these tests also pin the two rebuilds together.
"""

import numpy as np
import pandas as pd

from ob_analytics.analytics import set_order_types
from ob_analytics.depth import price_level_volume

_BASE = pd.Timestamp("2026-01-05 10:00:00", tz="UTC")


def _events(rows: list[tuple]) -> pd.DataFrame:
    """Build a classified events frame.

    Each row is ``(event_id, id, t_seconds, price, volume, direction, action,
    fill)``.  Prices are integer ticks and volumes integer lots, per the schema.
    """
    ts = pd.Series(
        [_BASE + pd.Timedelta(seconds=r[2]) for r in rows], dtype="datetime64[ns, UTC]"
    )
    events = pd.DataFrame(
        {
            "event_id": np.array([r[0] for r in rows], dtype=np.int64),
            "id": np.array([r[1] for r in rows], dtype=np.int64),
            "timestamp": ts,
            "exchange_timestamp": ts.copy(),
            "price": np.array([r[3] for r in rows], dtype=np.int64),
            "volume": np.array([r[4] for r in rows], dtype=np.int64),
            "direction": pd.Categorical(
                [r[5] for r in rows], categories=["bid", "ask"], ordered=True
            ),
            "action": pd.Categorical(
                [r[6] for r in rows],
                categories=["created", "changed", "deleted"],
                ordered=True,
            ),
            "fill": np.array([r[7] for r in rows], dtype=np.int64),
        }
    )
    return set_order_types(
        events,
        pd.DataFrame(
            {
                "maker_event_id": pd.array([], dtype=object),
                "taker_event_id": pd.array([], dtype=object),
            }
        ),
    )


def _final_volume(depth: pd.DataFrame, price: int, direction: str) -> float:
    """Volume the level carries after the last row that touched it."""
    rows = depth[(depth["price"] == price) & (depth["direction"] == direction)]
    if rows.empty:
        return 0.0
    return float(rows.sort_values("timestamp", kind="stable")["volume"].iloc[-1])


class TestDeleteAtADifferentPrice:
    """The delete carries a price the order never rested at."""

    def test_created_level_empties(self):
        """The volume leaves the level it was added to, not the reported one."""
        depth = price_level_volume(
            _events(
                [
                    (1, 1, 0.0, 10_000, 500, "ask", "created", 0),
                    (2, 1, 1.0, 10_100, 500, "ask", "deleted", 0),
                ]
            )
        )
        assert _final_volume(depth, 10_000, "ask") == 0.0

    def test_the_reported_level_is_untouched(self):
        """A level the order never rested on keeps the volume it really holds."""
        depth = price_level_volume(
            _events(
                [
                    (1, 1, 0.0, 10_000, 500, "ask", "created", 0),
                    (2, 2, 1.0, 10_100, 700, "ask", "created", 0),
                    (3, 1, 2.0, 10_100, 500, "ask", "deleted", 0),
                ]
            )
        )
        assert _final_volume(depth, 10_000, "ask") == 0.0
        # Order 2 is still resting at 10_100; order 1's delete must not take
        # its volume, and the clip at zero must not hide the overdraw either.
        assert _final_volume(depth, 10_100, "ask") == 700.0

    def test_a_fill_reported_at_another_price(self):
        """An execution reported away from the resting price still reduces it."""
        depth = price_level_volume(
            _events(
                [
                    (1, 1, 0.0, 10_000, 500, "bid", "created", 0),
                    (2, 1, 1.0, 9_900, 300, "bid", "changed", 200),
                    (3, 1, 2.0, 9_900, 300, "bid", "deleted", 0),
                ]
            )
        )
        assert _final_volume(depth, 10_000, "bid") == 0.0


class TestUnaffectedCases:
    """A venue that reports one price per order is unchanged."""

    def test_matching_prices_still_empty_the_level(self):
        depth = price_level_volume(
            _events(
                [
                    (1, 1, 0.0, 10_000, 500, "ask", "created", 0),
                    (2, 1, 1.0, 10_000, 500, "ask", "deleted", 0),
                ]
            )
        )
        assert _final_volume(depth, 10_000, "ask") == 0.0

    def test_a_resting_order_keeps_its_volume(self):
        depth = price_level_volume(
            _events([(1, 1, 0.0, 10_000, 500, "ask", "created", 0)])
        )
        assert _final_volume(depth, 10_000, "ask") == 500.0


class TestAgainstTheBundledSample:
    """No level may hold volume that no order is resting on."""

    def test_no_phantom_levels_on_the_bitstamp_sample(self, bitstamp_sample_dir):
        from ob_analytics.bitstamp import BitstampSource
        from ob_analytics.pipeline import Pipeline

        result = Pipeline(source=BitstampSource()).run(
            str(bitstamp_sample_dir / "orders.csv.gz")
        )
        events, depth = result.events, result.depth

        # Read both books at the last real event, before the capture's
        # end-of-run synthetic deletes clear everything.
        end = events["timestamp"].max()
        levels = (
            depth[depth["timestamp"] < end]
            .sort_values("timestamp", kind="stable")
            .groupby(["direction", "price"], observed=True)["volume"]
            .last()
        )
        levels = levels[levels > 0]

        latest = (
            events[events["timestamp"] < end]
            .sort_values(["timestamp", "event_id"], kind="stable")
            .groupby("id")
            .tail(1)
        )
        resting = latest[
            (latest["action"] != "deleted")
            & (latest["volume"] > 0)
            & (~latest["type"].isin(["market", "market-limit"]))
        ]
        occupied = resting.groupby(["direction", "price"], observed=True)[
            "volume"
        ].sum()

        phantom = levels.index.difference(occupied.index)
        assert len(phantom) == 0, (
            f"{len(phantom)} price level(s) hold volume with no order resting "
            f"on them: {list(phantom)[:5]}"
        )
