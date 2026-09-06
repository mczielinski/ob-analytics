"""Regression tests for BitstampLoader.fill computation.

The legacy code zeroed the fill whenever the order's reported price
changed between consecutive events for the same id.  On modern Bitstamp
that pattern is the *normal* shape of a taker fill (created at limit
price, deleted at the matching engine's fill price), so the guard
silently lost ~38% of all taker fills.
"""

from __future__ import annotations

import pandas as pd

from ob_analytics.bitstamp import BitstampLoader


def _csv(rows: list[dict]) -> str:
    cols = [
        "id",
        "timestamp",
        "exchange_timestamp",
        "price",
        "volume",
        "action",
        "direction",
    ]
    return pd.DataFrame(rows, columns=cols).to_csv(index=False)


class TestFillOnPriceChange:
    def test_taker_create_then_delete_records_fill(self, tmp_path):
        path = tmp_path / "orders.csv"
        path.write_text(
            _csv(
                [
                    {
                        "id": 1,
                        "timestamp": 100,
                        "exchange_timestamp": 100,
                        "price": 79107.0,
                        "volume": 0.5,
                        "action": "created",
                        "direction": "bid",
                    },
                    {
                        "id": 1,
                        "timestamp": 101,
                        "exchange_timestamp": 101,
                        "price": 78323.0,
                        "volume": 0.0,
                        "action": "deleted",
                        "direction": "bid",
                    },
                ]
            )
        )
        events = BitstampLoader().load(path)
        # The CSV is in the venue's base-asset floats; the loaded frame is in
        # integer lots on the default 1e-8 grid (issue #226), so 0.5 base
        # asset is 50,000,000 lots.
        assert events.loc[events["action"] == "deleted", "fill"].iloc[0] == 50_000_000

    def test_aggressor_walking_levels_records_every_fill(self, tmp_path):
        path = tmp_path / "orders.csv"
        rows = [
            {
                "id": 1,
                "timestamp": 100,
                "exchange_timestamp": 100,
                "price": 80000.0,
                "volume": 1.0,
                "action": "created",
                "direction": "bid",
            },
            {
                "id": 1,
                "timestamp": 100,
                "exchange_timestamp": 100,
                "price": 79000.0,
                "volume": 0.7,
                "action": "changed",
                "direction": "bid",
            },
            {
                "id": 1,
                "timestamp": 100,
                "exchange_timestamp": 100,
                "price": 79000.0,
                "volume": 0.5,
                "action": "changed",
                "direction": "bid",
            },
            {
                "id": 1,
                "timestamp": 100,
                "exchange_timestamp": 100,
                "price": 79100.0,
                "volume": 0.1,
                "action": "changed",
                "direction": "bid",
            },
            {
                "id": 1,
                "timestamp": 101,
                "exchange_timestamp": 101,
                "price": 79200.0,
                "volume": 0.0,
                "action": "deleted",
                "direction": "bid",
            },
        ]
        path.write_text(_csv(rows))
        events = BitstampLoader().load(path)
        # 1.0 base asset of fills, in integer lots on the default 1e-8 grid.
        # Exact rather than approximate: that is the point of storing lots
        # (issue #226) — the four partial fills sum back to the whole.
        assert events["fill"].sum() == 100_000_000
        non_zero_fills = (events["fill"] > 0).sum()
        assert non_zero_fills == 4
