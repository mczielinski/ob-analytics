"""Regression tests for BitstampLoader.fill computation.

The legacy code zeroed the fill whenever the order's reported price
changed between consecutive events for the same id.  On modern Bitstamp
that pattern is the *normal* shape of a taker fill (created at limit
price, deleted at the matching engine's fill price), so the guard
silently lost ~38% of all taker fills.
"""

from __future__ import annotations

import pandas as pd
import pytest

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
                    dict(
                        id=1,
                        timestamp=100,
                        exchange_timestamp=100,
                        price=79107.0,
                        volume=0.5,
                        action="created",
                        direction="bid",
                    ),
                    dict(
                        id=1,
                        timestamp=101,
                        exchange_timestamp=101,
                        price=78323.0,
                        volume=0.0,
                        action="deleted",
                        direction="bid",
                    ),
                ]
            )
        )
        events = BitstampLoader().load(path)
        assert events.loc[events["action"] == "deleted", "fill"].iloc[0] == 0.5

    def test_aggressor_walking_levels_records_every_fill(self, tmp_path):
        path = tmp_path / "orders.csv"
        rows = [
            dict(
                id=1,
                timestamp=100,
                exchange_timestamp=100,
                price=80000.0,
                volume=1.0,
                action="created",
                direction="bid",
            ),
            dict(
                id=1,
                timestamp=100,
                exchange_timestamp=100,
                price=79000.0,
                volume=0.7,
                action="changed",
                direction="bid",
            ),
            dict(
                id=1,
                timestamp=100,
                exchange_timestamp=100,
                price=79000.0,
                volume=0.5,
                action="changed",
                direction="bid",
            ),
            dict(
                id=1,
                timestamp=100,
                exchange_timestamp=100,
                price=79100.0,
                volume=0.1,
                action="changed",
                direction="bid",
            ),
            dict(
                id=1,
                timestamp=101,
                exchange_timestamp=101,
                price=79200.0,
                volume=0.0,
                action="deleted",
                direction="bid",
            ),
        ]
        path.write_text(_csv(rows))
        events = BitstampLoader().load(path)
        assert events["fill"].sum() == pytest.approx(1.0)
        non_zero_fills = (events["fill"] > 0).sum()
        assert non_zero_fills == 4
