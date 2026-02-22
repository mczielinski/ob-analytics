"""Tests for set_order_types — order classification logic."""

import pandas as pd

from ob_analytics.order_types import set_order_types


def _events(*rows):
    """Build events DataFrame from (event_id, id, action, direction, price, volume) tuples."""
    return pd.DataFrame(
        rows, columns=["event_id", "id", "action", "direction", "price", "volume"],
    )


def _trades(*rows):
    """Build trades DataFrame from (maker_event_id, taker_event_id) tuples."""
    return pd.DataFrame(rows, columns=["maker_event_id", "taker_event_id"])


class TestSetOrderTypes:
    """Unit tests for all order type classification branches."""

    def test_pure_market_order(self):
        """Taker-only, never maker → 'market'."""
        events = _events(
            (1, 10, "created", "bid", 100, 5),
            (2, 10, "deleted", "bid", 100, 0),
        )
        trades = _trades((99, 1))  # event 1 is a taker
        result = set_order_types(events, trades)
        assert (result[result["id"] == 10]["type"] == "market").all()

    def test_resting_limit_forever(self):
        """Created, never changed, never deleted → 'resting-limit'."""
        events = _events(
            (1, 10, "created", "bid", 100, 5),
        )
        trades = _trades()  # no trades at all
        result = set_order_types(events, trades)
        assert (result[result["id"] == 10]["type"] == "resting-limit").all()

    def test_resting_limit_maker_only(self):
        """Maker-only, never taker → 'resting-limit'."""
        events = _events(
            (1, 10, "created", "bid", 100, 5),
            (2, 10, "deleted", "bid", 100, 0),
        )
        trades = _trades((1, 99))  # event 1 is a maker
        result = set_order_types(events, trades)
        assert (result[result["id"] == 10]["type"] == "resting-limit").all()

    def test_flashed_limit(self):
        """Created + deleted with identical volume, no fills → 'flashed-limit'."""
        events = _events(
            (1, 10, "created", "bid", 100, 5),
            (2, 10, "deleted", "bid", 100, 5),  # same volume
        )
        trades = _trades()  # no trades
        result = set_order_types(events, trades)
        assert (result[result["id"] == 10]["type"] == "flashed-limit").all()

    def test_pacman_order(self):
        """Order with multiple distinct prices → 'pacman'."""
        events = _events(
            (1, 10, "created", "bid", 100, 5),
            (2, 10, "changed", "bid", 101, 4),  # price changed!
            (3, 10, "deleted", "bid", 101, 0),
        )
        trades = _trades()
        result = set_order_types(events, trades)
        assert (result[result["id"] == 10]["type"] == "pacman").all()

    def test_market_limit_order(self):
        """Both maker and taker → 'market-limit'."""
        events = _events(
            (1, 10, "created", "bid", 100, 5),
            (2, 10, "deleted", "bid", 100, 0),
        )
        trades = _trades(
            (1, 99),   # event 1 is a maker
            (88, 1),   # event 1 is also a taker
        )
        result = set_order_types(events, trades)
        assert (result[result["id"] == 10]["type"] == "market-limit").all()

    def test_unknown_order_remains(self):
        """Order that doesn't fit patterns stays 'unknown'."""
        events = _events(
            (1, 10, "created", "bid", 100, 5),
            (2, 10, "deleted", "bid", 100, 3),  # different volume → not flashed
        )
        trades = _trades()  # no trades → not market/maker/taker
        result = set_order_types(events, trades)
        # It's created+deleted with volume mismatch, no changed, not forever → unknown
        assert (result[result["id"] == 10]["type"] == "unknown").all()
