"""A short Databento MBO stream shared by the tests that need one.

This module is a helper, not a test module: its name has no ``test_`` prefix
so pytest does not collect it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ob_analytics.databento import DBN_PRICE_DIVISOR

#: The stream's first instant, in nanoseconds since the epoch.
T0 = 1_700_000_000_000_000_000


def px(decimal: float) -> int:
    """Return *decimal* as Databento's fixed-point integer price."""
    return round(decimal * DBN_PRICE_DIVISOR)


#: A short MBO stream of ``(ms, order_id, action, side, price, size)``: two
#: bids and two asks, a partial cancel, a buyer who takes one ask and part of
#: the next, then two modifies, one of which moves a bid.  A trade's ``T`` and
#: ``F`` share the instant of the book record that follows them, as Databento
#: sends them.
MBO = [
    (0, 1, "A", "B", px(100.25), 100),
    (1, 2, "A", "B", px(100.20), 70),
    (2, 3, "A", "A", px(100.35), 50),
    (3, 4, "A", "A", px(100.40), 80),
    (4, 1, "C", "B", px(100.25), 40),
    (5, 0, "T", "B", px(100.35), 50),
    (5, 3, "F", "A", px(100.35), 50),
    (5, 3, "C", "A", px(100.35), 50),
    (6, 0, "T", "B", px(100.40), 30),
    (6, 4, "F", "A", px(100.40), 30),
    (6, 4, "M", "A", px(100.40), 50),
    (7, 2, "M", "B", px(100.30), 70),
]


def mbo_frame() -> pd.DataFrame:
    """:data:`MBO` as the frame ``DBNStore.to_df(price_type="fixed")`` gives."""
    n = len(MBO)
    ns = np.array([r[0] for r in MBO], dtype="int64") * 1_000_000 + T0
    return pd.DataFrame(
        {
            "ts_recv": pd.to_datetime(ns + 500, utc=True),
            "ts_event": pd.to_datetime(ns, utc=True),
            "action": [r[2] for r in MBO],
            "side": [r[3] for r in MBO],
            "price": [r[4] for r in MBO],
            "size": [r[5] for r in MBO],
            "order_id": [r[1] for r in MBO],
            "flags": np.zeros(n, dtype="uint8"),
            "sequence": np.arange(1, n + 1, dtype="uint32"),
            "instrument_id": np.full(n, 1108, dtype="uint32"),
            "publisher_id": np.full(n, 2, dtype="uint16"),
        }
    )
