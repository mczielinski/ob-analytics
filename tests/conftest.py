"""Shared fixtures for ob-analytics tests."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SAMPLE_CSV = (
    Path(__file__).resolve().parent.parent
    / "ob_analytics"
    / "_sample_data"
    / "orders.csv.gz"
)


@pytest.fixture(scope="module")
def sample_csv_path() -> Path:
    """Path to the bundled sample orders.csv.gz."""
    if not SAMPLE_CSV.exists():
        pytest.skip(f"Sample data not found: {SAMPLE_CSV}")
    return SAMPLE_CSV


@pytest.fixture(scope="session")
def _sample_events_cache() -> pd.DataFrame:
    """Bundled sample loaded once per session (the load costs ~15s)."""
    if not SAMPLE_CSV.exists():
        pytest.skip(f"Sample data not found: {SAMPLE_CSV}")
    from ob_analytics.bitstamp import BitstampLoader

    return BitstampLoader().load(SAMPLE_CSV)


@pytest.fixture
def sample_events(_sample_events_cache: pd.DataFrame) -> pd.DataFrame:
    """Per-test copy of the session-cached sample events.

    The copy (~0.1s) keeps tests isolated from each other's mutations while
    sharing the single expensive load.
    """
    return _sample_events_cache.copy()


@pytest.fixture(scope="session")
def tiny_bitstamp_orders_csv(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Programmatically-built minimal Bitstamp orders.csv (+ trades.csv).

    Hand-crafted to exercise Pipeline.run end-to-end without checking
    real-capture data into the repo. ~12 order events + 4 trades spread
    over 30 seconds — enough for time-windowed metrics (VPIN/OFI/Kyle)
    to produce non-empty DataFrames on tiny bucket/window settings.
    Pipeline.run on this fixture is well under a second.
    """
    d = tmp_path_factory.mktemp("tiny_bitstamp")

    # Build a self-consistent micro orderbook: three resting bids, three
    # resting asks, then a sequence of changed/deleted events implying
    # five trades that hit the resting book.
    base = 1_000  # ms
    orders = pd.DataFrame(
        [
            # Resting book at t=base.
            dict(
                id=1,
                timestamp=base,
                exchange_timestamp=base,
                price=99.0,
                volume=2.0,
                action="created",
                direction="bid",
            ),  # noqa: E501
            dict(
                id=2,
                timestamp=base,
                exchange_timestamp=base,
                price=99.5,
                volume=2.0,
                action="created",
                direction="bid",
            ),  # noqa: E501
            dict(
                id=3,
                timestamp=base,
                exchange_timestamp=base,
                price=99.9,
                volume=2.0,
                action="created",
                direction="bid",
            ),  # noqa: E501
            dict(
                id=10,
                timestamp=base,
                exchange_timestamp=base,
                price=100.1,
                volume=2.0,
                action="created",
                direction="ask",
            ),  # noqa: E501
            dict(
                id=11,
                timestamp=base,
                exchange_timestamp=base,
                price=100.5,
                volume=2.0,
                action="created",
                direction="ask",
            ),  # noqa: E501
            dict(
                id=12,
                timestamp=base,
                exchange_timestamp=base,
                price=101.0,
                volume=2.0,
                action="created",
                direction="ask",
            ),  # noqa: E501
            # Trades over the next 30s: each is one resting order being
            # partially filled (changed) or fully consumed (deleted).
            dict(
                id=10,
                timestamp=base + 5_000,
                exchange_timestamp=base + 5_000,
                price=100.1,
                volume=1.5,
                action="changed",
                direction="ask",
            ),  # noqa: E501
            dict(
                id=10,
                timestamp=base + 10_000,
                exchange_timestamp=base + 10_000,
                price=100.1,
                volume=0.0,
                action="deleted",
                direction="ask",
            ),  # noqa: E501
            dict(
                id=3,
                timestamp=base + 15_000,
                exchange_timestamp=base + 15_000,
                price=99.9,
                volume=1.0,
                action="changed",
                direction="bid",
            ),  # noqa: E501
            dict(
                id=11,
                timestamp=base + 20_000,
                exchange_timestamp=base + 20_000,
                price=100.5,
                volume=1.5,
                action="changed",
                direction="ask",
            ),  # noqa: E501
            dict(
                id=3,
                timestamp=base + 25_000,
                exchange_timestamp=base + 25_000,
                price=99.9,
                volume=0.0,
                action="deleted",
                direction="bid",
            ),  # noqa: E501
            dict(
                id=2,
                timestamp=base + 30_000,
                exchange_timestamp=base + 30_000,
                price=99.5,
                volume=1.5,
                action="changed",
                direction="bid",
            ),  # noqa: E501
        ]
    )
    orders.to_csv(d / "orders.csv", index=False)

    # Trades reference the order IDs above; aggressor side alternates so
    # OFI/Kyle Lambda see meaningful signed flow.
    trades = pd.DataFrame(
        [
            dict(
                trade_id=1,
                timestamp=base + 5_000,
                exchange_timestamp=base + 5_000,
                price=100.1,
                amount=0.5,
                buy_order_id=100,
                sell_order_id=10,
                side="buy",
            ),  # noqa: E501
            dict(
                trade_id=2,
                timestamp=base + 10_000,
                exchange_timestamp=base + 10_000,
                price=100.1,
                amount=1.5,
                buy_order_id=101,
                sell_order_id=10,
                side="buy",
            ),  # noqa: E501
            dict(
                trade_id=3,
                timestamp=base + 15_000,
                exchange_timestamp=base + 15_000,
                price=99.9,
                amount=1.0,
                buy_order_id=3,
                sell_order_id=102,
                side="sell",
            ),  # noqa: E501
            dict(
                trade_id=4,
                timestamp=base + 20_000,
                exchange_timestamp=base + 20_000,
                price=100.5,
                amount=0.5,
                buy_order_id=103,
                sell_order_id=11,
                side="buy",
            ),  # noqa: E501
            dict(
                trade_id=5,
                timestamp=base + 25_000,
                exchange_timestamp=base + 25_000,
                price=99.9,
                amount=1.0,
                buy_order_id=3,
                sell_order_id=104,
                side="sell",
            ),  # noqa: E501
            dict(
                trade_id=6,
                timestamp=base + 30_000,
                exchange_timestamp=base + 30_000,
                price=99.5,
                amount=0.5,
                buy_order_id=2,
                sell_order_id=105,
                side="sell",
            ),  # noqa: E501
        ]
    )
    trades.to_csv(d / "trades.csv", index=False)

    return d / "orders.csv"


@pytest.fixture(scope="module")
def bitstamp_sample_dir() -> Path:
    """Path to the bundled Bitstamp sample directory (orders.csv.gz + trades.csv).

    Use when a test needs the directory itself (e.g. to feed
    ``BitstampTradeReader.load(events, source=...)``) rather than just
    the orders.csv.gz path.
    """
    from ob_analytics import sample_data_dir

    d = sample_data_dir()
    if not (d / "orders.csv.gz").exists() or not (d / "trades.csv").exists():
        pytest.skip(f"Bitstamp sample data not found in: {d}")
    return d


@pytest.fixture
def tiny_events() -> pd.DataFrame:
    """Minimal valid events DataFrame (4 rows, 2 bid + 2 ask fills)."""
    ts = pd.Timestamp("2015-10-10 21:32:00", tz="UTC")
    return pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4],
            "id": [100, 200, 300, 400],
            "timestamp": [
                ts,
                ts + pd.Timedelta(milliseconds=10),
                ts + pd.Timedelta(seconds=10),
                ts + pd.Timedelta(seconds=10, milliseconds=10),
            ],
            "exchange_timestamp": [
                ts - pd.Timedelta(seconds=1),
                ts - pd.Timedelta(seconds=1),
                ts + pd.Timedelta(seconds=9),
                ts + pd.Timedelta(seconds=9),
            ],
            "price": [236.50, 236.50, 236.75, 236.75],
            "volume": [1000.0, 1000.0, 500.0, 500.0],
            "action": pd.Categorical(
                ["changed", "changed", "changed", "changed"],
                categories=["created", "changed", "deleted"],
                ordered=True,
            ),
            "direction": pd.Categorical(
                ["bid", "ask", "bid", "ask"],
                categories=["bid", "ask"],
                ordered=True,
            ),
            "fill": [1234.0, 1234.0, 1234.0, 1234.0],
            "original_number": [1, 2, 3, 4],
        }
    )


@pytest.fixture
def sample_cancellation_events() -> pd.DataFrame:
    """Paired created/deleted flashed-limit orders for the L3 cancellations face.

    Each order ``id`` has a ``created`` row carrying ``aggressiveness_bps``
    (placement distance) and a later ``deleted`` row carrying the cancelled
    volume -- exactly what :func:`prepare_cancellations_l3_data` merges on.
    """
    ts = pd.Timestamp("2015-05-01 01:00:00", tz="UTC")
    rng = np.random.default_rng(7)
    rows: list[dict] = []
    for i in range(8):
        oid = i + 1
        direction = "bid" if i % 2 == 0 else "ask"
        created_ts = ts + pd.Timedelta(seconds=i)
        age = float(rng.uniform(1.0, 20.0))
        vol = float(rng.uniform(100, 1000))
        rows.append(
            {
                "id": oid,
                "timestamp": created_ts,
                "volume": vol,
                "direction": direction,
                "action": "created",
                "type": "flashed-limit",
                "aggressiveness_bps": float(rng.uniform(-10.0, 10.0)),
            }
        )
        rows.append(
            {
                "id": oid,
                "timestamp": created_ts + pd.Timedelta(seconds=age),
                "volume": vol,
                "direction": direction,
                "action": "deleted",
                "type": "flashed-limit",
                "aggressiveness_bps": np.nan,
            }
        )
    df = pd.DataFrame(rows)
    df["direction"] = pd.Categorical(df["direction"], categories=["bid", "ask"])
    df["action"] = pd.Categorical(
        df["action"], categories=["created", "changed", "deleted"], ordered=True
    )
    return df


@pytest.fixture
def sample_order_lifecycle_events() -> pd.DataFrame:
    """Limit-order lifecycles for the order_activity L3 Gantt face.

    Three populations exercise every span branch in
    :func:`prepare_order_activity_l3_data`:

    - ``flashed-limit`` bids (ids 100-102): created then deleted with unchanged
      volume -- placed and pulled (**cancelled**); the span ends at the delete.
    - ``resting-limit`` asks (ids 200-202): created, changed, then deleted --
      rested and provided liquidity before terminating at the delete.
    - one ``resting-limit`` bid (id 300): created only, never removed -- still on
      the book at window end, so its span must extend to *end_time*.

    Prices sit in a tight band so the 1st--99th percentile clip keeps the
    mid-band orders (notably the forever-resting id 300) the tests assert on.
    """
    ts = pd.Timestamp("2015-05-01 01:00:00", tz="UTC")
    rows: list[dict] = []

    flashed = {100: 236.40, 101: 236.45, 102: 236.50}
    for offset, (oid, price) in enumerate(flashed.items()):
        rows.append(
            dict(
                id=oid,
                timestamp=ts + pd.Timedelta(seconds=offset),
                price=price,
                volume=300.0,
                direction="bid",
                action="created",
                fill=0.0,
                type="flashed-limit",
            )
        )
        rows.append(
            dict(
                id=oid,
                timestamp=ts + pd.Timedelta(seconds=offset + 3),
                price=price,
                volume=300.0,
                direction="bid",
                action="deleted",
                fill=0.0,
                type="flashed-limit",
            )
        )

    resting = {200: 236.80, 201: 236.90, 202: 237.00}
    for offset, (oid, price) in enumerate(resting.items()):
        rows.append(
            dict(
                id=oid,
                timestamp=ts + pd.Timedelta(seconds=offset),
                price=price,
                volume=500.0,
                direction="ask",
                action="created",
                fill=0.0,
                type="resting-limit",
            )
        )
        rows.append(
            dict(
                id=oid,
                timestamp=ts + pd.Timedelta(seconds=offset + 5),
                price=price,
                volume=300.0,
                direction="ask",
                action="changed",
                fill=200.0,
                type="resting-limit",
            )
        )
        rows.append(
            dict(
                id=oid,
                timestamp=ts + pd.Timedelta(seconds=offset + 9),
                price=price,
                volume=0.0,
                direction="ask",
                action="deleted",
                fill=300.0,
                type="resting-limit",
            )
        )

    # Forever-resting bid: created once, never removed -> still on the book.
    rows.append(
        dict(
            id=300,
            timestamp=ts,
            price=236.60,
            volume=400.0,
            direction="bid",
            action="created",
            fill=0.0,
            type="resting-limit",
        )
    )

    df = pd.DataFrame(rows)
    df["direction"] = pd.Categorical(df["direction"], categories=["bid", "ask"])
    df["action"] = pd.Categorical(
        df["action"], categories=["created", "changed", "deleted"], ordered=True
    )
    df["type"] = pd.Categorical(
        df["type"],
        categories=[
            "unknown",
            "flashed-limit",
            "resting-limit",
            "market-limit",
            "market",
        ],
        ordered=True,
    )
    return df


@pytest.fixture
def sample_executed_orders() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Created limit orders + executing trades for the L3 execution faces.

    Returns ``(events, trades)``.  Event lifecycles follow the canonical
    schema (outcomes derive from ``order_lifecycles`` via the ``fill``
    column); the trades frame feeds :func:`prepare_trade_tape_l3_data`'s
    maker-bar merge:

    - **id 1** (bid): fully filled (maker of trade T1)        -> ``filled``.
    - **id 2** (ask): half filled (maker of T2) then deleted  -> ``partial``.
    - **id 3** (bid): deleted with no execution               -> ``cancelled``.
    - **id 4** (ask): never filled, never deleted             -> ``resting`` (dropped).
    - **id 5** (bid): fully filled as the *taker* of trade T3 -> ``filled``.
    - **id 6** (ask): fully filled as the maker of trade T4   -> ``filled``.

    Takers that are not measured orders (and the one un-tracked maker in T3) use
    sentinel event ids absent from ``events``; the tape's inner merges drop them.
    """
    ts = pd.Timestamp("2015-05-01 01:00:00", tz="UTC")
    created = [
        dict(event_id=1, id=1, price=100.0, volume=10.0, direction="bid", bps=2.0),
        dict(event_id=2, id=2, price=100.2, volume=10.0, direction="ask", bps=-1.0),
        dict(event_id=3, id=3, price=100.0, volume=8.0, direction="bid", bps=1.0),
        dict(event_id=4, id=4, price=100.2, volume=6.0, direction="ask", bps=-2.0),
        dict(event_id=5, id=5, price=100.1, volume=12.0, direction="bid", bps=3.0),
        dict(event_id=6, id=6, price=100.1, volume=9.0, direction="ask", bps=-3.0),
    ]
    rows: list[dict] = []
    for offset, o in enumerate(created):
        rows.append(
            dict(
                event_id=o["event_id"],
                id=o["id"],
                timestamp=ts + pd.Timedelta(seconds=offset),
                price=o["price"],
                volume=o["volume"],
                direction=o["direction"],
                action="created",
                fill=0.0,
                aggressiveness_bps=o["bps"],
            )
        )
    # Lifecycle rows under the canonical schema: `volume` = outstanding after
    # the event (deleted rows = size removed), `fill` = executed delta.
    # id 1/5/6 fully fill; id 2 half-fills then cancels the rest; id 3 cancels.
    lifecycle_rows = (
        # event_id, id, secs, price, volume, action, fill, direction
        (11, 1, 10, 100.0, 0.0, "deleted", 10.0, "bid"),
        (12, 2, 11, 100.2, 5.0, "changed", 5.0, "ask"),
        (13, 2, 30, 100.2, 5.0, "deleted", 0.0, "ask"),
        (14, 3, 30, 100.0, 8.0, "deleted", 0.0, "bid"),
        (15, 5, 12, 100.1, 0.0, "deleted", 12.0, "bid"),
        (16, 6, 13, 100.1, 0.0, "deleted", 9.0, "ask"),
    )
    for event_id, oid, secs, price, volume, action, fill, direction in lifecycle_rows:
        rows.append(
            dict(
                event_id=event_id,
                id=oid,
                timestamp=ts + pd.Timedelta(seconds=secs),
                price=price,
                volume=volume,
                direction=direction,
                action=action,
                fill=fill,
                aggressiveness_bps=np.nan,
            )
        )
    events = pd.DataFrame(rows)
    events["direction"] = pd.Categorical(events["direction"], categories=["bid", "ask"])
    events["action"] = pd.Categorical(
        events["action"], categories=["created", "changed", "deleted"], ordered=True
    )

    trades = pd.DataFrame(
        [
            # maker, taker, price, volume, aggressor direction
            dict(
                maker_event_id=1,
                taker_event_id=9001,
                price=100.0,
                volume=10.0,
                direction="sell",
            ),
            dict(
                maker_event_id=2,
                taker_event_id=9002,
                price=100.2,
                volume=5.0,
                direction="buy",
            ),
            dict(
                maker_event_id=9003,
                taker_event_id=5,
                price=100.1,
                volume=12.0,
                direction="buy",
            ),
            dict(
                maker_event_id=6,
                taker_event_id=9004,
                price=100.1,
                volume=9.0,
                direction="sell",
            ),
        ]
    )
    trades["timestamp"] = [
        ts + pd.Timedelta(seconds=10 + i) for i in range(len(trades))
    ]
    trades["direction"] = pd.Categorical(
        trades["direction"], categories=["buy", "sell"]
    )
    return events, trades


@pytest.fixture
def cli_runner():
    """Invoke the ob-analytics CLI as a subprocess.

    Returns a callable: ``run(*args, **kwargs) -> subprocess.CompletedProcess``.
    Always passes ``check=False`` (the test decides whether to assert).
    """

    def _run(*args: str, **kwargs) -> subprocess.CompletedProcess:
        cmd = [sys.executable, "-m", "ob_analytics", *args]
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            **kwargs,
        )

    return _run


@pytest.fixture
def bitstamp_sample_orders_only(tmp_path, sample_csv_path) -> Path:
    """Copy just the bundled orders.csv.gz into a temp dir (no trades.csv).

    Use to test the error path when BitstampTradeReader can't find its
    companion file.
    """
    import shutil

    dest = tmp_path / "orders.csv.gz"
    shutil.copy(sample_csv_path, dest)
    return dest


@pytest.fixture
def tiny_depth() -> pd.DataFrame:
    """Minimal depth DataFrame for testing depth_metrics."""
    ts_base = pd.Timestamp("2015-05-01 00:00:00")
    return pd.DataFrame(
        {
            "timestamp": [
                ts_base,
                ts_base + pd.Timedelta(seconds=1),
                ts_base + pd.Timedelta(seconds=2),
                ts_base + pd.Timedelta(seconds=3),
            ],
            "price": [236.50, 237.00, 236.45, 237.10],
            "volume": [10000, 5000, 8000, 3000],
            "direction": pd.Categorical(
                ["bid", "ask", "bid", "ask"],
                categories=["bid", "ask"],
                ordered=True,
            ),
        }
    )
