"""Shared fixtures for ob-analytics tests."""

from pathlib import Path

import pandas as pd
import pytest

SAMPLE_CSV = (
    Path(__file__).resolve().parent.parent
    / "ob_analytics"
    / "_sample_data"
    / "orders.csv"
)


@pytest.fixture(scope="module")
def sample_csv_path() -> Path:
    """Path to the bundled sample orders.csv."""
    if not SAMPLE_CSV.exists():
        pytest.skip(f"Sample data not found: {SAMPLE_CSV}")
    return SAMPLE_CSV


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
    """Path to the bundled Bitstamp sample directory (orders.csv + trades.csv).

    Use when a test needs the directory itself (e.g. to feed
    ``BitstampTradeReader.load(events, source=...)``) rather than just
    the orders.csv path.
    """
    from ob_analytics import sample_data_dir

    d = sample_data_dir()
    if not (d / "orders.csv").exists() or not (d / "trades.csv").exists():
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
