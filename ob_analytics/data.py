"""Data processing utilities: loading, saving, and processing order book data."""

from pathlib import Path

import numpy as np
import pandas as pd

from loguru import logger

from ob_analytics._utils import validate_columns
from ob_analytics.depth import depth_metrics, price_level_volume
from ob_analytics.event_processing import load_event_data, order_aggressiveness
from ob_analytics.matching_engine import event_match
from ob_analytics.order_types import set_order_types
from ob_analytics.trades import match_trades


def get_zombie_ids(events: pd.DataFrame, trades: pd.DataFrame) -> list[int]:
    """
    Identify zombie orders that should be removed from the events DataFrame.

    Parameters
    ----------
    events : pandas.DataFrame
        DataFrame containing limit order events.
    trades : pandas.DataFrame
        DataFrame containing trade executions.

    Returns
    -------
    list of int
        A list of order IDs that are considered zombies.
    """
    validate_columns(
        events,
        {"action", "id", "direction", "timestamp", "price"},
        "get_zombie_ids(events)",
    )
    validate_columns(
        trades,
        {"direction", "timestamp", "price"},
        "get_zombie_ids(trades)",
    )

    # Filter cancelled events
    cancelled = events[events["action"] == "deleted"]["id"]
    zombies = events[~events["id"].isin(cancelled)]

    # We only care about the last state of each zombie
    zombies_last = zombies.drop_duplicates(subset=["id"], keep="last")

    # Separate bid and ask zombies
    bid_zombies = zombies_last[zombies_last["direction"] == "bid"].copy()
    ask_zombies = zombies_last[zombies_last["direction"] == "ask"].copy()

    trades = trades.sort_values("timestamp")

    sell_trades = trades[trades["direction"] == "sell"].copy()
    if not sell_trades.empty:
        sell_trades["min_price_after"] = np.minimum.accumulate(
            np.asarray(sell_trades["price"].values[::-1])
        )[::-1]

    buy_trades = trades[trades["direction"] == "buy"].copy()
    if not buy_trades.empty:
        buy_trades["max_price_after"] = np.maximum.accumulate(
            np.asarray(buy_trades["price"].values[::-1])
        )[::-1]

    valid_bid_zombies = []
    if not bid_zombies.empty and not sell_trades.empty:
        bid_zombies = bid_zombies.sort_values("timestamp")
        merged_bids = pd.merge_asof(
            bid_zombies,
            sell_trades[["timestamp", "min_price_after"]],
            on="timestamp",
            direction="forward",
        )
        valid_bids = merged_bids[merged_bids["price"] > merged_bids["min_price_after"]]
        valid_bid_zombies = valid_bids["id"].tolist()

    valid_ask_zombies = []
    if not ask_zombies.empty and not buy_trades.empty:
        ask_zombies = ask_zombies.sort_values("timestamp")
        merged_asks = pd.merge_asof(
            ask_zombies,
            buy_trades[["timestamp", "max_price_after"]],
            on="timestamp",
            direction="forward",
        )
        valid_asks = merged_asks[merged_asks["price"] < merged_asks["max_price_after"]]
        valid_ask_zombies = valid_asks["id"].tolist()

    # Combine bid and ask zombies
    return valid_bid_zombies + valid_ask_zombies


def process_data(
    csv_file: str, price_digits: int = 2, volume_digits: int = 8
) -> dict[str, pd.DataFrame]:
    """
    Import and preprocess limit order data from a CSV file.

    Parameters
    ----------
    csv_file : str
        The path to the CSV file containing limit order data.
    price_digits : int, optional
        The number of decimal places for the 'price' column. Default is 2.
    volume_digits : int, optional
        The number of decimal places for the 'volume' column. Default is 8.

    Returns
    -------
    dict of str to pandas.DataFrame
        A dictionary containing four pandas DataFrames:
        - 'events': Limit order events.
        - 'trades': Inferred trades (executions).
        - 'depth': Order book price level depth through time.
        - 'depth_summary': Limit order book summary statistics.
    """
    events = load_event_data(csv_file, price_digits, volume_digits)
    events = event_match(events)
    trades = match_trades(events)
    events = set_order_types(events, trades)
    zombie_ids = get_zombie_ids(events, trades)
    events = events[~events["id"].isin(zombie_ids)]

    depth = price_level_volume(events)
    depth_summary = depth_metrics(depth)
    events = order_aggressiveness(events, depth_summary)
    offset = pd.Timedelta(minutes=1)
    return {
        "events": events,
        "trades": trades,
        "depth": depth,
        "depth_summary": depth_summary[
            depth_summary["timestamp"] >= events["timestamp"].min() + offset
        ],
    }


def load_data(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load pre-processed pipeline data from a Parquet directory or pickle file.

    Parameters
    ----------
    path : str or Path
        If *path* is a directory, each ``.parquet`` file inside is loaded
        as a DataFrame keyed by its stem (``events.parquet`` → ``"events"``).
        If *path* is a single file with a ``.pkl`` / ``.pickle`` extension,
        it is loaded via :func:`pandas.read_pickle` for backward
        compatibility (**not recommended** for untrusted data).

    Returns
    -------
    dict of str to pandas.DataFrame
    """
    p = Path(path)
    if p.is_dir():
        result = {}
        for pq in sorted(p.glob("*.parquet")):
            result[pq.stem] = pd.read_parquet(pq)
        if not result:
            raise FileNotFoundError(f"No .parquet files found in {p}")
        return result
    if p.suffix in (".pkl", ".pickle"):
        logger.warning(
            "Loading from pickle ({}). Pickle is insecure for untrusted "
            "data; prefer Parquet via save_data().",
            p,
        )
        return pd.read_pickle(p)
    raise ValueError(
        f"Unsupported format: {p.suffix}. Use a Parquet directory or .pkl file."
    )


def save_data(
    lob_data: dict[str, pd.DataFrame],
    path: str | Path,
    *,
    fmt: str = "parquet",
) -> None:
    """Save pipeline data to disk.

    Parameters
    ----------
    lob_data : dict of str to pandas.DataFrame
        The DataFrames to save (keys become file stems).
    path : str or Path
        Destination directory (Parquet) or file (pickle).
    fmt : ``"parquet"`` or ``"pickle"``
        Serialisation format.  Parquet is the default and recommended
        format -- it is portable, fast, and safe for untrusted data.
    """
    p = Path(path)
    if fmt == "parquet":
        p.mkdir(parents=True, exist_ok=True)
        for name, df in lob_data.items():
            df.to_parquet(p / f"{name}.parquet", index=False)
    elif fmt == "pickle":
        logger.warning(
            "Saving as pickle. Consider using fmt='parquet' for "
            "portability and security."
        )
        pd.to_pickle(lob_data, p)  # type: ignore
    else:
        raise ValueError(f"Unsupported format: {fmt!r}. Use 'parquet' or 'pickle'.")
