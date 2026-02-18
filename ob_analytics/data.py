import pandas as pd

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
    # Filter cancelled events
    cancelled = events[events["action"] == "deleted"]["id"]
    zombies = events[~events["id"].isin(cancelled)]

    # Separate bid and ask zombies
    bid_zombies = zombies[zombies["direction"] == "bid"]
    ask_zombies = zombies[zombies["direction"] == "ask"]

    # Identify bid zombies
    bid_zombie_ids = bid_zombies["id"].unique()
    valid_bid_zombies = []
    for bid_id in bid_zombie_ids:
        zombie = bid_zombies[bid_zombies["id"] == bid_id].iloc[-1]
        if any(
            (trades["direction"] == "sell")
            & (trades["timestamp"] >= zombie["timestamp"])
            & (trades["price"] < zombie["price"])
        ):
            valid_bid_zombies.append(bid_id)

    # Identify ask zombies
    ask_zombie_ids = ask_zombies["id"].unique()
    valid_ask_zombies = []
    for ask_id in ask_zombie_ids:
        zombie = ask_zombies[ask_zombies["id"] == ask_id].iloc[-1]
        if any(
            (trades["direction"] == "buy")
            & (trades["timestamp"] >= zombie["timestamp"])
            & (trades["price"] > zombie["price"])
        ):
            valid_ask_zombies.append(ask_id)

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


def load_data(bin_file: str) -> dict[str, pd.DataFrame]:
    """
    Load pre-processed data from a file.

    Parameters
    ----------
    bin_file : str
        The path to the file containing pre-processed data.

    Returns
    -------
    dict of str to pandas.DataFrame
        A dictionary containing the loaded DataFrames, similar to the output of `process_data`.
    """
    return pd.read_pickle(bin_file)  # Assuming pickle format


def save_data(lob_data: dict[str, pd.DataFrame], bin_file: str) -> None:
    """
    Save processed data to a file.

    Parameters
    ----------
    lob_data : dict of str to pandas.DataFrame
        A dictionary containing the DataFrames to save.
    bin_file : str
        The path to the file where the data will be saved.

    Returns
    -------
    None
    """
    pd.to_pickle(lob_data, bin_file)  # Assuming pickle format
