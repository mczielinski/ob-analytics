import logging

import numpy as np
import pandas as pd

from ob_analytics._utils import validate_columns, validate_non_empty, vwap
from ob_analytics.exceptions import MatchingError

logger = logging.getLogger(__name__)


def match_trades(events: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a DataFrame of inferred trades (executions).

    Parameters
    ----------
    events : pandas.DataFrame
        The limit order events DataFrame with assigned maker/taker event IDs.

    Returns
    -------
    pandas.DataFrame
        A DataFrame describing trade executions.
    """
    validate_columns(
        events,
        {
            "direction", "matching.event", "event.id", "exchange.timestamp",
            "timestamp", "price", "fill", "id", "original_number",
        },
        "match_trades",
    )
    validate_non_empty(events, "match_trades")

    matching_bids = events[
        (events["direction"] == "bid") & ~pd.isna(events["matching.event"])
    ].sort_values(by="event.id", kind="stable")
    matching_asks = events[
        (events["direction"] == "ask") & ~pd.isna(events["matching.event"])
    ].sort_values(by="matching.event", kind="stable")

    if not all(matching_bids["event.id"].values == matching_asks["matching.event"].values):
        raise MatchingError(
            "Bid event IDs do not align with ask matching events. "
            "This indicates a matching error in the upstream eventMatch step."
        )

    matching_bids = matching_bids.reset_index(drop=True)
    matching_asks = matching_asks.reset_index(drop=True)
    bid_exchange_ts = matching_bids["exchange.timestamp"]
    ask_exchange_ts = matching_asks["exchange.timestamp"]
    bid_maker = (bid_exchange_ts < ask_exchange_ts) | (
        (bid_exchange_ts == ask_exchange_ts)
        & (matching_bids["id"] < matching_asks["id"])
    )

    timestamp = np.where(
        matching_bids["timestamp"] <= matching_asks["timestamp"],
        matching_bids["timestamp"],
        matching_asks["timestamp"],
    )

    price = np.where(bid_maker, matching_bids["price"], matching_asks["price"])

    volume = matching_bids["fill"]

    direction = pd.Categorical(
        np.where(bid_maker, "sell", "buy"), categories=["buy", "sell"], ordered=True
    ).astype(str)

    maker_event_id = np.where(
        bid_maker, matching_bids["event.id"], matching_asks["event.id"]
    )
    taker_event_id = np.where(
        bid_maker, matching_asks["event.id"], matching_bids["event.id"]
    )

    # Create a mapping dictionary for id and original_number
    id_to_id = dict(zip(events["event.id"], events["id"]))
    id_to_original_number = dict(zip(events["event.id"], events["original_number"]))

    # Use map to ensure order is preserved
    maker = pd.Series(maker_event_id).map(id_to_id).values
    taker = pd.Series(taker_event_id).map(id_to_id).values

    maker_og = pd.Series(maker_event_id).map(id_to_original_number).values
    taker_og = pd.Series(taker_event_id).map(id_to_original_number).values

    combined = pd.DataFrame(
        {
            "timestamp": timestamp,
            "price": price,
            "volume": volume,
            "direction": direction,
            "maker.event.id": maker_event_id,
            "taker.event.id": taker_event_id,
            "maker": maker,
            "taker": taker,
            "maker_og": maker_og,
            "taker_og": taker_og,
        }
    )
    combined = combined.sort_values(by="timestamp", kind="stable")

    jumps = np.where(abs(np.diff(combined["price"])) > 10)[0]
    if len(jumps) > 0:
        if jumps[0] == 0:
            jumps = jumps[1:]
        logger.warning(
            "%s: %d jumps > $10 (swapping makers with takers)",
            combined["timestamp"].iloc[0].strftime("%Y-%m-%d"),
            len(jumps),
        )
        for i in jumps:
            prev_jump, this_jump = combined.iloc[i - 1], combined.iloc[i]
            if abs(this_jump["price"] - prev_jump["price"]) > 10:
                taker_event_id = this_jump["taker.event.id"]
                taker_price = events.loc[
                    events["event.id"] == taker_event_id, "price"
                ].iloc[0]
                taker_dir = "sell" if this_jump["direction"] == "buy" else "buy"
                swap = pd.DataFrame(
                    {
                        "price": taker_price,
                        "direction": taker_dir,
                        "maker.event.id": taker_event_id,
                        "taker.event.id": this_jump["maker.event.id"],
                        "maker": this_jump["taker"],
                        "taker": this_jump["maker"],
                        "maker_og": this_jump["taker_og"],
                        "taker_og": this_jump["maker_og"],
                    },
                    index=[i],
                )

                combined.loc[
                    i,
                    [
                        "price",
                        "direction",
                        "maker.event.id",
                        "taker.event.id",
                        "maker",
                        "taker",
                        "maker_og",
                        "taker_og",
                    ],
                ] = swap.iloc[0]

    return combined


def trade_impacts(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a DataFrame containing order book impact summaries.

    Parameters
    ----------
    trades : pandas.DataFrame
        The trades DataFrame.

    Returns
    -------
    pandas.DataFrame
        A DataFrame summarizing market order impacts.
    """

    validate_columns(
        trades,
        {"taker", "price", "volume", "timestamp", "direction"},
        "trade_impacts",
    )
    validate_non_empty(trades, "trade_impacts")

    def impact_summary(impact: pd.DataFrame) -> dict:
        """
        Summarize the impact of a trade.

        Parameters
        ----------
        impact : pandas.DataFrame
            DataFrame containing trade information for a single taker.

        Returns
        -------
        dict
            A dictionary summarizing the trade impact.
        """
        return {
            "id": impact["taker"].iloc[-1],
            "min.price": impact["price"].min(),
            "max.price": impact["price"].max(),
            "vwap": vwap(impact["price"], impact["volume"]),
            "hits": len(impact),
            "vol": impact["volume"].sum(),
            "start.time": impact["timestamp"].min(),
            "end.time": impact["timestamp"].max(),
            "dir": impact["direction"].iloc[-1],
        }

    impacts = trades.groupby("taker").apply(impact_summary).reset_index(drop=True)
    return pd.DataFrame(impacts)
