"""Trade inference from matched events.

Contains :class:`DefaultTradeInferrer` (the default
:class:`TradeInferrer` implementation) and :func:`trade_impacts`.
"""


import numpy as np
import pandas as pd

from loguru import logger

from ob_analytics._utils import validate_columns, validate_non_empty, vwap
from ob_analytics.config import PipelineConfig
from ob_analytics.exceptions import MatchingError


class DefaultTradeInferrer:
    """Construct trade records from matched bid/ask event pairs.

    Satisfies the :class:`~ob_analytics.protocols.TradeInferrer` protocol.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration.  ``price_jump_threshold`` controls when the
        maker/taker swap heuristic is triggered.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    def infer_trades(self, events: pd.DataFrame) -> pd.DataFrame:
        """Build a trades DataFrame from matched events.

        Determines maker vs taker using exchange timestamps, and applies
        a price-jump heuristic to correct misattributions.

        Parameters
        ----------
        events : pandas.DataFrame
            Events with ``matching_event`` column populated.

        Returns
        -------
        pandas.DataFrame
            Trades with columns ``timestamp``, ``price``, ``volume``,
            ``direction``, ``maker_event_id``, ``taker_event_id``,
            ``maker``, ``taker``.
        """
        validate_columns(
            events,
            {
                "direction", "matching_event", "event_id", "exchange_timestamp",
                "timestamp", "price", "fill", "id", "original_number",
            },
            "DefaultTradeInferrer.infer_trades",
        )
        validate_non_empty(events, "DefaultTradeInferrer.infer_trades")

        matching_bids = events[
            (events["direction"] == "bid") & ~pd.isna(events["matching_event"])
        ].sort_values(by="event_id", kind="stable")
        matching_asks = events[
            (events["direction"] == "ask") & ~pd.isna(events["matching_event"])
        ].sort_values(by="matching_event", kind="stable")

        if not np.asarray(
            matching_bids["event_id"].values == matching_asks["matching_event"].values
        ).all():
            raise MatchingError(
                "Bid event IDs do not align with ask matching events. "
                "This indicates a matching error in the upstream eventMatch step."
            )

        matching_bids = matching_bids.reset_index(drop=True)
        matching_asks = matching_asks.reset_index(drop=True)
        bid_exchange_ts = matching_bids["exchange_timestamp"]
        ask_exchange_ts = matching_asks["exchange_timestamp"]
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
            bid_maker, matching_bids["event_id"], matching_asks["event_id"]
        )
        taker_event_id = np.where(
            bid_maker, matching_asks["event_id"], matching_bids["event_id"]
        )

        id_to_id = dict(zip(events["event_id"], events["id"]))
        id_to_original_number = dict(zip(events["event_id"], events["original_number"]))

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
                "maker_event_id": maker_event_id,
                "taker_event_id": taker_event_id,
                "maker": maker,
                "taker": taker,
                "maker_og": maker_og,
                "taker_og": taker_og,
            }
        )
        combined = combined.sort_values(by="timestamp", kind="stable")

        self._fix_price_jumps(combined, events)
        return combined

    def _fix_price_jumps(
        self, combined: pd.DataFrame, events: pd.DataFrame
    ) -> None:
        """Swap maker/taker when consecutive prices jump beyond threshold."""
        threshold = self._config.price_jump_threshold
        jumps = np.where(abs(np.diff(combined["price"])) > threshold)[0]
        if len(jumps) == 0:
            return

        if jumps[0] == 0:
            jumps = jumps[1:]

        logger.warning(
            "{}: {} jumps > ${:.0f} (swapping makers with takers)",
            combined["timestamp"].iloc[0].strftime("%Y-%m-%d"),
            len(jumps),
            threshold,
        )

        for i in jumps:
            prev_jump, this_jump = combined.iloc[i - 1], combined.iloc[i]
            if abs(this_jump["price"] - prev_jump["price"]) > threshold:
                taker_eid = this_jump["taker_event_id"]
                taker_price = events.loc[
                    events["event_id"] == taker_eid, "price"
                ].iloc[0]
                taker_dir = "sell" if this_jump["direction"] == "buy" else "buy"
                swap = pd.DataFrame(
                    {
                        "price": taker_price,
                        "direction": taker_dir,
                        "maker_event_id": taker_eid,
                        "taker_event_id": this_jump["maker_event_id"],
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
                        "price", "direction", "maker_event_id",
                        "taker_event_id", "maker", "taker",
                        "maker_og", "taker_og",
                    ],
                ] = swap.iloc[0]


# ── Backward-compatible module-level functions ────────────────────────


def match_trades(events: pd.DataFrame) -> pd.DataFrame:
    """Construct a DataFrame of inferred trades (executions).

    This is a convenience wrapper around :class:`DefaultTradeInferrer`.

    Parameters
    ----------
    events : pandas.DataFrame
        The limit order events DataFrame with assigned maker/taker event IDs.

    Returns
    -------
    pandas.DataFrame
        A DataFrame describing trade executions.
    """
    return DefaultTradeInferrer().infer_trades(events)


def trade_impacts(trades: pd.DataFrame) -> pd.DataFrame:
    """Generate a DataFrame containing order book impact summaries.

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
        return {
            "id": impact["taker"].iloc[-1],
            "min_price": impact["price"].min(),
            "max_price": impact["price"].max(),
            "vwap": vwap(np.asarray(impact["price"].values), np.asarray(impact["volume"].values)),
            "hits": len(impact),
            "vol": impact["volume"].sum(),
            "start_time": impact["timestamp"].min(),
            "end_time": impact["timestamp"].max(),
            "dir": impact["direction"].iloc[-1],
        }

    impacts = trades.groupby("taker").apply(impact_summary).reset_index(drop=True)  # type: ignore
    return pd.DataFrame(impacts)
