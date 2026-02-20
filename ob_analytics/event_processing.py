"""Event loading and aggressiveness computation.

Contains :class:`BitstampLoader` (the default :class:`EventLoader`
implementation) and the :func:`order_aggressiveness` calculation.
"""


from pathlib import Path

import numpy as np
import pandas as pd

from loguru import logger

from ob_analytics._utils import validate_columns, validate_non_empty
from ob_analytics.config import PipelineConfig
from ob_analytics.exceptions import InvalidDataError


class BitstampLoader:
    """Load raw limit-order events from a Bitstamp-format CSV.

    Satisfies the :class:`~ob_analytics.protocols.EventLoader` protocol.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration.  ``price_decimals`` and ``volume_decimals``
        control rounding precision.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    def load(self, source: str | Path) -> pd.DataFrame:
        """Read *source* CSV and return a cleaned events DataFrame.

        Parameters
        ----------
        source : str or Path
            Path to a Bitstamp-format CSV with columns ``id``,
            ``timestamp``, ``exchange.timestamp`` (or ``exchange_timestamp``),
            ``price``, ``volume``, ``action``, ``direction``.

        Returns
        -------
        pandas.DataFrame
            Cleaned events with columns ``id``, ``timestamp``,
            ``exchange_timestamp``, ``price``, ``volume``, ``action``,
            ``direction``, ``event_id``, ``fill``, ``original_number``.
        """
        price_digits = self._config.price_decimals
        volume_digits = self._config.volume_decimals

        events = pd.read_csv(source)
        if "exchange.timestamp" in events.columns:
            events = events.rename(columns={"exchange.timestamp": "exchange_timestamp"})
        validate_columns(
            events,
            {"id", "timestamp", "exchange_timestamp", "price", "volume", "action", "direction"},
            "BitstampLoader.load",
        )
        validate_non_empty(events, "BitstampLoader.load")

        events = events[events["volume"] >= 0]
        events = events.reset_index().rename(columns={"index": "original_number"})
        events.original_number = events.original_number + 1
        events["volume"] = events["volume"].round(volume_digits)
        events["price"] = events["price"].round(price_digits)

        events["timestamp"] = pd.to_datetime(events["timestamp"] / 1000, unit="s")
        events["exchange_timestamp"] = pd.to_datetime(
            events["exchange_timestamp"] / 1000, unit="s"
        )
        events["action"] = pd.Categorical(
            events["action"], categories=["created", "changed", "deleted"], ordered=True
        )
        events["direction"] = pd.Categorical(
            events["direction"], categories=["bid", "ask"], ordered=True
        )

        events = events.sort_values(
            by=["id", "volume", "action", "timestamp"],
            ascending=[True, False, True, True],
            kind="stable",
        )

        events["event_id"] = np.arange(1, len(events) + 1)
        events = self._remove_duplicates(events)

        fill_deltas = events.groupby("id")["volume"].diff().fillna(0)
        price_deltas = events.groupby("id")["price"].diff().fillna(0)
        fill_deltas = fill_deltas.where(price_deltas == 0, 0)
        events["fill"] = fill_deltas.abs().round(volume_digits)

        ts_sorted = events.groupby("id")["timestamp"].transform(
            lambda x: np.sort(x.values, kind="stable")
        )
        events["timestamp"] = ts_sorted

        return events

    @staticmethod
    def _remove_duplicates(events: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate delete events, matching R's removeDuplicates."""
        deletes = events[events["action"] == "deleted"].sort_values(
            by=["id", "volume"], kind="stable"
        )
        dup_ids = deletes.loc[deletes["id"].duplicated(), "id"]
        duplicate_deletes = deletes[deletes["id"].isin(dup_ids)]
        duplicate_event_ids = duplicate_deletes.loc[
            duplicate_deletes["id"].duplicated(), "event_id"
        ]

        rem_dup = len(duplicate_event_ids)
        if rem_dup > 0:
            removed_ids = events.loc[
                events["event_id"].isin(duplicate_event_ids), "id"
            ]
            logger.warning(
                "Removed {} duplicate order cancellations: {}",
                rem_dup,
                " ".join(removed_ids.astype(str)),
            )

        return events[~events["event_id"].isin(duplicate_event_ids)]


# ── Backward-compatible module-level function ─────────────────────────


def load_event_data(
    file: str, price_digits: int = 2, volume_digits: int = 8
) -> pd.DataFrame:
    """Read raw limit order event data from a CSV file.

    This is a convenience wrapper around :class:`BitstampLoader`.

    Parameters
    ----------
    file : str
        The path to the CSV file containing limit order events.
    price_digits : int, optional
        The number of decimal places for the 'price' column. Default is 2.
    volume_digits : int, optional
        The number of decimal places for the 'volume' column. Default is 8.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the raw limit order events data.
    """
    config = PipelineConfig(price_decimals=price_digits, volume_decimals=volume_digits)
    return BitstampLoader(config).load(file)


# ── Order aggressiveness (standalone function) ────────────────────────


def order_aggressiveness(
    events: pd.DataFrame, depth_summary: pd.DataFrame
) -> pd.DataFrame:
    """Calculate order aggressiveness with respect to the best bid or ask in BPS.

    Parameters
    ----------
    events : pandas.DataFrame
        The events DataFrame.
    depth_summary : pandas.DataFrame
        The order book summary statistics DataFrame.

    Returns
    -------
    pandas.DataFrame
        The events DataFrame with an added ``aggressiveness_bps`` column.
    """
    validate_columns(
        events,
        {"direction", "action", "type", "timestamp", "event_id", "price"},
        "order_aggressiveness(events)",
    )
    validate_columns(
        depth_summary,
        {"timestamp"},
        "order_aggressiveness(depth_summary)",
    )

    def event_diff_bps(events: pd.DataFrame, direction: int) -> pd.DataFrame:
        side = "bid" if direction == 1 else "ask"
        orders = events[
            (events["direction"] == side)
            & (events["action"] != "changed")
            & events["type"].isin(["flashed-limit", "resting-limit"])
        ].sort_values(by="timestamp", kind="stable")

        if not all(orders["timestamp"].isin(depth_summary["timestamp"])):
            raise InvalidDataError(
                "Not all order timestamps are present in depth_summary. "
                "Ensure depth_summary covers the full event time range."
            )

        best_price_col = f"best_{side}_price"

        unique_depth_summary = depth_summary.drop_duplicates(subset=["timestamp"])
        merged = pd.merge(
            orders,
            unique_depth_summary[["timestamp", best_price_col]],
            on="timestamp",
            how="left",
        )

        best = merged[best_price_col].shift(1)

        merged = merged.iloc[1:].copy()
        best = best.iloc[1:]

        diff_price = direction * (merged["price"] - best)
        diff_bps = 10000 * diff_price / best
        return pd.DataFrame({"event_id": merged["event_id"], "diff_bps": diff_bps})

    bid_diff = event_diff_bps(events, 1)
    ask_diff = event_diff_bps(events, -1)
    events["aggressiveness_bps"] = np.nan

    if not bid_diff.empty:
        events = pd.merge(events, bid_diff, on="event_id", how="left")
        events["aggressiveness_bps"] = events["aggressiveness_bps"].fillna(events["diff_bps"])
        events.drop(columns=["diff_bps"], inplace=True)

    if not ask_diff.empty:
        events = pd.merge(events, ask_diff, on="event_id", how="left")
        events["aggressiveness_bps"] = events["aggressiveness_bps"].fillna(events["diff_bps"])
        events.drop(columns=["diff_bps"], inplace=True)

    return events
