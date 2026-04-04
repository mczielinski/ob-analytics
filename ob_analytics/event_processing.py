"""Bitstamp data format: event loading and writing.

Contains :class:`BitstampLoader` (the :class:`EventLoader` implementation
for Bitstamp CSV data), :class:`BitstampWriter` (Bitstamp CSV writer),
and :class:`BitstampFormat` (format descriptor).

Format-agnostic analytics (e.g. :func:`~ob_analytics.analytics.order_aggressiveness`)
live in :mod:`ob_analytics.analytics`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from loguru import logger

from ob_analytics._utils import validate_columns, validate_non_empty
from ob_analytics.config import PipelineConfig
from ob_analytics.protocols import (
    DataWriter,
    EventLoader,
    Format,
    MatchingEngine,
    TradeInferrer,
)


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
            {
                "id",
                "timestamp",
                "exchange_timestamp",
                "price",
                "volume",
                "action",
                "direction",
            },
            "BitstampLoader.load",
        )
        validate_non_empty(events, "BitstampLoader.load")

        events = events[events["volume"] >= 0]
        events = events.reset_index().rename(columns={"index": "original_number"})
        events["original_number"] = events["original_number"] + 1
        events["volume"] = events["volume"].round(volume_digits)
        events["price"] = events["price"].round(price_digits)

        ts_unit = self._config.timestamp_unit
        events["timestamp"] = pd.to_datetime(events["timestamp"], unit=ts_unit)
        events["exchange_timestamp"] = pd.to_datetime(
            events["exchange_timestamp"], unit=ts_unit
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

        ts_sorted: pd.Series = events.groupby("id")["timestamp"].transform(
            lambda x: np.sort(np.asarray(x.values), kind="stable")
        )
        events["timestamp"] = ts_sorted

        events["raw_event_type"] = pd.NA

        return events

    @staticmethod
    def _remove_duplicates(events: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate delete events (port of R ``removeDuplicates``)."""
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
            removed_ids = events.loc[events["event_id"].isin(duplicate_event_ids), "id"]
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



# ── BitstampWriter ────────────────────────────────────────────────────


def _datetime_to_numeric(series: pd.Series, unit: str) -> pd.Series:
    """Convert a datetime Series back to numeric timestamps."""
    epoch = pd.Timestamp("1970-01-01")
    delta = series - epoch
    divisors = {"ms": 1_000_000, "us": 1_000, "ns": 1}
    return (delta.view("int64") // divisors[unit]).astype("int64")


class BitstampWriter:
    """Write pipeline events back to Bitstamp-format CSV.

    Satisfies the :class:`~ob_analytics.protocols.DataWriter` protocol.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    def write(
        self,
        data: dict[str, pd.DataFrame],
        dest: str | Path,
        **kwargs: Any,
    ) -> Path:
        """Write the events DataFrame to a Bitstamp-format CSV.

        Parameters
        ----------
        data : dict of str to DataFrame
            Must contain an ``"events"`` key.
        dest : str or Path
            Output CSV file path.

        Returns
        -------
        Path
            The written file path.
        """
        events = data["events"]
        p = Path(dest)
        ts_unit = self._config.timestamp_unit
        out = pd.DataFrame(
            {
                "id": events["id"],
                "timestamp": _datetime_to_numeric(events["timestamp"], ts_unit),
                "exchange.timestamp": _datetime_to_numeric(
                    events["exchange_timestamp"], ts_unit
                ),
                "price": events["price"],
                "volume": events["volume"],
                "action": events["action"].astype(str),
                "direction": events["direction"].astype(str),
            }
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(p, index=False)
        return p


# ── BitstampFormat descriptor ─────────────────────────────────────────


class BitstampFormat(Format):
    """Format descriptor for Bitstamp-style CSV data."""

    def create_loader(self, config: PipelineConfig) -> EventLoader:
        return BitstampLoader(config)

    def create_matcher(self, config: PipelineConfig) -> MatchingEngine:
        from ob_analytics.matching_engine import NeedlemanWunschMatcher

        return NeedlemanWunschMatcher(config)

    def create_trade_inferrer(self, config: PipelineConfig) -> TradeInferrer:
        from ob_analytics.trades import DefaultTradeInferrer

        return DefaultTradeInferrer(config)

    def create_writer(self, config: PipelineConfig) -> DataWriter:
        return BitstampWriter(config)

    def config_defaults(self) -> dict[str, Any]:
        return {
            "price_decimals": 2,
            "volume_decimals": 8,
            "timestamp_unit": "ms",
            "match_cutoff_ms": 5000,
            "price_jump_threshold": 10.0,
        }
