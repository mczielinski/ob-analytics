"""Bitstamp data format: event loading, trade reading, and writing.

Contains the symmetric set of Bitstamp-specific components:

* :class:`BitstampLoader` — loads Bitstamp CSV event data
* :class:`BitstampTradeReader` — reads companion ``trades.csv`` (live capture)
* :class:`BitstampWriter` — writes events back to Bitstamp CSV
* :class:`BitstampFormat` — format descriptor bundling all of the above

Format-agnostic analytics (e.g. :func:`~ob_analytics.analytics.order_aggressiveness`)
live in :mod:`ob_analytics.analytics`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from loguru import logger

from ob_analytics._utils import (
    datetime_to_epoch,
    empty_trades,
    epoch_to_datetime,
    validate_columns,
    validate_non_empty,
)
from ob_analytics.config import PipelineConfig
from ob_analytics.protocols import (
    DataWriter,
    EventLoader,
    RunContext,
    TradeSource,
)


# ── BitstampLoader ────────────────────────────────────────────────────


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
            ``timestamp``, ``exchange_timestamp``,
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
        events["timestamp"] = epoch_to_datetime(events["timestamp"], ts_unit)
        events["exchange_timestamp"] = epoch_to_datetime(
            events["exchange_timestamp"], ts_unit
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

        # A volume drop between consecutive events for the same id is always a
        # fill on Bitstamp — the venue does not allow order amendments, so a
        # price change between events is the matching engine reporting the
        # fill price (taker orders) or the order walking the book (aggressors),
        # not an in-place modification.
        fill_deltas = events.groupby("id")["volume"].diff().fillna(0)
        events["fill"] = fill_deltas.abs().round(volume_digits)

        # Sort timestamps within each id.  The frame is id-ordered (primary
        # key of the sort above; _remove_duplicates only filters rows), so a
        # stable global (id, timestamp) lexsort gathered back positionally is
        # equivalent to the per-group sort — without the per-group Python
        # lambda, which dominated load time (~19s on the bundled sample).
        order = np.lexsort((events["timestamp"].to_numpy(), events["id"].to_numpy()))
        events["timestamp"] = events["timestamp"].to_numpy()[order]

        events["raw_event_type"] = pd.NA

        return events

    @staticmethod
    def _remove_duplicates(events: pd.DataFrame) -> pd.DataFrame:
        """Drop redundant 'deleted' events that repeat a cancellation for one order id."""
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


# ── BitstampTradeReader ───────────────────────────────────────────────


_TRADES_COLUMNS = (
    "trade_id",
    "timestamp",
    "exchange_timestamp",
    "price",
    "amount",
    "buy_order_id",
    "sell_order_id",
    "side",
)


class BitstampTradeReader:
    """Build trades from the companion ``trades.csv`` produced by
    ``scripts/collect_bitstamp_btcusd.py``.

    Reads each row from ``<source_dir>/trades.csv`` and projects it into
    the canonical trades schema, joining against the events frame to
    resolve ``buy_order_id`` / ``sell_order_id`` into ``event_id`` and
    ``original_number`` references.

    Satisfies the :class:`~ob_analytics.protocols.TradeSource` protocol.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    def load(self, events: pd.DataFrame, source: Any) -> pd.DataFrame:
        path = self._resolve_trades_path(source)
        raw = pd.read_csv(path)

        if raw.empty:
            return empty_trades()

        validate_columns(raw, set(_TRADES_COLUMNS), "BitstampTradeReader.load")

        # `side` is the taker side: "buy" => taker is the buyer.
        taker_id = np.where(
            raw["side"].astype(str) == "buy",
            raw["buy_order_id"],
            raw["sell_order_id"],
        )
        maker_id = np.where(
            raw["side"].astype(str) == "buy",
            raw["sell_order_id"],
            raw["buy_order_id"],
        )

        ev_lookup = self._build_lookup(events)

        recv_ms = pd.to_datetime(raw["timestamp"], unit="ms", utc=True).dt.tz_convert(
            None
        )
        amounts = raw["amount"].astype(float).round(self._config.volume_decimals)

        maker_event_id = self._resolve_event_ids(maker_id, amounts, ev_lookup)
        taker_event_id = self._resolve_event_ids(taker_id, amounts, ev_lookup)

        id_to_og = dict(zip(events["event_id"], events["original_number"]))
        maker_og = pd.Series(maker_event_id).map(id_to_og)
        taker_og = pd.Series(taker_event_id).map(id_to_og)

        trades = pd.DataFrame(
            {
                "timestamp": recv_ms.values,
                "price": raw["price"].astype(float).values,
                "volume": amounts.values,
                "direction": pd.Categorical(
                    raw["side"].astype(str), categories=["buy", "sell"], ordered=True
                ),
                "maker_event_id": maker_event_id,
                "taker_event_id": taker_event_id,
                "maker": maker_id,
                "taker": taker_id,
                "maker_og": maker_og.values,
                "taker_og": taker_og.values,
            }
        )

        trades = trades.sort_values("timestamp", kind="stable").reset_index(drop=True)

        logger.info(
            "BitstampTradeReader: {} trades ({} maker_event_id resolved, "
            "{} taker_event_id resolved)",
            len(trades),
            trades["maker_event_id"].notna().sum(),
            trades["taker_event_id"].notna().sum(),
        )
        return trades

    @staticmethod
    def _resolve_trades_path(source: Any) -> Path:
        p = Path(source)
        if p.is_file():
            p = p.parent
        candidate = p / "trades.csv"
        if not candidate.exists():
            raise FileNotFoundError(
                f"BitstampTradeReader: no trades.csv next to {source!r} "
                f"(looked at {candidate}). Capture trades alongside orders "
                f"with scripts/collect_bitstamp_btcusd.py."
            )
        return candidate

    @staticmethod
    def _build_lookup(
        events: pd.DataFrame,
    ) -> dict[int, list[tuple[float, int]]]:
        out: dict[int, list[tuple[float, int]]] = {}
        non_zero = events[events["fill"] > 0]
        for oid, fill, eid in zip(
            non_zero["id"], non_zero["fill"], non_zero["event_id"]
        ):
            out.setdefault(int(oid), []).append((float(fill), int(eid)))
        return out

    def _resolve_event_ids(
        self,
        order_ids: np.ndarray,
        amounts: pd.Series,
        ev_lookup: dict[int, list[tuple[float, int]]],
    ) -> np.ndarray:
        digits = self._config.volume_decimals

        # Bucket each order's candidate fills by rounded volume, preserving
        # candidate order within each bucket: the earliest unconsumed match
        # wins, at O(1) per trade.  The index is built fresh per call so the
        # maker and taker passes consume independently.
        index: dict[int, dict[float, deque[int]]] = {}
        for oid, cand in ev_lookup.items():
            by_fill: dict[float, deque[int]] = {}
            for fill, eid in cand:
                by_fill.setdefault(round(fill, digits), deque()).append(eid)
            index[oid] = by_fill

        result: list[int | float] = []
        for oid, amt in zip(order_ids, amounts):
            picked: int | float = float("nan")
            bucket = index.get(int(oid))
            if bucket is not None:
                matches = bucket.get(round(float(amt), digits))
                if matches:
                    picked = matches.popleft()
            result.append(picked)

        return np.array(result, dtype=object)


# ── BitstampWriter ────────────────────────────────────────────────────


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

        When *data* also carries a ``"trades"`` frame, a companion
        ``trades.csv`` is written next to *dest* in capture format, so that a
        subsequent :class:`BitstampTradeReader` can reconstruct trades when the
        orders.csv is re-read.

        Parameters
        ----------
        data : dict of str to DataFrame
            Must contain an ``"events"`` key.  An optional ``"trades"`` key
            triggers a companion ``trades.csv`` alongside *dest*.
        dest : str or Path
            Output CSV file path.

        Returns
        -------
        Path
            The written events CSV path.
        """
        events = data["events"]
        p = Path(dest)
        ts_unit = self._config.timestamp_unit
        out = pd.DataFrame(
            {
                "id": events["id"],
                "timestamp": datetime_to_epoch(events["timestamp"], ts_unit),
                "exchange_timestamp": datetime_to_epoch(
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

        trades = data.get("trades")
        if trades is not None:
            self._write_companion_trades(trades, p.parent / "trades.csv")
        return p

    @staticmethod
    def _write_companion_trades(trades: pd.DataFrame, dest: Path) -> None:
        """Write a capture-style ``trades.csv`` next to the events CSV.

        Reconstructs the live-capture trade schema from the pipeline's
        ``maker``/``taker`` fields so that :class:`BitstampTradeReader` can
        re-read it.  An empty *trades* frame yields a header-only file.
        """
        cols = [
            "trade_id",
            "timestamp",
            "exchange_timestamp",
            "price",
            "amount",
            "buy_order_id",
            "sell_order_id",
            "side",
        ]
        if trades.empty:
            pd.DataFrame(columns=cols).to_csv(dest, index=False)
            return
        side = trades["direction"].astype(str)
        buy_order_id = np.where(side == "buy", trades["taker"], trades["maker"])
        sell_order_id = np.where(side == "buy", trades["maker"], trades["taker"])
        ts_ns = trades["timestamp"].astype("datetime64[ns]").astype(np.int64)
        ts_ms = ts_ns // 1_000_000
        out = pd.DataFrame(
            {
                "trade_id": np.arange(1, len(trades) + 1, dtype=np.int64),
                "timestamp": ts_ms,
                "exchange_timestamp": ts_ms,
                "price": trades["price"].to_numpy(),
                "amount": trades["volume"].to_numpy(),
                "buy_order_id": buy_order_id,
                "sell_order_id": sell_order_id,
                "side": side.to_numpy(),
            }
        )
        out.to_csv(dest, index=False)


# ── BitstampFormat descriptor ─────────────────────────────────────────


@dataclass
class BitstampFormat:
    """Format descriptor for Bitstamp-style CSV data.

    Conforms structurally to the :class:`~ob_analytics.protocols.Format`
    Protocol — no inheritance required.
    """

    name: str = "bitstamp"

    def create_loader(self, config: PipelineConfig, ctx: RunContext) -> EventLoader:
        return BitstampLoader(config)

    def create_trade_source(
        self, config: PipelineConfig, ctx: RunContext
    ) -> TradeSource:
        return BitstampTradeReader(config)

    def create_writer(self, config: PipelineConfig, ctx: RunContext) -> DataWriter:
        return BitstampWriter(config)

    def compute_depth(
        self,
        events: pd.DataFrame,
        config: Any,
        source: Any,
        ctx: RunContext,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        # Bitstamp uses the standard price_level_volume -> depth_metrics path.
        return None

    def config_defaults(self) -> dict[str, Any]:
        return {
            "price_decimals": 2,
            "volume_decimals": 8,
            "timestamp_unit": "ms",
        }


# ── Register this format and its writer ───────────────────────────────
# Imports sit at the bottom (deferred from the top of the module) to avoid a
# circular import: ``pipeline`` imports ``BitstampLoader``/``BitstampTradeReader``
# from here.
from ob_analytics.data import register_writer  # noqa: E402
from ob_analytics.pipeline import register_format  # noqa: E402

register_format("bitstamp", BitstampFormat)
register_writer("bitstamp", lambda config, ctx: BitstampWriter(config))
