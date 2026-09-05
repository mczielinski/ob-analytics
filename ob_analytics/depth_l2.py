"""L2 (price-level) depth-native data format.

The counterpart to :mod:`ob_analytics.bitstamp` / :mod:`ob_analytics.lobster`
for **price-level (L2 / market-by-price) feeds** — the ones that publish
``[price, quantity]`` levels and diffs with **no order IDs** (Binance, Kalshi,
Polymarket, most CCXT sources).  ob-analytics was built around an L3 per-order
stream; this module adds the shared path those aggregated feeds route through
instead of faking per-order state (see :class:`~ob_analytics.protocols.Level`).

The symmetric set, mirroring the other formats:

* :class:`L2DepthLoader` — reads the canonical L2 CSV schema (a snapshot
  followed by price-level updates) straight into the depth frame.  Satisfies
  :class:`~ob_analytics.protocols.DepthSource`.
* :class:`L2TradeReader` — reads a companion ``trades.csv`` of raw prints.
  Satisfies :class:`~ob_analytics.protocols.TradeSource`.
* :class:`DepthCsvWriter` — round-trips the depth frame (and trades) back to
  the L2 CSV schema.  Satisfies :class:`~ob_analytics.protocols.DataWriter`.
* :class:`DepthCsvSource` — the ``depth_csv`` source descriptor, declaring
  :attr:`~ob_analytics.protocols.Level.L2`.

The L2 CSV schema
-----------------

One row per price-level update, each carrying the level's **new absolute**
resting size (``0`` removes the level) — *not* a signed delta.  A book
snapshot is simply the opening block of rows.  This is exactly what
:class:`~ob_analytics.depth.DepthMetricsEngine` consumes, so there is nothing
to reconstruct.

======  ==================================================================
column  meaning
======  ==================================================================
timestamp   receive time — integer epoch (``config.timestamp_unit``) or any
            string :func:`pandas.to_datetime` understands
side        ``bid`` / ``ask`` (``buy`` / ``sell`` and ``b`` / ``a`` accepted)
price       price level (divided by ``config.price_divisor`` to the quote
            currency, then stored as integer ``tick_size`` counts — issue #155)
volume      new absolute resting size at that level (``0`` = level removed)
======  ==================================================================

Column names are flexible: ``side`` / ``direction`` and ``volume`` / ``size``
/ ``amount`` / ``quantity`` are all accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ob_analytics._utils import (
    UTC_NS_DTYPE,
    attach_ingest_seq,
    datetime_to_epoch,
    empty_trades,
    epoch_to_datetime,
    price_to_ticks,
    ticks_to_price,
    validate_columns,
    validate_non_empty,
)
from ob_analytics.config import PipelineConfig, SourceSettings
from ob_analytics.exceptions import ConfigError
from ob_analytics.protocols import (
    DataWriter,
    DepthSource,
    FeedType,
    Level,
    RunContext,
    TradeSource,
)
from ob_analytics.schemas import SEQUENCE_COLUMN, attach_instrument_identity

# ── Column-spelling tolerance ─────────────────────────────────────────

_TIMESTAMP_COLUMNS: tuple[str, ...] = ("timestamp", "time", "ts")
_SIDE_COLUMNS: tuple[str, ...] = ("side", "direction")
_VOLUME_COLUMNS: tuple[str, ...] = ("volume", "size", "amount", "quantity")
# Venue per-event sequence, when the capture recorded one (CCXT persists its
# `nonce` here). Absent from most price-level captures.
_SEQUENCE_COLUMNS: tuple[str, ...] = ("sequence", "seq", "nonce")
_SIDE_TO_DIRECTION: dict[str, str] = {
    "bid": "bid",
    "ask": "ask",
    "buy": "bid",
    "sell": "ask",
    "b": "bid",
    "a": "ask",
}
# Trade prints label the *taker* side (buy/sell), not a book side.
_TRADE_SIDE_MAP: dict[str, str] = {
    "buy": "buy",
    "sell": "sell",
    "b": "buy",
    "s": "sell",
    "bid": "sell",  # a trade that hit the bid is seller-initiated
    "ask": "buy",  # a trade that lifted the ask is buyer-initiated
}


def _first_present(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    """Return the first of *candidates* present in *columns* (or ``None``)."""
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _to_datetime(series: pd.Series, unit: str) -> pd.Series:
    """Parse a timestamp column onto the shared tz-aware UTC nanosecond clock.

    Integer-epoch columns go through :func:`epoch_to_datetime`; string columns
    are parsed with ``utc=True``, so a zone-carrying string is converted to UTC
    and a zone-less one is read as UTC (issue #154).
    """
    if pd.api.types.is_numeric_dtype(series):
        return epoch_to_datetime(series, unit)
    return pd.to_datetime(series, utc=True).astype(UTC_NS_DTYPE)


# ── L2DepthLoader ─────────────────────────────────────────────────────


class L2DepthLoader:
    """Load a price-level depth stream into the canonical depth frame.

    Satisfies the :class:`~ob_analytics.protocols.DepthSource` protocol.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration.  ``price_divisor`` scales the raw feed price to
        the quote currency and ``tick_size`` quantises it to integer ticks
        (issue #155); ``volume_decimals`` rounds size; ``timestamp_unit``
        interprets integer-epoch timestamps.
    venue, symbol : str, optional
        Optional instrument identity (issue #147).  When either is supplied,
        the loaded depth frame gains per-row ``venue`` / ``symbol`` columns.
        A generic price-level CSV carries no venue of its own, so ``venue`` is
        left NA unless supplied.  Both ``None`` (the default) leaves the frame
        untagged.
    """

    #: Depth-file basenames tried when *source* is a directory (in order).
    _DEPTH_NAMES: tuple[str, ...] = ("depth.csv", "depth_updates.csv", "l2.csv")

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        venue: str | None = None,
        symbol: str | None = None,
    ) -> None:
        self._config = config or PipelineConfig()
        self._venue = venue
        self._symbol = symbol

    def load(self, source: str | Path) -> pd.DataFrame:
        """Read *source* and return a canonical depth DataFrame.

        Parameters
        ----------
        source : str or Path
            The L2 CSV file, or a directory containing ``depth.csv``.

        Returns
        -------
        pandas.DataFrame
            Columns ``timestamp``, ``price``, ``volume`` (absolute level
            size), ``direction`` (categorical ``bid``/``ask``), sorted by
            ``timestamp`` — a :func:`~ob_analytics.schemas.validate_depth_df`
            frame.
        """
        path = self._resolve_depth_file(source)
        logger.info("L2DepthLoader: reading {}", path)
        raw = pd.read_csv(path)
        validate_non_empty(raw, "L2DepthLoader.load")

        ts_col = _first_present(raw.columns, _TIMESTAMP_COLUMNS)
        price_col = _first_present(raw.columns, ("price",))
        side_col = _first_present(raw.columns, _SIDE_COLUMNS)
        vol_col = _first_present(raw.columns, _VOLUME_COLUMNS)
        required = {
            "timestamp": ts_col,
            "price": price_col,
            "side": side_col,
            "volume": vol_col,
        }
        missing = [name for name, col in required.items() if col is None]
        if missing:
            raise ConfigError(
                f"L2DepthLoader.load: missing required columns {missing}. "
                f"Need a timestamp, price, side ({'/'.join(_SIDE_COLUMNS)}), and "
                f"volume ({'/'.join(_VOLUME_COLUMNS)}). "
                f"Present: {list(raw.columns)}"
            )

        cfg = self._config
        timestamp = _to_datetime(raw[ts_col], cfg.timestamp_unit)
        # Canonical price is integer ticks (issue #155): scale the raw feed
        # price to the quote currency, then quantise to ticks.
        price = price_to_ticks(
            raw[price_col].astype(float) / cfg.price_divisor, cfg.tick_size
        )
        volume = raw[vol_col].astype(float).round(cfg.volume_decimals)
        direction = (
            raw[side_col].astype(str).str.strip().str.lower().map(_SIDE_TO_DIRECTION)
        )

        depth = pd.DataFrame(
            {
                "timestamp": timestamp.array,
                "price": price,
                "volume": volume.to_numpy(),
                "direction": direction.to_numpy(),
            }
        )
        # Carry the venue sequence (nullable Int64) when the capture recorded
        # one and tracking is on; ``raw`` shares ``depth``'s row index, so the
        # column stays aligned through the filter/sort below.
        if cfg.track_sequence:
            seq_col = _first_present(raw.columns, _SEQUENCE_COLUMNS)
            if seq_col is not None:
                depth[SEQUENCE_COLUMN] = raw[seq_col].astype("Int64")

        unknown = depth["direction"].isna()
        if unknown.any():
            bad = sorted(raw.loc[unknown.to_numpy(), side_col].astype(str).unique())
            raise ConfigError(
                f"L2DepthLoader.load: unrecognised side value(s) {bad}; "
                f"expected one of {sorted(_SIDE_TO_DIRECTION)}."
            )

        # Absolute-size semantics: a negative level size is malformed.
        depth = depth[depth["volume"] >= 0]
        depth["direction"] = pd.Categorical(
            depth["direction"], categories=["bid", "ask"], ordered=True
        )
        depth = depth.sort_values("timestamp", kind="stable").reset_index(drop=True)

        # Local monotonic ingest counter over the returned (timestamp-sorted)
        # rows — a deterministic order key (opt-in, so the default frame is
        # unchanged).
        if cfg.track_sequence:
            depth = attach_ingest_seq(depth)

        logger.info(
            "L2DepthLoader: {} price-level updates ({} bid, {} ask)",
            len(depth),
            int((depth["direction"] == "bid").sum()),
            int((depth["direction"] == "ask").sum()),
        )

        # Optional per-row instrument identity (issue #147); a no-op unless a
        # venue/symbol was supplied for the run.  A generic price-level CSV has
        # no venue of its own, so there is no source-name default.
        depth = attach_instrument_identity(
            depth, venue=self._venue, symbol=self._symbol
        )
        return depth

    def _resolve_depth_file(self, source: str | Path) -> Path:
        p = Path(source)
        if p.is_file():
            return p
        if p.is_dir():
            for name in self._DEPTH_NAMES:
                candidate = p / name
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(
                f"L2DepthLoader: no depth file ({' / '.join(self._DEPTH_NAMES)}) "
                f"found in {p}."
            )
        raise FileNotFoundError(f"L2DepthLoader: path does not exist: {source}")


# ── L2TradeReader ─────────────────────────────────────────────────────


class L2TradeReader:
    """Read raw trade prints from a companion ``trades.csv``.

    Projects each print into the canonical trades schema.  Price-level feeds
    have no order IDs, so ``maker`` / ``taker`` (and their event-id / og
    columns) are left unset.  ``direction`` (the taker's aggressor side) is
    taken from a native ``side`` column when present; otherwise it is left
    unlabelled and the pipeline classifies it (Lee–Ready against the
    reconstructed BBO — see :meth:`~ob_analytics.pipeline.Pipeline._ensure_trade_signs`).

    Trades are optional for an L2 run: a missing / empty ``trades.csv`` yields
    an empty trades frame (depth analytics still run).

    Satisfies the :class:`~ob_analytics.protocols.TradeSource` protocol.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    def load(self, events: pd.DataFrame, source: Any) -> pd.DataFrame:
        path = self._resolve_trades_file(source)
        if path is None:
            logger.info("L2TradeReader: no trades.csv next to {}; no trades", source)
            return empty_trades()

        raw = pd.read_csv(path)
        if raw.empty:
            return empty_trades()

        ts_col = _first_present(raw.columns, _TIMESTAMP_COLUMNS)
        price_col = _first_present(raw.columns, ("price",))
        vol_col = _first_present(raw.columns, _VOLUME_COLUMNS)
        if ts_col is None or price_col is None or vol_col is None:
            raise ConfigError(
                "L2TradeReader.load: trades.csv needs timestamp, price, and a "
                f"volume ({'/'.join(_VOLUME_COLUMNS)}) column. "
                f"Present: {list(raw.columns)}"
            )

        cfg = self._config
        timestamp = _to_datetime(raw[ts_col], cfg.timestamp_unit)
        # Trade prints share the depth grid: integer ticks (issue #155).
        price = price_to_ticks(
            raw[price_col].astype(float) / cfg.price_divisor, cfg.tick_size
        )
        volume = raw[vol_col].astype(float).round(cfg.volume_decimals)
        direction = self._read_direction(raw)

        n = len(raw)
        na = pd.array([pd.NA] * n, dtype="object")
        trades = pd.DataFrame(
            {
                "timestamp": timestamp.array,
                "price": price,
                "volume": volume.to_numpy(),
                "direction": direction,
                # No order identity on a price-level feed.
                "maker_event_id": na,
                "taker_event_id": na.copy(),
                "maker": na.copy(),
                "taker": na.copy(),
                "maker_og": na.copy(),
                "taker_og": na.copy(),
            }
        )
        trades = trades.sort_values("timestamp", kind="stable").reset_index(drop=True)
        labelled = int(trades["direction"].notna().sum())
        logger.info(
            "L2TradeReader: {} trades ({})",
            len(trades),
            f"{labelled} natively signed"
            if labelled
            else "unsigned — pipeline classifies",
        )
        return trades

    @staticmethod
    def _read_direction(raw: pd.DataFrame) -> pd.Categorical:
        """Taker side per trade: native ``side`` mapped to buy/sell, else NA."""
        side_col = _first_present(raw.columns, _SIDE_COLUMNS)
        n = len(raw)
        if side_col is None:
            values: list[str | float] = [np.nan] * n
        else:
            values = (
                raw[side_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(_TRADE_SIDE_MAP)
                .to_list()
            )
        return pd.Categorical(values, categories=["buy", "sell"], ordered=True)

    @staticmethod
    def _resolve_trades_file(source: Any) -> Path | None:
        p = Path(source)
        base = p.parent if p.is_file() else p
        candidate = base / "trades.csv"
        return candidate if candidate.exists() else None


# ── DepthCsvWriter ────────────────────────────────────────────────────


class DepthCsvWriter:
    """Write a depth frame (and trades) back to the L2 CSV schema.

    Round-trips :class:`L2DepthLoader` / :class:`L2TradeReader`: writes
    ``depth.csv`` (and a companion ``trades.csv`` when trades are supplied)
    into the *dest* directory.

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
        """Write ``data['depth']`` (and optional ``data['trades']``) to *dest*.

        Parameters
        ----------
        data : dict of str to DataFrame
            Must contain a ``"depth"`` key (the canonical depth frame).  An
            optional ``"trades"`` key triggers a companion ``trades.csv``.
        dest : str or Path
            Output **directory**.

        Returns
        -------
        Path
            The written ``depth.csv`` path.
        """
        if "depth" not in data:
            raise ConfigError("DepthCsvWriter.write: data has no 'depth' frame.")
        cfg = self._config
        out_dir = Path(dest)
        out_dir.mkdir(parents=True, exist_ok=True)

        depth = data["depth"]
        validate_columns(
            depth, {"timestamp", "price", "volume", "direction"}, "DepthCsvWriter.write"
        )
        # datetime_to_epoch accepts the canonical tz-aware UTC column directly
        # (and reads a tz-naive one as UTC), so the round-trip is exact.
        depth_out = pd.DataFrame(
            {
                "timestamp": datetime_to_epoch(depth["timestamp"], cfg.timestamp_unit),
                "side": depth["direction"].astype(str),
                # Restore the raw feed price from integer ticks (issue #155).
                "price": (
                    ticks_to_price(depth["price"], cfg.tick_size) * cfg.price_divisor
                ).round(cfg.price_decimals),
                "volume": depth["volume"],
            }
        )
        depth_path = out_dir / "depth.csv"
        depth_out.to_csv(depth_path, index=False)

        trades = data.get("trades")
        if trades is not None and not trades.empty:
            self._write_trades(trades, out_dir / "trades.csv")
        logger.info("DepthCsvWriter: wrote {} depth rows to {}", len(depth), depth_path)
        return depth_path

    def _write_trades(self, trades: pd.DataFrame, dest: Path) -> None:
        cfg = self._config
        out = pd.DataFrame(
            {
                "timestamp": datetime_to_epoch(trades["timestamp"], cfg.timestamp_unit),
                # Restore the raw feed price from integer ticks (issue #155).
                "price": (
                    ticks_to_price(trades["price"], cfg.tick_size) * cfg.price_divisor
                ).round(cfg.price_decimals),
                "amount": trades["volume"],
                "side": trades["direction"]
                .astype("object")
                .where(trades["direction"].notna(), other=pd.NA),
            }
        )
        out.to_csv(dest, index=False)


# ── DepthCsvSource descriptor ─────────────────────────────────────────


@dataclass
class DepthCsvSource:
    """The canonical L2 (price-level) CSV source — offline depth replay.

    Declares :attr:`~ob_analytics.protocols.Level.L2`, so
    :class:`~ob_analytics.pipeline.Pipeline` takes the price-level path:
    depth in, per-order stages skipped.  The shared entry point the aggregated
    venue connectors (Binance, Kalshi, Polymarket) narrow to domain modelling
    on top of, and the offline replay target for a live L2 capture (e.g. the
    ``depth.csv`` a :class:`~ob_analytics.live.ccxt_source.CcxtSource` writes).

    Conforms structurally to :class:`~ob_analytics.protocols.OfflineSource` —
    no inheritance required.
    """

    name: str = "depth_csv"
    level: Level = Level.L2
    # A price-level feed is the venue's own aggregated view: bids never rest
    # above asks, so the reconstructed book is not crossed.
    feed_type: FeedType = FeedType.MATCHED_BOOK
    # No per-source knobs; empty typed settings keep construction uniform.
    settings: SourceSettings = field(default_factory=SourceSettings)

    def create_loader(self, config: PipelineConfig, ctx: RunContext) -> DepthSource:
        return L2DepthLoader(config, venue=ctx.venue, symbol=ctx.symbol)

    def create_trade_source(
        self, config: PipelineConfig, ctx: RunContext
    ) -> TradeSource:
        return L2TradeReader(config)

    def create_writer(
        self, config: PipelineConfig | None, ctx: RunContext
    ) -> DataWriter:
        return DepthCsvWriter(config)

    def compute_depth(
        self,
        events: pd.DataFrame,
        config: Any,
        source: Any,
        ctx: RunContext,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        # Unused on the L2 path: the loader *is* the depth source, so the
        # pipeline never folds events into depth (there are no events). Present
        # only to satisfy the OfflineSource protocol; returns None like the L3
        # sources that use the standard depth stages.
        return None

    def config_defaults(self) -> dict[str, Any]:
        return {
            "tick_size": 0.01,
            "price_decimals": 2,
            "volume_decimals": 8,
            "timestamp_unit": "ms",
        }

    def required_context(self) -> list[str]:
        # Price-level captures carry their own receive timestamps.
        return []


# ── Register this source ──────────────────────────────────────────────
# Registration runs at the bottom, after ``DepthCsvSource`` is defined.
from ob_analytics.sources import register_source

register_source("depth_csv", DepthCsvSource)
