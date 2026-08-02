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
* :class:`DepthCsvFormat` — the ``depth_csv`` format descriptor, declaring
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
price       price level (scaled by ``config.price_divisor``, rounded)
volume      new absolute resting size at that level (``0`` = level removed)
======  ==================================================================

Column names are flexible: ``side`` / ``direction`` and ``volume`` / ``size``
/ ``amount`` / ``quantity`` are all accepted.
"""

from __future__ import annotations

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
from ob_analytics.exceptions import ConfigError
from ob_analytics.protocols import (
    DataWriter,
    DepthSource,
    FeedType,
    Level,
    RunContext,
    TradeSource,
)

# ── Column-spelling tolerance ─────────────────────────────────────────

_TIMESTAMP_COLUMNS: tuple[str, ...] = ("timestamp", "time", "ts")
_SIDE_COLUMNS: tuple[str, ...] = ("side", "direction")
_VOLUME_COLUMNS: tuple[str, ...] = ("volume", "size", "amount", "quantity")
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
    """Parse a timestamp column: integer epoch (*unit*) or parseable strings."""
    if pd.api.types.is_numeric_dtype(series):
        return epoch_to_datetime(series, unit)
    return pd.to_datetime(series).astype("datetime64[ns]")


# ── L2DepthLoader ─────────────────────────────────────────────────────


class L2DepthLoader:
    """Load a price-level depth stream into the canonical depth frame.

    Satisfies the :class:`~ob_analytics.protocols.DepthSource` protocol.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration.  ``price_decimals`` / ``price_divisor`` /
        ``volume_decimals`` control price scaling and rounding;
        ``timestamp_unit`` interprets integer-epoch timestamps.
    """

    #: Depth-file basenames tried when *source* is a directory (in order).
    _DEPTH_NAMES: tuple[str, ...] = ("depth.csv", "depth_updates.csv", "l2.csv")

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

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
        price = (raw[price_col].astype(float) / cfg.price_divisor).round(
            cfg.price_decimals
        )
        volume = raw[vol_col].astype(float).round(cfg.volume_decimals)
        direction = (
            raw[side_col].astype(str).str.strip().str.lower().map(_SIDE_TO_DIRECTION)
        )

        depth = pd.DataFrame(
            {
                "timestamp": timestamp.to_numpy(),
                "price": price.to_numpy(),
                "volume": volume.to_numpy(),
                "direction": direction.to_numpy(),
            }
        )

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

        logger.info(
            "L2DepthLoader: {} price-level updates ({} bid, {} ask)",
            len(depth),
            int((depth["direction"] == "bid").sum()),
            int((depth["direction"] == "ask").sum()),
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
        price = (raw[price_col].astype(float) / cfg.price_divisor).round(
            cfg.price_decimals
        )
        volume = raw[vol_col].astype(float).round(cfg.volume_decimals)
        direction = self._read_direction(raw)

        n = len(raw)
        na = pd.array([pd.NA] * n, dtype="object")
        trades = pd.DataFrame(
            {
                "timestamp": timestamp.to_numpy(),
                "price": price.to_numpy(),
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
        # datetime_to_epoch reads an int64 ns count; normalise the resolution
        # so a datetime64[ms] frame (e.g. the toy fixture) round-trips exactly.
        ts_ns = depth["timestamp"].astype("datetime64[ns]")
        depth_out = pd.DataFrame(
            {
                "timestamp": datetime_to_epoch(ts_ns, cfg.timestamp_unit),
                "side": depth["direction"].astype(str),
                "price": (depth["price"] * cfg.price_divisor).round(cfg.price_decimals),
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
        ts_ns = trades["timestamp"].astype("datetime64[ns]")
        out = pd.DataFrame(
            {
                "timestamp": datetime_to_epoch(ts_ns, cfg.timestamp_unit),
                "price": (trades["price"] * cfg.price_divisor).round(
                    cfg.price_decimals
                ),
                "amount": trades["volume"],
                "side": trades["direction"]
                .astype("object")
                .where(trades["direction"].notna(), other=pd.NA),
            }
        )
        out.to_csv(dest, index=False)


# ── DepthCsvFormat descriptor ─────────────────────────────────────────


@dataclass
class DepthCsvFormat:
    """Format descriptor for the canonical L2 (price-level) CSV schema.

    Declares :attr:`~ob_analytics.protocols.Level.L2`, so
    :class:`~ob_analytics.pipeline.Pipeline` takes the price-level path:
    depth in, per-order stages skipped.  The shared entry point the aggregated
    venue connectors (Binance, Kalshi, Polymarket) narrow to domain modelling
    on top of.

    Conforms structurally to the :class:`~ob_analytics.protocols.Format`
    Protocol — no inheritance required.
    """

    name: str = "depth_csv"
    resolution: Level = Level.L2
    # A price-level feed is the venue's own aggregated view: bids never rest
    # above asks, so the reconstructed book is not crossed.
    feed_type: FeedType = FeedType.MATCHED_BOOK

    def create_loader(self, config: PipelineConfig, ctx: RunContext) -> DepthSource:
        return L2DepthLoader(config)

    def create_trade_source(
        self, config: PipelineConfig, ctx: RunContext
    ) -> TradeSource:
        return L2TradeReader(config)

    def create_writer(self, config: PipelineConfig, ctx: RunContext) -> DataWriter:
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
        # only to satisfy the Format protocol; returns None like the L3 formats
        # that use the standard depth stages.
        return None

    def config_defaults(self) -> dict[str, Any]:
        return {
            "price_decimals": 2,
            "volume_decimals": 8,
            "timestamp_unit": "ms",
        }

    def required_context(self) -> list[str]:
        # Price-level captures carry their own receive timestamps.
        return []


# ── Register this format and its writer ───────────────────────────────
# Imports sit at the bottom (deferred from the top of the module) to avoid a
# circular import: ``pipeline`` imports the format modules to self-register.
from ob_analytics.data import register_writer
from ob_analytics.pipeline import register_format

register_format("depth_csv", DepthCsvFormat)
register_writer("depth_csv", lambda config, ctx: DepthCsvWriter(config))
