"""L2 (price-level) depth-native data format.

The counterpart to :mod:`ob_analytics.bitstamp` / :mod:`ob_analytics.lobster`
for **price-level (L2 / market-by-price) feeds** — the ones that publish
``[price, quantity]`` levels and diffs with **no order IDs** (Binance, Kalshi,
Polymarket, most CCXT sources).  ob-analytics was built around an L3 per-order
stream; this module adds the shared path those aggregated feeds route through
instead of faking per-order state (see :class:`~ob_analytics.protocols.Level`).

The symmetric set, mirroring the other formats:

* :class:`DepthCsvLoader` — reads the canonical L2 CSV schema (a snapshot
  followed by price-level updates) straight into the depth frame.  Satisfies
  :class:`~ob_analytics.protocols.DepthLoader`.
* :class:`DepthCsvTradeReader` — reads a companion ``trades.csv`` of raw prints.
  Satisfies :class:`~ob_analytics.protocols.TradeReader`.
* :class:`DepthCsvWriter` — round-trips the depth frame (and trades) back to
  the L2 CSV schema.  Satisfies :class:`~ob_analytics.protocols.DataWriter`.
* :class:`DepthCsvSource` — the ``depth_csv`` source descriptor, declaring
  :attr:`~ob_analytics.protocols.Level.L2`.

The L2 CSV schema
-----------------

One row per price-level update, each carrying the level's **new absolute**
resting size (``0`` removes the level) — *not* a signed delta.  A book
snapshot is simply the opening block of rows.  This is exactly what
:class:`~ob_analytics.depth.DepthSummaryEngine` consumes, so there is nothing
to reconstruct.

======  ==================================================================
column  meaning
======  ==================================================================
timestamp   receive time — integer epoch (see ``DepthCsvSettings.timestamp_unit``)
            or any string :func:`pandas.to_datetime` understands
exchange_timestamp  optional: the venue's time, in the same form as
            ``timestamp``.  Kept in the depth frame when present, so the
            clock checks can compare the two clocks.
direction   ``bid`` / ``ask`` (``buy`` / ``sell`` and ``b`` / ``a`` accepted)
price       price level (divided by ``DepthCsvSettings.price_divisor`` to the quote
            currency, then stored as integer ``tick_size`` counts; a price
            that is not a whole number of ticks raises ``ConfigError``)
volume      new absolute resting size at that level (``0`` = level removed)
======  ==================================================================

Column names are flexible: ``direction`` / ``side`` and ``volume`` / ``size``
/ ``amount`` / ``quantity`` are all accepted.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import Field

from ob_analytics._utils import (
    FrameFiles,
    attach_ingest_seq,
    checked_labels,
    checked_times,
    datetime_to_epoch,
    empty_trades,
    finite_floats,
    lots_to_size,
    read_csv_file,
    read_csv_parts,
    rows_counted_from,
    ticks_to_price,
    to_grid,
    validate_non_empty,
    window_labels,
)
from ob_analytics.capture_files import DEPTH_NAME, TRADES_NAME, record_format
from ob_analytics.capture_record import CURRENT_FORMAT, FORMAT_TIME_UNITS, read_format
from ob_analytics.config import InstrumentGrid, PipelineConfig, SourceSettings
from ob_analytics.exceptions import ConfigError
from ob_analytics.protocols import (
    DataWriter,
    DepthLoader,
    FeedType,
    Level,
    LoadedWindow,
    RunContext,
    TradeAttribution,
    TradeReader,
    WriteTarget,
)
from ob_analytics.schemas import (
    BOOK_SIDE_DTYPE,
    ORIGIN_COLUMN,
    SEQUENCE_COLUMN,
    TRADE_DTYPES,
    TRADE_SIDE_DTYPE,
    attach_instrument_identity,
    cast_columns,
    validate_columns,
)

__all__ = [
    "DepthCsvLoader",
    "DepthCsvSettings",
    "DepthCsvSource",
    "DepthCsvTradeReader",
    "DepthCsvWriter",
]

# ── Column-spelling tolerance ─────────────────────────────────────────

_TIMESTAMP_COLUMNS: tuple[str, ...] = ("timestamp", "time", "ts")
# The venue's own time, when the capture recorded one (every live capture
# does).  Optional: a price-level CSV with one clock has only ``timestamp``.
_EXCHANGE_TIMESTAMP_COLUMNS: tuple[str, ...] = ("exchange_timestamp",)
_SIDE_COLUMNS: tuple[str, ...] = ("direction", "side")
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


class DepthCsvSettings(SourceSettings):
    """How a price-level CSV encodes its timestamps and prices.

    The settings of :class:`DepthCsvSource`, read by :class:`DepthCsvLoader`,
    :class:`DepthCsvTradeReader` and :class:`DepthCsvWriter`.

    Attributes
    ----------
    timestamp_unit : {"ms", "us", "ns"}
        The unit of an integer epoch timestamp.  A timestamp written as a
        string is parsed as it is, whatever this says.  When it is not set,
        the unit is the one the folder's ``meta.json`` records for a capture
        (nanoseconds since capture format version 2), else ``"ms"``; the
        writer then writes nanoseconds and records the capture format.
    price_divisor : int
        What a price in the file is divided by to reach the quote currency.
        1 (the default) means it is in the quote currency already.
    """

    timestamp_unit: Literal["ms", "us", "ns"] = Field(
        default="ms", description="The unit of an integer epoch timestamp"
    )
    price_divisor: int = Field(
        default=1,
        ge=1,
        description="What a price in the file is divided by to reach the quote currency",
    )


def _csv_settings(settings: SourceSettings | None, where: str) -> DepthCsvSettings:
    """Return *settings* as :class:`DepthCsvSettings`.

    ``None`` and the empty base :class:`SourceSettings` give the defaults.
    """
    if settings is None or type(settings) is SourceSettings:
        return DepthCsvSettings()
    if not isinstance(settings, DepthCsvSettings):
        raise ConfigError(
            f"{where}: settings must be DepthCsvSettings, got "
            f"{type(settings).__name__}."
        )
    return settings


def _time_unit(fmt: DepthCsvSettings, path: Any) -> str:
    """The unit of the integer times in the file at *path*.

    The unit *fmt* sets, else the one the capture's ``meta.json`` records.
    """
    if "timestamp_unit" in fmt.model_fields_set:
        return fmt.timestamp_unit
    return read_format(path).time_unit


def _first_present(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    """Return the first of *candidates* present in *columns* (or ``None``)."""
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _to_ticks_on_grid(
    raw_price: pd.Series, cfg: PipelineConfig, price_divisor: int, where: str
) -> np.ndarray:
    """Convert a raw feed price column to integer ticks, refusing off-grid prices.

    A price-level CSV does not carry its own tick size, so a wrong one is easy
    to pass and cannot be seen afterwards; :func:`~ob_analytics._utils.to_grid`
    refuses a price that is not a whole number of ticks.
    """
    quote = finite_floats(raw_price, "price", where) / price_divisor
    return to_grid(
        quote, cfg.tick_size, kind="price", column=str(raw_price.name), where=where
    )


def _to_lots(raw_size: pd.Series, cfg: PipelineConfig, where: str) -> np.ndarray:
    """Convert a raw size column to integer lots, warning about off-grid sizes.

    A capture records its venue's size step as ``lot_size``, so a size between
    two lots means the recorded step is wrong, and the rounded size is not the
    venue's (see :func:`~ob_analytics._utils.to_grid`).
    """
    return to_grid(
        raw_size, cfg.lot_size, kind="size", column=str(raw_size.name), where=where
    )


# ── DepthCsvLoader ─────────────────────────────────────────────────────


class DepthCsvLoader:
    """Load a price-level depth stream into the canonical depth frame.

    Satisfies the :class:`~ob_analytics.protocols.DepthLoader` protocol.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration.  ``tick_size`` quantises the price to integer
        ticks and ``lot_size`` the size to integer lots.
    settings : DepthCsvSettings, optional
        How the file encodes its timestamps and prices.
    venue, symbol : str, optional
        Optional instrument identity.  When either is supplied, the loaded
        depth frame gains per-row ``venue`` / ``symbol`` columns.  A generic
        price-level CSV carries no venue of its own, so ``venue`` is left NA
        unless supplied.  Each one not supplied is taken from the capture's
        ``meta.json`` when it records one, so a replayed capture names its
        instrument.  With neither, the frame is left untagged.
    """

    #: Depth-file basenames tried when *path* is a directory (in order).
    _DEPTH_NAMES: tuple[str, ...] = (DEPTH_NAME, "depth_updates.csv", "l2.csv")

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        settings: DepthCsvSettings | None = None,
        venue: str | None = None,
        symbol: str | None = None,
    ) -> None:
        self._config = config or PipelineConfig()
        self._settings = _csv_settings(settings, "DepthCsvLoader")
        self._venue = venue
        self._symbol = symbol

    #: How many rows :meth:`load_windows` reads from the file at a time.
    read_size: int = 250_000

    def load(self, path: str | Path) -> pd.DataFrame:
        """Read *path* and return a canonical depth DataFrame.

        Parameters
        ----------
        path : str or Path
            The L2 CSV file, or a directory containing ``depth.csv``.

        Returns
        -------
        pandas.DataFrame
            Columns ``timestamp``, ``price``, ``volume`` (absolute level
            size), ``direction`` (categorical ``bid``/``ask``), sorted by
            ``timestamp`` — a :func:`~ob_analytics.schemas.validate_depth_df`
            frame.  ``exchange_timestamp`` follows ``timestamp`` when the file
            has it, and ``origin`` (which part of a capture wrote the row) is
            kept when the file has it.
        """
        file = self._resolve_depth_file(path)
        where = f"DepthCsvLoader.load({file})"
        logger.info("DepthCsvLoader: reading {}", file)
        capture_format = read_format(file)
        raw = read_csv_file(file, where)
        validate_non_empty(raw, "DepthCsvLoader.load")
        depth = self._in_time_order(self._read_rows(raw, file, where), first=0)
        logger.info(
            "DepthCsvLoader: {} price-level updates ({} bid, {} ask)",
            len(depth),
            int((depth["direction"] == "bid").sum()),
            int((depth["direction"] == "ask").sum()),
        )
        return self._with_identity(depth, capture_format)

    def load_windows(
        self,
        path: str | Path,
        windows: list[tuple[pd.Timestamp | None, pd.Timestamp | None]],
        *,
        working: Path | None = None,
    ) -> Iterator[LoadedWindow]:
        """Yield the depth rows of each time window of *path* in turn.

        The file is read :attr:`read_size` rows at a time.  Each part's rows
        are sorted into one file per window in *working*, and each window is
        then read back on its own, so no more than one part or one window is
        in memory at once.  Each window holds the rows :meth:`load` gives,
        cut at the windows and in the same order, so the file need not be in
        time order.

        Parameters
        ----------
        path : str or Path
            The L2 CSV file, or a directory containing ``depth.csv``.
        windows : list of tuple
            ``(start, end)`` per window, tz-aware UTC, ``None`` for an open
            end, covering the whole file: what
            :func:`~ob_analytics._windows.window_bounds` returns.
        working : Path, optional
            An empty folder for the files, which the caller deletes.  By
            default a temporary folder is made and deleted.

        Yields
        ------
        LoadedWindow
            The window's depth rows.  A price-level row names no order, so no
            event a trade names is given.
        """
        with ExitStack() as stack:
            if working is None:
                working = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            yield from self._load_windows(path, windows, FrameFiles(working))

    def _load_windows(
        self,
        path: str | Path,
        windows: list[tuple[pd.Timestamp | None, pd.Timestamp | None]],
        saved: FrameFiles,
    ) -> Iterator[LoadedWindow]:
        """Yield the windows of *path*, sorting its rows through *saved*."""
        file = self._resolve_depth_file(path)
        where = f"DepthCsvLoader.load({file})"
        logger.info("DepthCsvLoader: reading {} one window at a time", file)
        capture_format = read_format(file)
        template: pd.DataFrame | None = None
        for raw in read_csv_parts(file, where, self.read_size):
            with rows_counted_from(int(raw.index[0]) + 1):
                depth = self._read_rows(raw, file, where)
            if template is None:
                template = depth.iloc[:0]
            saved.split(depth, window_labels(depth["timestamp"], windows))
        if template is None:
            # No rows: read the header alone, for the error a whole read gives.
            raw = read_csv_file(file, where, nrows=0)
            validate_non_empty(raw, "DepthCsvLoader.load")

        first = 0
        for number in range(len(windows)):
            parts = saved.take(number)
            depth = pd.concat(parts) if parts else template
            assert depth is not None
            depth = self._in_time_order(depth, first=first)
            first += len(depth)
            logger.info(
                "DepthCsvLoader: window {} of {}: {} price-level updates",
                number + 1,
                len(windows),
                len(depth),
            )
            yield LoadedWindow(
                self._with_identity(depth, capture_format), np.empty(0, np.int64)
            )

    def _read_rows(self, raw: pd.DataFrame, file: Path, where: str) -> pd.DataFrame:
        """Return the depth rows of *raw*, read from *file*, in file order."""
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
                f"DepthCsvLoader.load: missing required columns {missing}. "
                f"Need a timestamp, price, side ({'/'.join(_SIDE_COLUMNS)}), and "
                f"volume ({'/'.join(_VOLUME_COLUMNS)}). "
                f"Present: {list(raw.columns)}"
            )
        assert side_col is not None and vol_col is not None

        cfg, fmt = self._config, self._settings
        unit = _time_unit(fmt, file)
        timestamp = checked_times(raw[ts_col], unit, "timestamp", where)
        # Canonical price is integer ticks (issue #155): scale the raw feed
        # price to the quote currency, then convert to ticks.
        price = _to_ticks_on_grid(raw[price_col], cfg, fmt.price_divisor, where)
        # Canonical size is integer lots (issue #226); the raw feed carries a
        # base-asset float, so convert on the way in.
        volume = pd.Series(_to_lots(raw[vol_col], cfg, where), index=raw.index)
        direction = (
            raw[side_col].astype(str).str.strip().str.lower().map(_SIDE_TO_DIRECTION)
        )

        clocks = {"timestamp": timestamp.array}
        venue_col = _first_present(raw.columns, _EXCHANGE_TIMESTAMP_COLUMNS)
        if venue_col is not None:
            clocks["exchange_timestamp"] = checked_times(
                raw[venue_col], unit, venue_col, where, required=False
            ).array
        depth = pd.DataFrame(
            {
                **clocks,
                "price": price,
                "volume": volume.to_numpy(),
                "direction": direction.to_numpy(),
            }
        )
        # Carry which part of a capture wrote each row, when the file says: the
        # clock checks leave out the opening book's rows (see ORIGIN_COLUMN).
        origin_col = _first_present(raw.columns, (ORIGIN_COLUMN,))
        if origin_col is not None:
            depth[ORIGIN_COLUMN] = raw[origin_col].astype("string").array
        # Carry the venue sequence (nullable Int64) when the capture recorded
        # one and tracking is on.
        if cfg.track_sequence:
            seq_col = _first_present(raw.columns, _SEQUENCE_COLUMNS)
            if seq_col is not None:
                depth[SEQUENCE_COLUMN] = raw[seq_col].astype("Int64").array

        unknown = depth["direction"].isna()
        if unknown.any():
            bad = sorted(raw.loc[unknown.to_numpy(), side_col].astype(str).unique())
            raise ConfigError(
                f"DepthCsvLoader.load: unrecognised side value(s) {bad}; "
                f"expected one of {sorted(_SIDE_TO_DIRECTION)}."
            )

        # Absolute-size semantics: a negative level size is malformed.
        depth = depth[depth["volume"] >= 0]
        depth["direction"] = depth["direction"].astype(BOOK_SIDE_DTYPE)
        return depth

    def _in_time_order(self, depth: pd.DataFrame, *, first: int) -> pd.DataFrame:
        """Return *depth* sorted by ``timestamp``, rows of one instant in file order.

        *first* is the number of rows that come before *depth* once the whole
        file is sorted; the ingest counter goes on from it.
        """
        depth = depth.sort_values("timestamp", kind="stable").reset_index(drop=True)
        # Local monotonic ingest counter over the returned (timestamp-sorted)
        # rows — a deterministic order key (opt-in, so the default frame is
        # unchanged).
        if self._config.track_sequence:
            depth = attach_ingest_seq(depth, start=first)
        return depth

    def _with_identity(self, depth: pd.DataFrame, capture_format: Any) -> pd.DataFrame:
        """Return *depth* with the run's instrument identity, when it has one."""
        # Optional per-row instrument identity (issue #147): the run's, else
        # what the capture recorded (issue #494).  A generic price-level CSV
        # has no venue of its own, so there is no source-name default.
        return attach_instrument_identity(
            depth,
            venue=self._venue or capture_format.venue,
            symbol=self._symbol or capture_format.symbol,
        )

    def _resolve_depth_file(self, path: str | Path) -> Path:
        p = Path(path)
        if p.is_file():
            return p
        if p.is_dir():
            for name in self._DEPTH_NAMES:
                candidate = p / name
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(
                f"DepthCsvLoader: no depth file ({' / '.join(self._DEPTH_NAMES)}) "
                f"found in {p}."
            )
        raise FileNotFoundError(f"DepthCsvLoader: path does not exist: {path}")


# ── DepthCsvTradeReader ─────────────────────────────────────────────────────


class DepthCsvTradeReader:
    """Read raw trade prints from a companion ``trades.csv``.

    Projects each print into the canonical trades schema.  Price-level feeds
    have no order IDs, so ``maker`` / ``taker`` (and their event-id / og
    columns) are left unset.  ``direction`` (the taker's aggressor side) is
    taken from a native ``direction`` column (``side`` in older files) when
    present; otherwise it is left unlabelled and the pipeline classifies it
    (Lee–Ready against the reconstructed touch — see
    :func:`~ob_analytics.trade_sign.resolve_direction`).

    Trades are optional for an L2 run: a missing / empty ``trades.csv`` yields
    an empty trades frame (depth analytics still run).

    Satisfies the :class:`~ob_analytics.protocols.TradeReader` protocol.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration: the instrument's ``tick_size`` and
        ``lot_size``.
    settings : DepthCsvSettings, optional
        How the file encodes its timestamps and prices, as for
        :class:`DepthCsvLoader`.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        settings: DepthCsvSettings | None = None,
    ) -> None:
        self._config = config or PipelineConfig()
        self._settings = _csv_settings(settings, "DepthCsvTradeReader")

    def load(self, events: pd.DataFrame, path: Any) -> pd.DataFrame:
        file = self._resolve_trades_file(path)
        if file is None:
            logger.info(
                "DepthCsvTradeReader: no trades.csv next to {}; no trades", path
            )
            return empty_trades()

        where = f"DepthCsvTradeReader.load({file})"
        raw = read_csv_file(file, where)
        if raw.empty:
            return empty_trades()

        ts_col = _first_present(raw.columns, _TIMESTAMP_COLUMNS)
        price_col = _first_present(raw.columns, ("price",))
        vol_col = _first_present(raw.columns, _VOLUME_COLUMNS)
        if ts_col is None or price_col is None or vol_col is None:
            raise ConfigError(
                "DepthCsvTradeReader.load: trades.csv needs timestamp, price, and a "
                f"volume ({'/'.join(_VOLUME_COLUMNS)}) column. "
                f"Present: {list(raw.columns)}"
            )

        cfg, fmt = self._config, self._settings
        timestamp = checked_times(
            raw[ts_col], _time_unit(fmt, file), "timestamp", where
        )
        # Trade prints share the depth grid: integer ticks (issue #155).
        price = _to_ticks_on_grid(raw[price_col], cfg, fmt.price_divisor, where)
        # Canonical size is integer lots (issue #226); the raw feed carries a
        # base-asset float, so convert on the way in.
        volume = pd.Series(_to_lots(raw[vol_col], cfg, where), index=raw.index)
        direction = self._read_direction(raw, where)

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
        trades = cast_columns(
            trades.sort_values("timestamp", kind="stable").reset_index(drop=True),
            TRADE_DTYPES,
        )
        labelled = int(trades["direction"].notna().sum())
        logger.info(
            "DepthCsvTradeReader: {} trades ({})",
            len(trades),
            f"{labelled} natively signed"
            if labelled
            else "unsigned — pipeline classifies",
        )
        return trades

    @staticmethod
    def _read_direction(raw: pd.DataFrame, where: str) -> pd.Categorical:
        """Taker side per trade: native ``direction`` mapped to buy/sell, else NA.

        An empty side is unlabelled, for the pipeline to classify; a side that
        is not one of the known spellings is refused.
        """
        side_col = _first_present(raw.columns, _SIDE_COLUMNS)
        values: list[str | float] = [np.nan] * len(raw)
        if side_col is not None:
            spelled = raw[side_col].astype("string").str.strip().str.lower()
            checked_labels(
                spelled, "side", where, allowed=_TRADE_SIDE_MAP, required=False
            )
            values = [*spelled.map(_TRADE_SIDE_MAP)]
        return pd.Categorical(values, dtype=TRADE_SIDE_DTYPE)

    @staticmethod
    def _resolve_trades_file(path: Any) -> Path | None:
        p = Path(path)
        base = p.parent if p.is_file() else p
        candidate = base / TRADES_NAME
        return candidate if candidate.exists() else None


# ── DepthCsvWriter ────────────────────────────────────────────────────


class DepthCsvWriter:
    """Write a depth frame (and trades) back to the L2 CSV schema.

    Round-trips :class:`DepthCsvLoader` / :class:`DepthCsvTradeReader`: writes
    ``depth.csv`` (and a companion ``trades.csv`` when trades are supplied)
    into the *dest* directory.

    Satisfies the :class:`~ob_analytics.protocols.DataWriter` protocol.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration: the ``tick_size`` and ``lot_size`` the frames
        are in.
    settings : DepthCsvSettings, optional
        How to encode the timestamps and prices, as the loader reads them.
    """

    target = WriteTarget.FOLDER

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        settings: DepthCsvSettings | None = None,
    ) -> None:
        self._config = config or PipelineConfig()
        self._settings = _csv_settings(settings, "DepthCsvWriter")

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
        cfg, fmt = self._config, self._settings
        out_dir = Path(dest)
        out_dir.mkdir(parents=True, exist_ok=True)
        # With no unit set, write the capture format and record it, so the
        # loader reads the folder back the same way.
        capture = "timestamp_unit" not in fmt.model_fields_set
        unit = FORMAT_TIME_UNITS[CURRENT_FORMAT] if capture else fmt.timestamp_unit

        depth = data["depth"]
        validate_columns(
            depth, {"timestamp", "price", "volume", "direction"}, "DepthCsvWriter.write"
        )
        # datetime_to_epoch accepts the canonical tz-aware UTC column directly
        # (and reads a tz-naive one as UTC), so the round-trip is exact.
        clocks = {"timestamp": datetime_to_epoch(depth["timestamp"], unit)}
        if "exchange_timestamp" in depth.columns:
            clocks["exchange_timestamp"] = datetime_to_epoch(
                depth["exchange_timestamp"], unit
            )
        # The capture's own columns, when the frame carries them: the venue
        # sequence, and which part of the capture wrote each row.
        kept = {
            column: depth[column].array
            for column in (SEQUENCE_COLUMN, ORIGIN_COLUMN)
            if column in depth.columns
        }
        depth_out = pd.DataFrame(
            {
                **clocks,
                "direction": depth["direction"].astype(str),
                # Restore the raw feed price from integer ticks (issue #155).
                "price": (
                    ticks_to_price(depth["price"], cfg.tick_size) * fmt.price_divisor
                ).round(cfg.price_decimals),
                # Restore the base-asset float from integer lots (#226).
                "volume": lots_to_size(
                    depth["volume"],
                    self._config.lot_size,
                    decimals=self._config.volume_decimals,
                ),
                **kept,
            }
        )
        depth_path = out_dir / DEPTH_NAME
        depth_out.to_csv(depth_path, index=False)

        trades = data.get("trades")
        if trades is not None and not trades.empty:
            self._write_trades(trades, out_dir / TRADES_NAME, unit)
        if capture:
            record_format(out_dir)
        logger.info("DepthCsvWriter: wrote {} depth rows to {}", len(depth), depth_path)
        return depth_path

    def _write_trades(self, trades: pd.DataFrame, dest: Path, unit: str) -> None:
        cfg, fmt = self._config, self._settings
        out = pd.DataFrame(
            {
                "timestamp": datetime_to_epoch(trades["timestamp"], unit),
                # Restore the raw feed price from integer ticks (issue #155).
                "price": (
                    ticks_to_price(trades["price"], cfg.tick_size) * fmt.price_divisor
                ).round(cfg.price_decimals),
                "volume": lots_to_size(
                    trades["volume"],
                    self._config.lot_size,
                    decimals=self._config.volume_decimals,
                ),
                "direction": trades["direction"]
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
    # A price-level feed: the venue's total size at each price.
    feed_type: FeedType = FeedType.PRICE_LEVELS
    # Price levels carry no order identity, so no order of a trade is named.
    trade_attribution: TradeAttribution = TradeAttribution.NONE
    # How the files encode timestamps and prices (DepthCsvSettings).  Declared
    # as the base type the ``Source`` protocol states, as every source does.
    settings: SourceSettings = field(default_factory=DepthCsvSettings)
    # A cent tick and satoshi lots; a capture's record gives its own.
    grid: InstrumentGrid = field(default_factory=InstrumentGrid)

    def create_loader(self, config: PipelineConfig, ctx: RunContext) -> DepthLoader:
        return DepthCsvLoader(
            config,
            settings=_csv_settings(self.settings, "DepthCsvSource"),
            venue=ctx.venue,
            symbol=ctx.symbol,
        )

    def create_trade_reader(
        self, config: PipelineConfig, ctx: RunContext, loader: Any
    ) -> TradeReader:
        return DepthCsvTradeReader(
            config, settings=_csv_settings(self.settings, "DepthCsvSource")
        )

    def create_writer(
        self, config: PipelineConfig | None, ctx: RunContext
    ) -> DataWriter:
        return DepthCsvWriter(
            config, settings=_csv_settings(self.settings, "DepthCsvSource")
        )


# ── Register this source ──────────────────────────────────────────────
# Registration runs at the bottom, after ``DepthCsvSource`` is defined.
from ob_analytics.sources import register_source

register_source("depth_csv", DepthCsvSource)
