"""The capture folder: its file names, their columns, and how they are written.

A live capture writes one folder (one per segment of a long capture):

* ``orders.csv`` -- L3 only: one row per order event, read by
  :class:`~ob_analytics.bitstamp.BitstampLoader`;
* ``depth.csv`` -- L2 only: one row per price-level change, read by
  :class:`~ob_analytics.depth_l2.L2DepthLoader`;
* ``trades.csv`` -- one row per trade, for both levels;
* ``raw.jsonl`` -- every frame the venue sent, one JSON object per line,
  when the capture keeps them;
* ``meta.json`` -- the capture record (see :mod:`ob_analytics.capture_record`).

This module owns the file names and columns, the sink that writes them
(:class:`FileCaptureSink`), and the repair of a folder a capture never closed
(:func:`repair_files`).  The writers that save a run in the same format
(:class:`~ob_analytics.bitstamp.BitstampWriter`,
:class:`~ob_analytics.depth_l2.DepthCsvWriter`) take their names and columns
from here too.

Like :mod:`ob_analytics.capture_record`, it is outside
:mod:`ob_analytics.live`, so code that only replays files does not import the
live-capture package.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import date, time
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

from ob_analytics._secrets import RedactingFile, any_known, redact
from ob_analytics.capture_record import write_record
from ob_analytics.protocols import Level
from ob_analytics.schemas import ORIGIN_COLUMN, SEQUENCE_COLUMN, SNAPSHOT_ORIGIN

if TYPE_CHECKING:
    from ob_analytics.live._base import CaptureResult, EventDict

#: The L3 book file: one row per order event.
ORDERS_NAME = "orders.csv"
#: The L2 book file: one row per price-level change.
DEPTH_NAME = "depth.csv"
#: The trades file, written at both levels.
TRADES_NAME = "trades.csv"
#: Every frame the venue sent, one JSON object per line.
RAW_NAME = "raw.jsonl"
#: The book files, in the order a reader looks for them.
BOOK_NAMES = (ORDERS_NAME, DEPTH_NAME)

#: Which part of the capture wrote a book row (the :data:`ORIGIN_COLUMN`):
#: the source's opening book, a live message, or a close-out at the end.  The
#: runner stamps it from the phase it is in, so no source has to, and a row's
#: origin never has to be guessed from its timestamp.  A source whose opening
#: book arrives through the stream (cryptofeed) marks those rows itself, and
#: the mark is kept.
ORIGIN_SNAPSHOT = SNAPSHOT_ORIGIN
ORIGIN_STREAM = "stream"
ORIGIN_SHUTDOWN = "shutdown"

#: The columns of ``orders.csv``.  ``sequence`` is the venue's own number for
#: the event when the source gives one (blank otherwise);
#: :class:`~ob_analytics.bitstamp.BitstampLoader` reads it back for gap
#: detection.
ORDER_COLUMNS: tuple[str, ...] = (
    "id",
    "timestamp",
    "exchange_timestamp",
    "price",
    "volume",
    "action",
    "direction",
    SEQUENCE_COLUMN,
    ORIGIN_COLUMN,
)
#: The columns of ``trades.csv``.
TRADE_COLUMNS: tuple[str, ...] = (
    "trade_id",
    "timestamp",
    "exchange_timestamp",
    "price",
    "amount",
    "buy_order_id",
    "sell_order_id",
    "side",
)
#: The columns of ``depth.csv``.  ``volume`` is the new size at ``price`` (0
#: removes the level).  ``sequence`` is the venue's number for the book when
#: the source gives one (blank otherwise).  ``timestamp`` is the receive time
#: a replay sorts on; ``exchange_timestamp`` keeps the venue's time.
DEPTH_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "exchange_timestamp",
    "side",
    "price",
    "volume",
    SEQUENCE_COLUMN,
    ORIGIN_COLUMN,
)

#: Each CSV file of a capture with its columns.
CSV_FILES: dict[str, tuple[str, ...]] = {
    ORDERS_NAME: ORDER_COLUMNS,
    DEPTH_NAME: DEPTH_COLUMNS,
    TRADES_NAME: TRADE_COLUMNS,
}


def book_file(folder: Path) -> Path | None:
    """The book file in a capture *folder*: ``orders.csv`` or ``depth.csv``.

    ``None`` when the folder has neither.
    """
    for name in BOOK_NAMES:
        if (folder / name).is_file():
            return folder / name
    return None


# ---------------------------------------------------------------------------
# Writing a capture
# ---------------------------------------------------------------------------


def _ts_ms(ts: pd.Timestamp | float) -> int:
    if isinstance(ts, pd.Timestamp):
        return int(ts.value // 1_000_000)
    return int(ts)


class FileCaptureSink:
    """Default sink: writes the book file, trades.csv, raw.jsonl, meta.json.

    It satisfies :class:`~ob_analytics.live.CaptureSink`.  The book file
    follows the source's :class:`~ob_analytics.protocols.Level`: **L3** ->
    ``orders.csv`` (one row per order event); **L2** -> ``depth.csv`` (one row
    per price-level change).  ``trades.csv`` is written for both.

    When a source's settings hold an API key, every file passes through
    :func:`ob_analytics._secrets.redact`, so the key is not written even if a
    frame or an error message carries it.  The keys are registered before the
    sink is built (:func:`~ob_analytics.live._runner.check_credentials`); with
    none, the files are plain.
    """

    def __init__(
        self,
        out_dir: Path,
        *,
        keep_raw: bool,
        level: Level = Level.L3,
        raw_warned: set[str] | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._keep_raw = keep_raw
        self._level = level

        # Exactly one book writer is opened, per resolution. The other stays
        # None so write_order / write_depth are safe no-ops on the wrong side.
        self._orders_fp: Any = None
        self._orders: csv.DictWriter[str] | None = None
        self._depth_fp: Any = None
        self._depth: csv.DictWriter[str] | None = None
        if level is Level.L2:
            self._depth_fp = _open_redacting(self.out_dir / DEPTH_NAME)
            self._depth = csv.DictWriter(
                self._depth_fp, fieldnames=DEPTH_COLUMNS, extrasaction="ignore"
            )
            self._depth.writeheader()
        else:
            self._orders_fp = _open_redacting(self.out_dir / ORDERS_NAME)
            self._orders = csv.DictWriter(
                self._orders_fp, fieldnames=ORDER_COLUMNS, extrasaction="ignore"
            )
            self._orders.writeheader()

        self._trades_fp = _open_redacting(self.out_dir / TRADES_NAME)
        self._trades = csv.DictWriter(
            self._trades_fp, fieldnames=TRADE_COLUMNS, extrasaction="ignore"
        )
        self._trades.writeheader()

        self._raw_fp = _open_redacting(self.out_dir / RAW_NAME) if keep_raw else None
        # raw.jsonl is a record for debugging, so what it cannot hold never
        # stops the capture. The types it wrote as str(value) and the frames it
        # skipped go to meta.json. The warnings already logged are shared by
        # the caller across a capture's segments, so each is logged once.
        self._raw_text_types: set[str] = set()
        self._raw_skipped = 0
        self._raw_warned = set() if raw_warned is None else raw_warned
        # The types the frame being encoded wrote as text. They join
        # _raw_text_types only once the frame is written.
        self._frame_text_types: set[str] = set()
        self._raw_type_names: dict[type, str] = {}
        # Whether the last frame had a dict key JSON cannot hold. A source that
        # sends one usually sends them in every frame, so the next frame goes
        # straight to _encode instead of failing json.dumps first. A frame
        # with no such key turns it off, since _encode is slower.
        self._raw_keys_as_text = False
        self._frame_had_text_key = False
        #: The first error closing a file, set by :meth:`finalize`.
        self.close_error: str | None = None

    def write_order(self, event: EventDict) -> None:
        if self._orders is None:
            return
        row = {
            **event,
            "timestamp": _ts_ms(event["timestamp"]),
            "exchange_timestamp": _ts_ms(event["exchange_timestamp"]),
        }
        self._orders.writerow(row)

    def write_depth(self, event: EventDict) -> None:
        if self._depth is None:
            return
        # Only the DEPTH_COLUMNS are persisted (extras ignored).
        row = {
            **event,
            "timestamp": _ts_ms(event["timestamp"]),
            "exchange_timestamp": _ts_ms(event["exchange_timestamp"]),
        }
        self._depth.writerow(row)

    def write_trade(self, event: EventDict) -> None:
        row = {
            **event,
            "timestamp": _ts_ms(event["timestamp"]),
            "exchange_timestamp": _ts_ms(event["exchange_timestamp"]),
        }
        self._trades.writerow(row)

    def write_raw(self, frame: Any) -> None:
        if self._raw_fp is None or frame is None:
            return
        self._frame_text_types.clear()
        self._frame_had_text_key = False
        try:
            if self._raw_keys_as_text:
                line = self._encode(frame, set())
                self._raw_keys_as_text = self._frame_had_text_key
            else:
                try:
                    line = json.dumps(
                        frame, separators=(",", ":"), default=self._raw_default
                    )
                except TypeError:
                    # json.dumps never passes a dict key to default=, so a key
                    # that is not a str, int, float, bool or None raises.
                    self._frame_text_types.clear()
                    line = self._encode(frame, set())
                    self._raw_keys_as_text = self._frame_had_text_key
        except Exception as exc:  # noqa: BLE001 - one bad frame must not kill the run
            # A circular reference, or a value whose str() raises: JSON cannot
            # hold the frame at all.
            self._raw_skipped += 1
            kind = type(exc).__name__
            self._warn_once(
                f"skip:{kind}",
                f"raw.jsonl: skipping frames JSON cannot hold ({kind}: {exc})",
            )
            return
        self._raw_fp.write(line + "\n")
        for name in self._frame_text_types:
            self._raw_text_types.add(name)
            self._warn_once(
                f"text:{name}",
                f"raw.jsonl: writing {name} values or keys as text (str(value))",
            )

    def _encode(self, value: Any, path: set[int]) -> str:
        """Encode *value* as ``json.dumps`` would, writing any dict key as text.

        A key is written the way :meth:`_raw_default` writes a value. Keys
        that turn into the same text are all kept, as ``json.dumps`` keeps
        both ``1`` and ``"1"``. *path* holds the containers above *value*, so
        a circular reference raises ``ValueError`` as ``json.dumps`` would.
        """
        if not isinstance(value, dict | list | tuple):
            return json.dumps(value, default=self._raw_default)
        if id(value) in path:
            raise ValueError("Circular reference detected")
        path.add(id(value))
        try:
            if isinstance(value, dict):
                return (
                    "{"
                    + ",".join(
                        f"{json.dumps(self._key_text(k))}:{self._encode(v, path)}"
                        for k, v in value.items()
                    )
                    + "}"
                )
            return "[" + ",".join(self._encode(v, path) for v in value) + "]"
        finally:
            path.discard(id(value))

    def _key_text(self, key: Any) -> str:
        # As json.dumps writes a key: a str as it is, None, a bool or a number
        # as its JSON text.
        if isinstance(key, str):
            return key
        if key is None or isinstance(key, bool | int | float):
            return json.dumps(key)
        self._frame_had_text_key = True
        return str(self._raw_default(key))

    def _raw_default(self, value: Any) -> Any:
        # Some feeds parse prices and sizes as Decimal (cryptofeed does for
        # Bitstamp trades). Write them as strings so raw.jsonl keeps every digit.
        if isinstance(value, Decimal):
            return str(value)
        # cryptofeed parses most venues' frames with yapic json, which turns ISO
        # date and time strings into date, datetime and time objects
        # (independent_reserve, blockchain). Write them back as ISO 8601
        # strings. datetime is a subclass of date.
        if isinstance(value, (date, time)):
            return value.isoformat()
        kind = type(value)
        name = self._raw_type_names.get(kind)
        if name is None:
            name = self._raw_type_names[kind] = f"{kind.__module__}.{kind.__qualname__}"
        self._frame_text_types.add(name)
        return str(value)

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._raw_warned:
            self._raw_warned.add(key)
            logger.warning("{}", message)

    @property
    def raw_frames_skipped(self) -> int:
        """Frames raw.jsonl skipped because JSON cannot hold them."""
        return self._raw_skipped

    def raw_diagnostics(self) -> dict[str, Any]:
        """What raw.jsonl wrote as text or skipped, as ``meta.json`` fields.

        Empty when raw.jsonl is off.
        """
        if not self._keep_raw:
            return {}
        return {
            "raw_text_types": sorted(self._raw_text_types),
            "n_raw_frames_skipped": self._raw_skipped,
        }

    def flush(self) -> None:
        """Push buffered rows to disk, so a crash loses as few as possible."""
        for fp in (self._orders_fp, self._depth_fp, self._trades_fp, self._raw_fp):
            if fp is not None:
                fp.flush()

    def bytes_written(self) -> int:
        """Bytes written so far across every open file (what a size roll reads)."""
        return sum(
            fp.tell()
            for fp in (self._orders_fp, self._depth_fp, self._trades_fp, self._raw_fp)
            if fp is not None
        )

    def finalize(self, result: CaptureResult) -> None:
        """Close every file, then write ``meta.json`` last.

        Each file is closed on its own, so one that fails to flush (a full
        disk, say) leaves no other open.  The first failure is logged and
        recorded in ``meta.json`` as ``capture_error`` with the phase
        ``close``, unless the run already had an error, and kept in
        :attr:`close_error`.
        """
        close_error: str | None = None
        for fp in (self._orders_fp, self._depth_fp, self._trades_fp, self._raw_fp):
            if fp is None:
                continue
            try:
                try:
                    fp.flush()
                finally:
                    fp.close()
            except Exception as exc:  # noqa: BLE001 - recorded in meta.json
                error = redact(repr(exc))
                logger.error("Closing {} failed: {}", getattr(fp, "name", fp), error)
                close_error = close_error or error
        self._orders_fp = None
        self._depth_fp = None
        self._trades_fp = None  # type: ignore[assignment]
        self._raw_fp = None
        self.close_error = close_error

        meta: dict[str, Any] = {
            "out_dir": str(result.out_dir),
            "started": str(result.started),
            "ended": str(result.ended),
            "duration_seconds": (result.ended - result.started).total_seconds(),
            "n_order_events": result.n_order_events,
            "n_depth_events": result.n_depth_events,
            "n_trade_events": result.n_trade_events,
            "n_raw_frames": result.n_raw_frames,
            "n_snapshot_unconfirmed": result.n_snapshot_unconfirmed,
            "stream_started": _iso_or_none(result.stream_started),
            "stream_ended": _iso_or_none(result.stream_ended),
            **result.extras,
            **self.raw_diagnostics(),
        }
        if result.capture_error is not None:
            meta["capture_error"] = result.capture_error
            meta["capture_error_phase"] = result.capture_error_phase
        elif close_error is not None:
            meta["capture_error"] = close_error
            meta["capture_error_phase"] = "close"
        write_record(self.out_dir, meta)


def _open_redacting(path: Path) -> Any:
    """Open *path* for writing text, redacting every write when a key is loaded."""
    fp = path.open("w", newline="")
    return RedactingFile(fp) if any_known() else fp


def _iso_or_none(ts: pd.Timestamp | None) -> str | None:
    return None if ts is None else str(ts)


# ---------------------------------------------------------------------------
# Repairing a capture a crash left open
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepairedFiles:
    """What :func:`repair_files` found and wrote.

    Attributes
    ----------
    last_ms : int or None
        The latest receive ``timestamp`` in any CSV file, in epoch
        milliseconds; ``None`` when they hold no rows.
    orders_closed : int
        The ``deleted`` rows written for orders still open.
    n_order_events, n_depth_events, n_trade_events : int
        The rows each file holds after the repair.
    """

    last_ms: int | None
    orders_closed: int
    n_order_events: int
    n_depth_events: int
    n_trade_events: int


def repair_files(folder: Path) -> RepairedFiles:
    """Complete the files of a capture *folder* that its capture never closed.

    The last line of each file may be cut short, or, if the capture stopped
    before its first flush, a file may be empty.  This cuts each file back to
    its last whole line, puts back a missing header, and closes every L3 order
    still open at the last recorded time, with the same ``deleted`` rows a
    normal shutdown writes.  It does not touch ``meta.json``.
    """
    for name in (*CSV_FILES, RAW_NAME):
        _trim_to_last_line(folder / name)
    for name, columns in CSV_FILES.items():
        _restore_header(folder / name, columns)

    stamps = [
        ms
        for name in CSV_FILES
        if (ms := _last_timestamp_ms(folder / name)) is not None
    ]
    last_ms = max(stamps, default=None)
    orders = folder / ORDERS_NAME
    closed = 0
    if orders.is_file() and last_ms is not None:
        closed = _close_open_orders(orders, last_ms)
    return RepairedFiles(
        last_ms=last_ms,
        orders_closed=closed,
        n_order_events=_count_rows(orders),
        n_depth_events=_count_rows(folder / DEPTH_NAME),
        n_trade_events=_count_rows(folder / TRADES_NAME),
    )


_TRIM_BLOCK = 64 * 1024


def _trim_to_last_line(path: Path) -> None:
    """Cut *path* back to its last complete line (a crash can leave half of one)."""
    if not path.is_file():
        return
    with path.open("r+b") as fp:
        end = fp.seek(0, os.SEEK_END)
        if end == 0:
            return
        fp.seek(end - 1)
        if fp.read(1) == b"\n":
            return
        # Read back from the end a block at a time: raw.jsonl can be many GB.
        pos = end
        while pos > 0:
            start = max(0, pos - _TRIM_BLOCK)
            fp.seek(start)
            cut = fp.read(pos - start).rfind(b"\n")
            if cut >= 0:
                fp.truncate(start + cut + 1)
                return
            pos = start
        fp.truncate(0)


def _restore_header(path: Path, columns: tuple[str, ...]) -> None:
    """Write the header into a capture CSV the crash left empty."""
    if path.is_file() and path.stat().st_size == 0:
        path.write_text(",".join(columns) + "\n")


def _count_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="") as fp:
        return max(sum(1 for _ in fp) - 1, 0)  # minus the header


def _last_timestamp_ms(path: Path) -> int | None:
    """The latest receive ``timestamp`` in a capture CSV, in epoch ms."""
    if not path.is_file():
        return None
    latest: int | None = None
    with path.open(newline="") as fp:
        for row in csv.DictReader(fp):
            try:
                ts = int(row["timestamp"])
            except (KeyError, TypeError, ValueError):
                continue
            if latest is None or ts > latest:
                latest = ts
    return latest


def _close_open_orders(orders: Path, at_ms: int) -> int:
    """Append a ``deleted`` row at *at_ms* for every order still open."""
    open_orders: dict[str, dict[str, str]] = {}
    with orders.open(newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = reader.fieldnames or []
        for row in reader:
            if row.get("action") == "deleted":
                open_orders.pop(row["id"], None)
            else:
                open_orders[row["id"]] = row
    if not open_orders:
        return 0
    with orders.open("a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        for row in open_orders.values():
            writer.writerow(
                {
                    **row,
                    "timestamp": at_ms,
                    "exchange_timestamp": at_ms,
                    "action": "deleted",
                    SEQUENCE_COLUMN: "",
                    ORIGIN_COLUMN: ORIGIN_SHUTDOWN,
                }
            )
    return len(open_orders)
