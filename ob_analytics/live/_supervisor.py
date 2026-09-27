"""Run a live capture for days: segments, reconnects, rolls and restarts.

:func:`run_capture` drives :func:`~ob_analytics.live._runner.run_capturer`
once per segment (see :mod:`ob_analytics.live._manifest` for the layout).
Every break in a capture is handled the same way -- the current segment is
closed and a new one starts from a fresh snapshot:

* **Roll** (``roll_minutes`` / ``roll_mb``): the next segment starts while the
  current one still runs, and the current one stops once the next is
  streaming.  The two overlap, so a roll loses nothing.
* **Disconnect**: the source raises (or stops streaming).  The segment is
  closed with its closing rows, and after a wait that doubles on each failure
  in a row (up to :data:`BACKOFF_MAX_SECONDS`) a new segment starts.  The
  time between the two is recorded as a gap.
* **Restart**: a capture started into a directory that already has a
  ``manifest.json`` continues it.  A segment the dead process left open is
  closed first (:func:`_close_unfinished`), and the downtime is recorded as a
  gap.

Only a capture that has never streamed may stop on an error: when its first
segment fails before its first event the settings are probably wrong, and
retrying would hide that.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from loguru import logger

from ob_analytics.exceptions import ConfigError
from ob_analytics.live._base import (
    CaptureConfig,
    CaptureResult,
    LiveSource,
    SupportsDiagnostics,
    SupportsPreflight,
)
from ob_analytics.live._manifest import (
    CaptureManifest,
    EndReason,
    Segment,
    read_manifest,
)
from ob_analytics.live._runner import (
    _DEPTH_COLS,
    _ORDER_COLS,
    _TRADE_COLS,
    ORIGIN_SHUTDOWN,
    FileCaptureSink,
    _source_declarations,
    install_stop_signals,
    run_capturer,
)
from ob_analytics.protocols import Level

#: First wait after a segment fails, in seconds; it doubles on each failure in
#: a row, up to :data:`BACKOFF_MAX_SECONDS`.
BACKOFF_START_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 60.0
#: A segment that streamed at least this long resets the wait: its failure is
#: a new disconnect, not the next in a row.
BACKOFF_RESET_SECONDS = 60.0
#: How long a roll waits for the next segment's first event before it stops
#: the current one anyway.  A quiet market can take a while to send one; a
#: next segment that failed ends sooner than this.
HANDOVER_TIMEOUT_SECONDS = 120.0
#: How often running segments' files are flushed and the manifest and each
#: running segment's provisional ``meta.json`` rewritten, and so how much a
#: crash can leave unrecorded.
HEARTBEAT_SECONDS = 10.0
#: How often the supervisor checks the roll limits.
POLL_SECONDS = 1.0

SourceFactory = Callable[[], LiveSource]

#: The lock file a running capture holds in its directory.
LOCK_NAME = ".capture.lock"

# Written into a segment that a dead capture process left open.
_UNFINISHED_ERROR = "the capture process stopped without closing this segment"


@dataclass(frozen=True)
class CaptureRun:
    """What :func:`run_capture` returns.

    Attributes
    ----------
    out_dir : Path
        The capture directory.
    manifest : CaptureManifest
        The manifest as it was last written.
    error : str or None
        Why the run stopped early, when it could not start at all (the first
        snapshot failed).  ``None`` for a run that ended normally or was
        stopped by a signal -- gaps and failed segments along the way are in
        the manifest, not here.
    """

    out_dir: Path
    manifest: CaptureManifest
    error: str | None = None


@dataclass
class _Running:
    """A segment whose :func:`run_capturer` task is in flight."""

    segment: Segment
    source: LiveSource
    sink: FileCaptureSink
    stop: asyncio.Event
    streaming: asyncio.Event
    task: asyncio.Task[CaptureResult]
    opened: float  # monotonic time the segment started
    # Monotonic time it became the current segment, and the bytes it had
    # written by its first live event.  A roll counts from these, so the time
    # and the snapshot a segment spends taking over never count toward its
    # limit, and a new segment is never due to roll the moment it takes over.
    current_since: float | None = None
    bytes_at_stream: int | None = None
    # Set when the supervisor asks the segment to stop, and why.
    stop_reason: EndReason | None = None
    watchers: list[asyncio.Task[Any]] = field(default_factory=list)


async def run_capture(make_source: SourceFactory, config: CaptureConfig) -> CaptureRun:
    """Capture live data into a directory of segments, for as long as asked.

    *make_source* builds a fresh source for each segment: a source keeps
    connection and book state for one run and cannot be reused.  *config*
    gives the pair, the capture directory (``out_dir``), the total run time
    (``minutes``), and the roll limits (``roll_minutes``, ``roll_mb``).

    A new ``out_dir`` (or an empty one) starts a new capture.  One holding a
    ``manifest.json`` for the same source, pair and level is continued; any
    other contents are refused with :class:`~ob_analytics.exceptions.ConfigError`.
    So is a directory another capture process is writing to: the capture
    holds a lock on ``out_dir`` while it runs.

    SIGINT/SIGTERM stop the capture: the running segment is closed with its
    closing rows and the manifest is finished.
    """
    for name in ("roll_minutes", "roll_mb"):
        value = getattr(config, name)
        if value is not None and not value > 0:
            raise ConfigError(f"{name} must be greater than 0, not {value}")
    source = make_source()
    if isinstance(source, SupportsPreflight):
        source.preflight()
    root = Path(config.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    lock = _lock_capture_dir(root)
    try:
        manifest = _open_manifest(root, source, config)

        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        installed = install_stop_signals(loop, stop)
        supervisor = _Supervisor(make_source, config, root, manifest, stop)
        try:
            await supervisor.run(source)
        finally:
            for sig in installed:
                try:
                    loop.remove_signal_handler(sig)
                except (NotImplementedError, RuntimeError):
                    pass
    finally:
        lock.close()  # closing the file releases the lock
    return CaptureRun(out_dir=root, manifest=manifest, error=supervisor.error)


def _lock_capture_dir(root: Path) -> Any:
    """Take an exclusive lock on *root* for this process, or refuse.

    A second process continuing the same capture would take the first one's
    open segment for a crashed one and rewrite it under it.  The lock is an
    OS file lock, so it goes when the process does, however it ends.  Where
    ``fcntl`` is missing (Windows) no lock is taken.
    """
    fp = (root / LOCK_NAME).open("a")
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        return fp
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        fp.close()
        raise ConfigError(
            f"another capture process is writing to {root}; stop it, or "
            "capture into another directory."
        ) from exc
    return fp


class _Supervisor:
    def __init__(
        self,
        make_source: SourceFactory,
        config: CaptureConfig,
        root: Path,
        manifest: CaptureManifest,
        stop: asyncio.Event,
    ) -> None:
        self._make_source = make_source
        self._config = config
        self._root = root
        self._manifest = manifest
        self._stop = stop
        self._deadline = time.monotonic() + config.minutes * 60.0
        self._backoff = BACKOFF_START_SECONDS
        # A failing handover is retried no sooner than this (monotonic).
        self._retry_at = 0.0
        self._running: list[_Running] = []
        # Segments before this index belong to an earlier run (a restart).
        self._first_segment = len(manifest.segments)
        self.error: str | None = None

    # -- the loop -----------------------------------------------------------

    async def run(self, source: LiveSource | None) -> None:
        stop_task = asyncio.create_task(self._stop.wait())
        heartbeat = asyncio.create_task(self._heartbeat())
        current: _Running | None = None
        pending_roll: EndReason | None = None
        try:
            while not self._should_end():
                if source is not None:
                    new = self._start(source)
                    source = None
                    if current is None:
                        current = new
                        current.current_since = time.monotonic()
                        continue
                    current, adopted = await self._handover(
                        new, current, pending_roll, stop_task
                    )
                    if adopted:
                        pending_roll = None
                        continue
                    if new.task.done():
                        # The next segment failed before it streamed; the
                        # current one carries on and the roll is tried again
                        # after a wait.
                        await self._finish(new)
                        self._retry_at = time.monotonic() + self._next_backoff()
                    continue

                if current is None:
                    break
                what = await self._watch(current, stop_task)
                if what in (EndReason.ROLLED_TIME, EndReason.ROLLED_SIZE):
                    pending_roll = what
                    source = self._make_source()
                    continue
                if what is None:
                    break  # stopped or out of time
                # The segment ended by itself: a disconnect.
                segment = await self._finish(current)
                ended_run = current
                current = None
                if self._fatal(segment):
                    break
                if time.monotonic() - ended_run.opened >= BACKOFF_RESET_SECONDS and (
                    segment.stream_started is not None
                ):
                    self._backoff = BACKOFF_START_SECONDS
                if not await self._backoff_sleep(stop_task):
                    break
                source = self._make_source()
        finally:
            reason = EndReason.STOPPED if self._stop.is_set() else EndReason.FINISHED
            for running in list(self._running):
                await self._stop_segment(running, reason)
            heartbeat.cancel()
            stop_task.cancel()
            for task in (heartbeat, stop_task):
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001, S110 - draining helpers on shutdown
                    pass
            self._note_never_streamed()
            ended = pd.Timestamp.now(tz="UTC")
            self._manifest.ended = ended
            self._manifest.note_uncovered_end(ended)
            self._manifest.write(self._root)
            logger.info(
                "Capture '{}': {} segment(s), {} gap(s) ({:.1f} s), {} dropped",
                self._manifest.source,
                len(self._manifest.segments),
                len(self._manifest.gaps),
                self._manifest.gap_seconds,
                self._manifest.dropped,
            )

    def _should_end(self) -> bool:
        return self._stop.is_set() or self._remaining() <= 0

    def _remaining(self) -> float:
        return self._deadline - time.monotonic()

    def _next_backoff(self) -> float:
        wait = self._backoff
        self._backoff = min(self._backoff * 2, BACKOFF_MAX_SECONDS)
        return wait

    def _note_never_streamed(self) -> None:
        """Fail a run whose segments all failed without streaming anything.

        Such a run captured nothing, which is a failure even though each
        segment was retried; the last segment's error says why.
        """
        if self.error is not None:
            return
        this_run = self._manifest.segments[self._first_segment :]
        if any(s.stream_started is not None for s in this_run):
            return
        failed = [s for s in this_run if s.error is not None]
        if failed:
            self.error = failed[-1].error

    def _fatal(self, segment: Segment) -> bool:
        """Whether a failed segment should stop the run instead of retrying.

        Only a capture that has never streamed stops: a first segment that
        fails before its first event -- in its snapshot, or in a stream that
        rejects the pair -- means the settings are probably wrong.  Once a
        capture has worked, a failure (even the first one after a restart,
        with the network perhaps not up yet) is waited out.
        """
        if any(s.stream_started is not None for s in self._manifest.segments):
            return False
        if segment.end_reason is not EndReason.FAILED:
            return False
        self.error = segment.error
        logger.error(
            "Capture '{}' could not start: {}", self._manifest.source, segment.error
        )
        return True

    # -- one segment --------------------------------------------------------

    def _start(self, source: LiveSource) -> _Running:
        name = self._manifest.next_segment_name()
        out_dir = self._root / name
        level = getattr(source, "level", Level.L3)
        sink = FileCaptureSink(out_dir, keep_raw=self._config.keep_raw, level=level)
        # The source's own clock runs past the capture's end: the supervisor
        # stops each segment, so a source never ends a segment early by itself.
        seg_config = CaptureConfig(
            pair=self._config.pair,
            out_dir=out_dir,
            minutes=max(self._remaining(), 0.0) / 60.0 + 1.0,
            keep_raw=self._config.keep_raw,
        )
        stop = asyncio.Event()
        streaming = asyncio.Event()
        task = asyncio.create_task(
            run_capturer(source, seg_config, sink, stop=stop, streaming=streaming)
        )
        segment = Segment(name=name, started=pd.Timestamp.now(tz="UTC"))
        self._manifest.segments.append(segment)
        running = _Running(
            segment=segment,
            source=source,
            sink=sink,
            stop=stop,
            streaming=streaming,
            task=task,
            opened=time.monotonic(),
        )
        running.watchers.append(asyncio.create_task(self._on_streaming(running)))
        self._running.append(running)
        self._manifest.write(self._root)
        logger.info("Capture '{}': started {}", self._manifest.source, name)
        return running

    async def _on_streaming(self, running: _Running) -> None:
        """Record the segment's first live event, and any gap before it."""
        await running.streaming.wait()
        running.bytes_at_stream = running.sink.bytes_written()
        segment = running.segment
        segment.stream_started = pd.Timestamp.now(tz="UTC")
        self._manifest.note_streaming(segment)
        self._manifest.write(self._root)
        logger.info("Capture '{}': {} streaming", self._manifest.source, segment.name)

    async def _stop_segment(self, running: _Running, reason: EndReason) -> Segment:
        running.stop_reason = reason
        running.stop.set()
        return await self._finish(running)

    async def _finish(self, running: _Running) -> Segment:
        """Wait for the segment's task and record how it ended."""
        segment = running.segment
        result: CaptureResult | None = None
        try:
            result = await running.task
        except Exception as exc:  # noqa: BLE001 - recorded in the manifest
            # run_capturer keeps errors from its phases; one that escapes it
            # happened before any of them (the preflight, say).
            segment.error = repr(exc)
        for watcher in running.watchers:
            watcher.cancel()
        if running in self._running:
            self._running.remove(running)

        if result is not None:
            segment.ended = result.ended
            if segment.stream_started is None and result.stream_started is not None:
                segment.stream_started = result.stream_started
                self._manifest.note_streaming(segment)
            segment.stream_ended = (
                result.stream_ended if segment.stream_started is not None else None
            )
            segment.error = result.capture_error
            segment.n_book_events = result.n_order_events + result.n_depth_events
            segment.n_trade_events = result.n_trade_events
            segment.dropped = int(result.extras.get("dropped") or 0)
            segment.sequence_missing = int(result.extras.get("sequence_missing") or 0)
        else:
            segment.ended = pd.Timestamp.now(tz="UTC")

        if segment.error is not None:
            segment.end_reason = EndReason.FAILED
        elif running.stop_reason is not None:
            segment.end_reason = running.stop_reason
        else:
            segment.end_reason = EndReason.ENDED_EARLY
        self._manifest.write(self._root)
        log = logger.info if segment.error is None else logger.warning
        log(
            "Capture '{}': {} ended ({}){}",
            self._manifest.source,
            segment.name,
            segment.end_reason.value,
            f": {segment.error}" if segment.error else "",
        )
        return segment

    # -- waiting ------------------------------------------------------------

    async def _watch(
        self, current: _Running, stop_task: asyncio.Task[Any]
    ) -> EndReason | Literal["ended"] | None:
        """Wait until *current* ends, a roll is due, or the capture should end.

        Returns a roll reason, ``"ended"`` when the segment stopped by itself,
        or ``None`` when the capture is stopped or out of time.
        """
        roll_seconds = (
            None
            if self._config.roll_minutes is None
            else self._config.roll_minutes * 60
        )
        roll_bytes = (
            None if self._config.roll_mb is None else self._config.roll_mb * 1_000_000
        )
        while True:
            if current.task.done():
                return "ended"
            if self._should_end():
                return None
            now = time.monotonic()
            if now >= self._retry_at:
                since = current.current_since or current.opened
                if roll_seconds is not None and now - since >= roll_seconds:
                    return EndReason.ROLLED_TIME
                if (
                    roll_bytes is not None
                    and current.bytes_at_stream is not None
                    and current.sink.bytes_written() - current.bytes_at_stream
                    >= roll_bytes
                ):
                    return EndReason.ROLLED_SIZE
            timeout = max(0.0, min(POLL_SECONDS, self._remaining()))
            await asyncio.wait(
                {current.task, stop_task},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )

    async def _handover(
        self,
        new: _Running,
        current: _Running,
        reason: EndReason | None,
        stop_task: asyncio.Task[Any],
    ) -> tuple[_Running, bool]:
        """Let *new* take over from *current* once it streams.

        Returns ``(current, adopted)``: when *adopted*, *new* is now the
        current segment and the old one has been stopped.  When not, *new*
        failed before it streamed (or the capture is ending) and the old one,
        if it is still running, stays current.
        """
        waited_from = time.monotonic()
        streaming = asyncio.create_task(new.streaming.wait())
        try:
            while True:
                if new.streaming.is_set() or (
                    not new.task.done()
                    and time.monotonic() - waited_from >= HANDOVER_TIMEOUT_SECONDS
                ):
                    await self._stop_segment(current, reason or EndReason.ROLLED_TIME)
                    self._retry_at = 0.0
                    new.current_since = time.monotonic()
                    return new, True
                if new.task.done() or self._should_end():
                    return current, False
                if current.task.done():
                    # The old segment died while the new one was starting: the
                    # new one simply takes over, and any gap between them is
                    # recorded when it streams.
                    await self._finish(current)
                    new.current_since = time.monotonic()
                    return new, True
                await asyncio.wait(
                    {new.task, current.task, stop_task, streaming},
                    timeout=max(0.0, min(POLL_SECONDS, self._remaining())),
                    return_when=asyncio.FIRST_COMPLETED,
                )
        finally:
            streaming.cancel()

    async def _backoff_sleep(self, stop_task: asyncio.Task[Any]) -> bool:
        """Wait before the next segment; ``False`` if the capture should end."""
        wait = min(self._next_backoff(), max(self._remaining(), 0.0))
        logger.info(
            "Capture '{}': starting a new segment in {:.1f} s",
            self._manifest.source,
            wait,
        )
        await asyncio.wait({stop_task}, timeout=wait)
        return not self._should_end()

    async def _heartbeat(self) -> None:
        """Keep the manifest and running segments' ``meta.json`` current."""
        while True:
            await asyncio.sleep(HEARTBEAT_SECONDS)
            now = pd.Timestamp.now(tz="UTC")
            for running in self._running:
                if running.task.done():
                    continue
                running.segment.heartbeat = now
                running.sink.flush()
                _write_provisional_meta(running)
            self._manifest.write(self._root)


# ---------------------------------------------------------------------------
# Starting and continuing a capture directory
# ---------------------------------------------------------------------------


def _declarations(source: LiveSource) -> dict[str, Any]:
    declared = _source_declarations(source)
    declared.pop("source", None)
    return declared


def _open_manifest(
    root: Path, source: LiveSource, config: CaptureConfig
) -> CaptureManifest:
    """Start a new capture in *root*, or continue the one already there."""
    level = str(getattr(source, "level", Level.L3).value)
    try:
        existing = read_manifest(root)
    except ValueError as exc:
        raise ConfigError(f"{root}: {exc}") from exc
    if existing is None:
        if any(p.name != LOCK_NAME for p in root.iterdir()):
            raise ConfigError(
                f"{root} already holds files but no manifest.json, so it is not "
                "a capture this command can continue. Capture into a new or "
                "empty directory."
            )
        root.mkdir(parents=True, exist_ok=True)
        manifest = CaptureManifest(
            source=source.name,
            pair=config.pair,
            level=level,
            started=pd.Timestamp.now(tz="UTC"),
            declarations=_declarations(source),
            roll_minutes=config.roll_minutes,
            roll_mb=config.roll_mb,
        )
        manifest.write(root)
        return manifest

    found = (existing.source, existing.pair, existing.level)
    wanted = (source.name, config.pair, level)
    if found != wanted:
        raise ConfigError(
            f"{root} holds a capture of {found[0]} {found[1]} at {found[2]}, not "
            f"{wanted[0]} {wanted[1]} at {wanted[2]}. Capture into another directory."
        )
    existing.restarts += 1
    existing.ended = None
    existing.roll_minutes = config.roll_minutes
    existing.roll_mb = config.roll_mb
    for segment in existing.segments:
        if segment.end_reason is None:
            _close_unfinished(root / segment.name, segment, existing)
    existing.write(root)
    logger.info(
        "Continuing the capture in {} ({} segment(s) so far, restart #{})",
        root,
        len(existing.segments),
        existing.restarts,
    )
    return existing


def _close_unfinished(
    seg_dir: Path, segment: Segment, manifest: CaptureManifest
) -> None:
    """Close a segment a dead capture process left open, so it replays.

    The process never wrote the segment's closing rows or its final
    ``meta.json``, and its last line in each file may be cut short -- or, if
    the process died before its first flush, a file may be empty.  This cuts
    each file back to its last whole line, puts back a missing header, closes
    every L3 order still open at the last recorded time (the same ``deleted``
    rows a normal shutdown writes), and completes ``meta.json``.  The segment is marked
    :attr:`~ob_analytics.live._manifest.EndReason.UNFINISHED`; it is taken to
    cover up to its last row or last heartbeat, whichever is later.
    """
    for name in ("orders.csv", "depth.csv", "trades.csv", "raw.jsonl"):
        _trim_to_last_line(seg_dir / name)
    for name, columns in (
        ("orders.csv", _ORDER_COLS),
        ("depth.csv", _DEPTH_COLS),
        ("trades.csv", _TRADE_COLS),
    ):
        _restore_header(seg_dir / name, columns)

    stamps = [
        ms
        for name in ("orders.csv", "depth.csv", "trades.csv")
        if (ms := _last_timestamp_ms(seg_dir / name)) is not None
    ]
    last_ms = max(stamps, default=None)
    closed = 0
    orders = seg_dir / "orders.csv"
    if orders.is_file() and last_ms is not None:
        closed = _close_open_orders(orders, last_ms)

    last_row = None if last_ms is None else pd.Timestamp(last_ms, unit="ms", tz="UTC")
    covered_to = max(
        (t for t in (last_row, segment.heartbeat) if t is not None), default=None
    )
    n_book = _count_rows(orders) + _count_rows(seg_dir / "depth.csv")
    n_trade = _count_rows(seg_dir / "trades.csv")

    meta_path = seg_dir / "meta.json"
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            meta = {}
    meta.pop("provisional", None)
    meta.update(
        {
            "source": manifest.source,
            **manifest.declarations,
            "out_dir": str(seg_dir),
            "started": str(segment.started),
            "ended": str(covered_to),
            "n_order_events": _count_rows(orders),
            "n_depth_events": _count_rows(seg_dir / "depth.csv"),
            "n_trade_events": n_trade,
            "stream_started": None
            if segment.stream_started is None
            else str(segment.stream_started),
            "stream_ended": None if covered_to is None else str(covered_to),
            "synthetic_deleted": closed,
            "unfinished": True,
            "capture_error": _UNFINISHED_ERROR,
            "capture_error_phase": "stream",
            "errors": int(meta.get("errors") or 0) + 1,
        }
    )
    _write_json_atomic(meta_path, meta)

    segment.end_reason = EndReason.UNFINISHED
    segment.error = _UNFINISHED_ERROR
    segment.ended = covered_to or segment.started
    if segment.stream_started is not None:
        segment.stream_ended = covered_to
    segment.n_book_events = n_book
    segment.n_trade_events = n_trade
    segment.dropped = int(meta.get("dropped") or 0)
    segment.sequence_missing = int(meta.get("sequence_missing") or 0)
    logger.warning(
        "Closed {} left open by a capture that stopped ({} order(s) closed)",
        segment.name,
        closed,
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


def _restore_header(path: Path, columns: list[str]) -> None:
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
                    "sequence": "",
                    "origin": ORIGIN_SHUTDOWN,
                }
            )
    return len(open_orders)


def _write_provisional_meta(running: _Running) -> None:
    """Write what is known of a running segment to its ``meta.json``.

    The runner writes the real ``meta.json`` when the segment ends.  Until
    then this one keeps the source's declarations and counters (a ccxt tick
    size, say) on disk, so a segment a crash leaves open can still be read.
    """
    meta: dict[str, Any] = {
        **_source_declarations(running.source),
        "out_dir": str(running.sink.out_dir),
        "started": str(running.segment.started),
        "stream_started": None
        if running.segment.stream_started is None
        else str(running.segment.stream_started),
        "provisional": True,
    }
    if isinstance(running.source, SupportsDiagnostics):
        try:
            meta.update(running.source.diagnostics())
        except Exception as exc:  # noqa: BLE001 - a heartbeat must not stop the capture
            logger.debug("diagnostics() raised during heartbeat: {!r}", exc)
    _write_json_atomic(running.sink.out_dir / "meta.json", meta)


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)
