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

A segment asked to stop has :data:`STOP_TIMEOUT_SECONDS` to do so.  One that
takes longer is cancelled and closed from its files the same way, so a source
that hangs while it closes cannot hold up the capture or a signal to end it.

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
    _FirstEvent,
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
#: How long a segment asked to stop (at a roll, at the end, or on a signal) has
#: to close its connection and write its closing rows.  One that takes longer
#: is cancelled and closed from its files.
STOP_TIMEOUT_SECONDS = 20.0
#: How long a cancelled segment has to end before the capture goes on without it.
STOP_GRACE_SECONDS = 5.0

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
    streaming: _FirstEvent
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
    # When the supervisor asked it to stop, and whether it then did not stop
    # in time and was cancelled.
    stop_requested: pd.Timestamp | None = None
    stuck: bool = False
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
        # raw.jsonl warnings already logged, shared by every segment's sink so
        # each is logged once per capture.
        self._raw_warned: set[str] = set()
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
            # Together, so the capture ends within one stop's time limit.  One
            # that fails must not keep the rest of this block from running.
            stopping = list(self._running)
            outcomes = await asyncio.gather(
                *(self._stop_segment(r, reason) for r in stopping),
                return_exceptions=True,
            )
            for running, outcome in zip(stopping, outcomes, strict=True):
                if isinstance(outcome, BaseException):
                    logger.error(
                        "Capture '{}': stopping {} failed: {!r}",
                        self._manifest.source,
                        running.segment.name,
                        outcome,
                    )
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
            if self._manifest.raw_frames_skipped:
                logger.warning(
                    "Capture '{}': raw.jsonl skipped {} frame(s) JSON cannot hold "
                    "(see each segment's meta.json)",
                    self._manifest.source,
                    self._manifest.raw_frames_skipped,
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
        sink = FileCaptureSink(
            out_dir,
            keep_raw=self._config.keep_raw,
            level=level,
            raw_warned=self._raw_warned,
        )
        # The source's own clock runs past the capture's end: the supervisor
        # stops each segment, so a source never ends a segment early by itself.
        seg_config = CaptureConfig(
            pair=self._config.pair,
            out_dir=out_dir,
            minutes=max(self._remaining(), 0.0) / 60.0 + 1.0,
            keep_raw=self._config.keep_raw,
        )
        stop = asyncio.Event()
        streaming = _FirstEvent()
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
        first_event = running.streaming.at or pd.Timestamp.now(tz="UTC")
        if not _streamed_before_stop(first_event, running.stop_requested):
            return
        segment = running.segment
        segment.stream_started = first_event
        self._manifest.note_streaming(segment)
        self._manifest.write(self._root)
        logger.info("Capture '{}': {} streaming", self._manifest.source, segment.name)

    async def _stop_segment(self, running: _Running, reason: EndReason) -> Segment:
        """Stop the segment, cancelling it if it does not stop in time.

        A segment that hangs while it closes must not hold the capture: until
        it ends, there are no more rolls, a disconnect of the next segment
        goes unhandled, and a signal cannot end the capture.
        """
        running.stop_reason = reason
        running.stop_requested = pd.Timestamp.now(tz="UTC")
        running.stop.set()
        if not await _ends_within(running.task, STOP_TIMEOUT_SECONDS):
            logger.error(
                "Capture '{}': {} did not stop within {:.0f} s; cancelling it",
                self._manifest.source,
                running.segment.name,
                STOP_TIMEOUT_SECONDS,
            )
            running.stuck = True
            running.task.cancel()
            await _ends_within(running.task, STOP_GRACE_SECONDS)
        return await self._finish(running)

    async def _finish(self, running: _Running) -> Segment:
        """Wait for the segment's task and record how it ended."""
        segment = running.segment
        result: CaptureResult | None = None
        if not running.stuck:
            try:
                result = await running.task
            except Exception as exc:  # noqa: BLE001 - recorded in the manifest
                # run_capturer keeps errors from its phases; one that escapes
                # it happened before any of them (the preflight, say).
                segment.error = repr(exc)
        elif _finished_normally(running.task):
            # Late, but it ended by itself before the cancel took: its own
            # closing rows and meta.json are complete.
            result = running.task.result()
        for watcher in running.watchers:
            watcher.cancel()
        if running in self._running:
            self._running.remove(running)

        if result is None and running.stuck:
            await self._close_stuck(running)
        elif result is not None:
            segment.ended = result.ended
            if segment.stream_started is None and _streamed_before_stop(
                result.stream_started, running.stop_requested
            ):
                segment.stream_started = result.stream_started
                self._manifest.note_streaming(segment)
            segment.stream_ended = _covered_until(
                segment.stream_started, result.stream_ended, running.stop_requested
            )
            if (segment.stream_started, segment.stream_ended) != (
                result.stream_started,
                result.stream_ended,
            ):
                _record_coverage(self._root / segment.name, segment)
            segment.error = result.capture_error
            segment.n_book_events = result.n_order_events + result.n_depth_events
            segment.n_trade_events = result.n_trade_events
            segment.dropped = int(result.extras.get("dropped") or 0)
            segment.book_resyncs = int(result.extras.get("book_resyncs") or 0)
            segment.raw_frames_skipped = running.sink.raw_frames_skipped
            if running.stuck:
                late = (
                    f"the segment took more than {STOP_TIMEOUT_SECONDS:.0f} s to stop"
                )
                segment.error = "; ".join(e for e in (segment.error, late) if e)
        else:
            segment.ended = pd.Timestamp.now(tz="UTC")

        if running.stuck and running.stop_reason is not None:
            # It was asked to stop and did, only late: why it was stopped
            # stands, and the lateness is its error.  Recording it as failed
            # would count the end of its stream as a disconnect.
            segment.end_reason = running.stop_reason
        elif segment.error is not None:
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

    async def _close_stuck(self, running: _Running) -> None:
        """Record a segment that did not stop in time and was cancelled.

        Once it has ended, its files are closed the way a restart closes a
        segment a crash left open: every order still open gets its
        ``deleted`` row at the last recorded time.  One that ignored the
        cancel too is left as it is, since it may still be writing.  Either
        way the capture goes on, even if closing the files fails.
        """
        segment = running.segment
        task = running.task
        limit = f"the segment did not stop within {STOP_TIMEOUT_SECONDS:.0f} s"
        # What raw.jsonl skipped so far, in case its files cannot be closed.
        segment.raw_frames_skipped = running.sink.raw_frames_skipped
        if task.done():
            error = f"{limit}, so it was cancelled and closed from its files"
            failure = None if task.cancelled() else task.exception()
            if failure is not None:
                error += f" (it ended with {failure!r})"
            try:
                # In a thread: it reads the segment's files in full, and the
                # next segment is streaming on this loop.  It changes nothing
                # the loop uses; the results are applied here.
                closed = await asyncio.to_thread(
                    _close_segment_files,
                    self._root / segment.name,
                    segment,
                    self._manifest,
                    error,
                    # The task has ended, so the sink's counts are final.
                    running.sink.raw_diagnostics(),
                )
            except Exception as exc:  # noqa: BLE001 - the capture must go on
                logger.error(
                    "Capture '{}': closing the files of {} failed: {!r}",
                    self._manifest.source,
                    segment.name,
                    exc,
                )
                error = f"{limit}; closing its files failed: {exc!r}"
            else:
                _apply_closed_files(segment, closed, error)
                segment.ended = pd.Timestamp.now(tz="UTC")
                segment.stream_ended = _covered_until(
                    segment.stream_started, segment.stream_ended, running.stop_requested
                )
                if segment.stream_ended != closed.covered_to:
                    _record_coverage(self._root / segment.name, segment)
                return
        else:
            error = f"{limit}, even when cancelled; its files may be incomplete"
        # Nothing on disk says how far it got: count it as covering the market
        # only until it was asked to stop, so a gap after it is not hidden.
        segment.error = error
        segment.ended = pd.Timestamp.now(tz="UTC")
        segment.stream_ended = _covered_until(
            segment.stream_started, segment.ended, running.stop_requested
        )

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
                # A segment asked to stop no longer covers the market: its
                # heartbeat would count the time it takes to close.  Its rows
                # are still flushed, so a crash while it closes loses none.
                if not running.stop.is_set():
                    running.segment.heartbeat = now
                running.sink.flush()
                _write_provisional_meta(running)
            self._manifest.write(self._root)


# ---------------------------------------------------------------------------
# Starting and continuing a capture directory
# ---------------------------------------------------------------------------


def _streamed_before_stop(
    first_event: pd.Timestamp | None, stop_requested: pd.Timestamp | None
) -> bool:
    """Whether a segment's first live event came before it was asked to stop.

    One asked to stop first (a source can replay what it buffered during its
    snapshot) covers nothing, so it neither ends a gap nor starts one.
    """
    if first_event is None:
        return False
    return stop_requested is None or first_event < stop_requested


def _covered_until(
    stream_started: pd.Timestamp | None,
    stream_ended: pd.Timestamp | None,
    stop_requested: pd.Timestamp | None,
) -> pd.Timestamp | None:
    """How long a segment covered the market, given when its stream ended.

    Once asked to stop it covers the market no longer, however long closing
    its connection then takes.  A segment asked to stop before its first live
    event (still taking its snapshot, say) covered nothing: its coverage ends
    where it started, never before.  ``None`` for one that never streamed.
    """
    if stream_started is None or stream_ended is None:
        return None
    if stop_requested is not None:
        stream_ended = min(stream_ended, stop_requested)
    return max(stream_ended, stream_started)


def _record_coverage(seg_dir: Path, segment: Segment) -> None:
    """Make the segment's ``meta.json`` agree with the manifest's coverage."""
    meta_path = seg_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not update {}: {!r}", meta_path, exc)
        return
    for key in ("stream_started", "stream_ended"):
        value = getattr(segment, key)
        meta[key] = None if value is None else str(value)
    _write_json_atomic(meta_path, meta)


def _finished_normally(task: asyncio.Task[Any]) -> bool:
    """Whether *task* has ended with a result (not cancelled, not raised)."""
    return task.done() and not task.cancelled() and task.exception() is None


async def _ends_within(task: asyncio.Task[Any], seconds: float) -> bool:
    """Wait up to *seconds* for *task* to end; whether it did."""
    await asyncio.wait({task}, timeout=seconds)
    return task.done()


def _declarations(source: LiveSource) -> dict[str, Any]:
    declared = _source_declarations(source)
    declared.pop("source", None)
    # A live source learns its clocks from the venue's books, so the value is
    # not known when the manifest is opened.  Each segment's meta.json has it.
    declared.pop("clocks", None)
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
            closed = _close_unfinished(root / segment.name, segment, existing)
            logger.warning(
                "Closed {} left open by a capture that stopped ({} order(s) closed)",
                segment.name,
                closed,
            )
    existing.write(root)
    logger.info(
        "Continuing the capture in {} ({} segment(s) so far, restart #{})",
        root,
        len(existing.segments),
        existing.restarts,
    )
    return existing


def _close_unfinished(
    seg_dir: Path,
    segment: Segment,
    manifest: CaptureManifest,
    error: str = _UNFINISHED_ERROR,
) -> int:
    """Close a segment a dead capture process left open, so it replays.

    The process never wrote the segment's closing rows or its final
    ``meta.json``.  :func:`_close_segment_files` completes the files, and the
    segment is marked
    :attr:`~ob_analytics.live._manifest.EndReason.UNFINISHED`; it is taken to
    cover up to its last row or last heartbeat, whichever is later.

    *error* says why the segment was left open.  Returns how many orders it
    closed.
    """
    closed = _close_segment_files(seg_dir, segment, manifest, error)
    segment.end_reason = EndReason.UNFINISHED
    _apply_closed_files(segment, closed, error)
    return closed.orders_closed


@dataclass(frozen=True)
class _ClosedFiles:
    """What :func:`_close_segment_files` found and wrote."""

    covered_to: pd.Timestamp | None
    orders_closed: int
    n_book: int
    n_trade: int
    dropped: int
    book_resyncs: int
    raw_frames_skipped: int


def _close_segment_files(
    seg_dir: Path,
    segment: Segment,
    manifest: CaptureManifest,
    error: str,
    raw: dict[str, Any] | None = None,
) -> _ClosedFiles:
    """Complete the files of a segment whose capture never closed them.

    Its last line in each file may be cut short -- or, if the capture died
    before its first flush, a file may be empty.  This cuts each file back to
    its last whole line, puts back a missing header, closes every L3 order
    still open at the last recorded time (the same ``deleted`` rows a normal
    shutdown writes), and completes ``meta.json`` with *error*.

    *raw* is what the segment's sink counted for ``raw.jsonl`` (see
    :meth:`FileCaptureSink.raw_diagnostics`), when the sink is still at hand.
    It replaces the counts in the provisional ``meta.json``, which are as old
    as its last rewrite.  After a crash there is no sink, and those counts
    stand.

    It reads *segment* and *manifest* but changes neither, so it can run in a
    thread while the capture goes on.
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
            "capture_error": error,
            "capture_error_phase": "stream",
            "errors": int(meta.get("errors") or 0) + 1,
            **(raw or {}),
        }
    )
    _write_json_atomic(meta_path, meta)
    return _ClosedFiles(
        covered_to=covered_to,
        orders_closed=closed,
        n_book=n_book,
        n_trade=n_trade,
        dropped=int(meta.get("dropped") or 0),
        book_resyncs=int(meta.get("book_resyncs") or 0),
        raw_frames_skipped=int(meta.get("n_raw_frames_skipped") or 0),
    )


def _apply_closed_files(segment: Segment, closed: _ClosedFiles, error: str) -> None:
    """Record on *segment* what closing its files found."""
    segment.error = error
    segment.ended = closed.covered_to or segment.started
    if segment.stream_started is not None:
        segment.stream_ended = closed.covered_to
    segment.n_book_events = closed.n_book
    segment.n_trade_events = closed.n_trade
    segment.dropped = closed.dropped
    segment.book_resyncs = closed.book_resyncs
    segment.raw_frames_skipped = closed.raw_frames_skipped


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
    meta.update(running.sink.raw_diagnostics())
    _write_json_atomic(running.sink.out_dir / "meta.json", meta)


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)
