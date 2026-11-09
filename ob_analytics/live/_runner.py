"""Generic asyncio driver that turns any :class:`LiveSource` into files."""

from __future__ import annotations

import asyncio
import signal
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import pandas as pd
from loguru import logger

from ob_analytics._secrets import redact
from ob_analytics.capture_files import (
    ORIGIN_SHUTDOWN,
    ORIGIN_SNAPSHOT,
    ORIGIN_STREAM,
    FileCaptureSink,
)
from ob_analytics.capture_record import source_declarations
from ob_analytics.config import SourceSettings
from ob_analytics.live._base import (
    CaptureConfig,
    CaptureResult,
    CaptureSink,
    EventDict,
    LiveSource,
    SupportsDiagnostics,
    SupportsPreflight,
)
from ob_analytics.protocols import Level
from ob_analytics.schemas import ORIGIN_COLUMN

# How often a stopped stream that is still yielding items is cancelled again.
CANCEL_RETRY_SECONDS = 0.5


def check_credentials(source: Any) -> None:
    """Raise :class:`~ob_analytics.exceptions.ConfigError` if *source* lacks a key.

    Reads the credentials the source's settings declare (see
    :class:`~ob_analytics.config.Credential`); a source with none passes.  Also
    registers the keys that are set for redaction, so it must run before the
    capture's files are opened.
    """
    settings = getattr(source, "settings", None)
    if isinstance(settings, SourceSettings):
        settings.check_credentials(source.name)


async def run_capturer(
    capturer: LiveSource,
    config: CaptureConfig,
    sink: CaptureSink | None = None,
    *,
    stop: asyncio.Event | None = None,
    streaming: asyncio.Event | None = None,
) -> CaptureResult:
    """Drive a live source: snapshot, stream, shutdown -- writing through *sink*.

    This is one capture into one directory, and it ends at the first
    disconnect.  :func:`~ob_analytics.live.run_capture` runs it once per
    segment to capture for days.

    Handles SIGINT/SIGTERM by cancelling the streaming task; the shutdown
    synthetic events still run so every order id keeps a full lifecycle.  A
    second signal stops the run at once, in whatever phase it is: the files
    are closed with the rows written so far, and ``meta.json`` records the
    cut as :attr:`CaptureResult.capture_error`.  A caller that passes *stop*
    owns the signals instead: setting it ends the stream the same way, and no
    handler is installed.  *streaming*, when given, is set as the first live
    event arrives.

    A source that implements :class:`SupportsPreflight` is checked before any
    output is created, so a missing optional extra raises :class:`ImportError`
    here and leaves no output directory behind. Once output exists, an
    exception from the snapshot, the stream or the shutdown events does not
    propagate: the rows already written are kept, and the first error is
    recorded in :attr:`CaptureResult.capture_error` (with the phase in
    :attr:`CaptureResult.capture_error_phase`) and in ``meta.json``. A failed
    snapshot skips the stream; the shutdown events still run.

    A source whose settings declare a :class:`~ob_analytics.config.Credential`
    that is not set raises :class:`~ob_analytics.exceptions.ConfigError` here,
    also before any output is created.
    """
    if isinstance(capturer, SupportsPreflight):
        capturer.preflight()
    check_credentials(capturer)
    # The source declares its granularity; the runner routes book events to
    # the matching writer. Fall back to L3 for sources predating the attr.
    level = getattr(capturer, "level", Level.L3)
    if sink is None:
        sink = FileCaptureSink(config.out_dir, keep_raw=config.keep_raw, level=level)
    started = pd.Timestamp.now(tz="UTC")
    stream_times: dict[str, pd.Timestamp] = {}
    # Rows written per kind, by every phase.  Updated in place, so the counts
    # survive a cancellation (a second signal, or the supervisor giving up on
    # a segment) and meta.json reports every row on disk.
    counts = {"order": 0, "trade": 0, "depth": 0, "raw": 0}
    capture_error: str | None = None
    capture_error_phase: str | None = None
    phase = "snapshot"

    def _record_error(phase: str, exc: BaseException) -> None:
        nonlocal capture_error, capture_error_phase
        error = redact(repr(exc))
        logger.error("Capturer '{}' {} raised: {}", capturer.name, phase, error)
        # Keep the first error: a later one is usually a consequence of it.
        if capture_error is None:
            capture_error = error
            capture_error_phase = phase

    def _write_book(ev: EventDict) -> None:
        if level is Level.L2:
            sink.write_depth(ev)
            counts["depth"] += 1
        else:
            sink.write_order(ev)
            counts["order"] += 1

    loop = asyncio.get_event_loop()
    installed_signals: list[int] = []
    stopped_at_once = False
    if stop is None:
        stop = asyncio.Event()
        this_task = asyncio.current_task()

        def _stop_at_once() -> None:
            nonlocal stopped_at_once
            stopped_at_once = True
            if this_task is not None:
                this_task.cancel()

        installed_signals = install_stop_signals(loop, stop, _stop_at_once)

    # L3 only: ids from the opening book that nothing in the stream has yet
    # mentioned. _stream removes an id when an order event or a trade names
    # it, so what is left at the end is the opening book the stream never
    # confirmed. L2 levels carry no id, so an L2 run leaves this as None.
    unconfirmed: set[Any] | None = None if level is Level.L2 else set()

    try:
        try:
            logger.info("Capturer '{}': snapshot starting", capturer.name)
            try:
                async for ev in capturer.snapshot(config):
                    ev[ORIGIN_COLUMN] = ORIGIN_SNAPSHOT
                    _write_book(ev)
                    if unconfirmed is not None:
                        unconfirmed.add(ev["id"])
            except Exception as exc:  # noqa: BLE001 - recorded in meta.json
                _record_error("snapshot", exc)
            logger.info(
                "Capturer '{}': snapshot wrote {} book events",
                capturer.name,
                counts["depth"] if level is Level.L2 else counts["order"],
            )

            if capture_error is None:
                phase = "stream"
                await _run_stream(
                    capturer,
                    config,
                    sink,
                    _StreamState(counts, unconfirmed, stream_times, streaming),
                    stop,
                    _record_error,
                )
                stream_times["ended"] = pd.Timestamp.now(tz="UTC")

            phase = "shutdown"
            logger.info(
                "Capturer '{}': emitting shutdown synthetic events", capturer.name
            )
            try:
                async for ev in capturer.shutdown_synthetic_events():
                    ev.setdefault(ORIGIN_COLUMN, ORIGIN_SHUTDOWN)
                    _write_book(ev)
            except Exception as exc:  # noqa: BLE001 - recorded in meta.json
                _record_error("shutdown", exc)
        except asyncio.CancelledError:
            # Only the cancel a second signal asked for ends here; any other
            # (the caller's own) goes on up once the files are closed.
            task = asyncio.current_task()
            if not stopped_at_once or task is None or task.uncancel() > 0:
                raise
            logger.warning(
                "Capturer '{}': a second signal stopped the run at once",
                capturer.name,
            )
            if capture_error is None:
                capture_error = f"a second signal stopped the run at once ({phase})"
                capture_error_phase = phase
            if phase == "stream":
                stream_times["ended"] = pd.Timestamp.now(tz="UTC")
    finally:
        # Remove signal handlers we installed.
        for sig in installed_signals:
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError):
                pass

        ended = pd.Timestamp.now(tz="UTC")
        # What the source declares about its feed, so a later `audit` can hold
        # the capture to its own source's expectations.  The capture is read
        # back with a file format's source (a cryptofeed L3 capture replays as
        # bitstamp), which would otherwise supply the wrong ones.
        extras: dict[str, Any] = {
            **source_declarations(capturer),
            **read_diagnostics(capturer),
        }
        result = CaptureResult(
            out_dir=config.out_dir,
            n_order_events=counts["order"],
            n_trade_events=counts["trade"],
            n_raw_frames=counts["raw"],
            started=started,
            ended=ended,
            extras=extras,
            n_depth_events=counts["depth"],
            n_snapshot_unconfirmed=None if unconfirmed is None else len(unconfirmed),
            capture_error=capture_error,
            capture_error_phase=capture_error_phase,
            stream_started=stream_times.get("started"),
            stream_ended=stream_times.get("ended"),
        )
        sink.finalize(result)
        close_error = getattr(sink, "close_error", None)
        if close_error is not None and capture_error is None:
            result = replace(
                result, capture_error=close_error, capture_error_phase="close"
            )
        logger.info(
            "Capturer '{}': finished. orders={}, depth={}, trades={}, raw={}, "
            "dur={:.1f}s",
            capturer.name,
            counts["order"],
            counts["depth"],
            counts["trade"],
            counts["raw"],
            (ended - started).total_seconds(),
        )
    return result


def read_diagnostics(source: Any) -> dict[str, Any]:
    """Return the counters *source* reports for ``meta.json``.

    Empty for a source without :class:`SupportsDiagnostics`.  When its
    ``diagnostics()`` raises, the counters are lost: this warns, and returns
    ``diagnostics_error`` (the error's ``repr``) instead, so ``meta.json``
    says so and ``ob-analytics audit`` does not take the missing ``dropped``
    count for zero.
    """
    if not isinstance(source, SupportsDiagnostics):
        return {}
    try:
        return dict(source.diagnostics())
    except Exception as exc:  # noqa: BLE001 - recorded in meta.json
        error = redact(repr(exc))
        warnings.warn(
            f"Capturer '{source.name}' diagnostics() raised {error}; meta.json "
            "has none of its counters",
            UserWarning,
            stacklevel=2,
        )
        return {DIAGNOSTICS_ERROR: error}


#: The ``meta.json`` field that says the source's counters were lost.
DIAGNOSTICS_ERROR = "diagnostics_error"


def install_stop_signals(
    loop: asyncio.AbstractEventLoop,
    stop: asyncio.Event,
    at_once: Callable[[], None] | None = None,
) -> list[int]:
    """Make SIGINT/SIGTERM set *stop*; return the signals handled.

    The first signal sets *stop*, so the capture closes cleanly.  A second
    one calls *at_once*, which stops it without waiting for that.

    Where signals cannot be handled (Windows, a thread other than the main
    one, some test environments) nothing is installed and default handling
    stays.
    """
    received = 0

    def _on_signal() -> None:
        nonlocal received
        received += 1
        if received == 1:
            logger.warning(
                "Stopping the capture: closing its files. Send the signal again "
                "to stop at once."
            )
            stop.set()
        elif received == 2 and at_once is not None:
            logger.warning("Stopping the capture at once.")
            at_once()

    installed: list[int] = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal)
            installed.append(sig)
        except (NotImplementedError, RuntimeError):
            pass
    return installed


class _FirstEvent(asyncio.Event):
    """An event set at a stream's first live event, that also says when.

    :func:`run_capture` passes one as *streaming*, so the manifest records the
    time the runner saw the first event, not the later moment a watcher woke.
    """

    def __init__(self) -> None:
        super().__init__()
        self.at: pd.Timestamp | None = None


@dataclass
class _StreamState:
    """What the stream updates in place, so it survives a cancellation.

    ``counts`` are rows written per kind, and ``items`` everything the stream
    yielded, written or not; ``unconfirmed`` the opening-book
    order ids nothing has named yet (L3 only); ``times["started"]`` is set at
    the first live event, and so is ``streaming``.
    """

    counts: dict[str, int]
    unconfirmed: set[Any] | None
    times: dict[str, pd.Timestamp]
    streaming: asyncio.Event | None = None
    items: int = 0


async def _run_stream(
    capturer: LiveSource,
    config: CaptureConfig,
    sink: CaptureSink,
    state: _StreamState,
    stop: asyncio.Event,
    record_error: Callable[[str, BaseException], None],
) -> None:
    """Stream until the source ends, it raises, or a signal sets *stop*.

    An exception from the stream goes to *record_error*; a stop by signal is
    not an error.
    """
    logger.info(
        "Capturer '{}': streaming for {:.1f} min", capturer.name, config.minutes
    )
    stream_task = asyncio.create_task(_stream(capturer, config, sink, state))
    stop_task = asyncio.create_task(stop.wait())
    try:
        done, _pending = await asyncio.wait(
            {stream_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        stop_task.cancel()
        await _cancel_until_done(stream_task, state)
    except asyncio.CancelledError:
        # This task was cancelled too (the supervisor stopped waiting for it,
        # or a second signal): leave the stream cancelled and do not wait for
        # it, since it may hang while it closes.
        stop_task.cancel()
        stream_task.cancel()
        stream_task.add_done_callback(_retrieve_outcome)
        raise
    # Drain cancellation cleanly.
    for t in (stream_task, stop_task):
        try:
            await t
        except (asyncio.CancelledError, Exception):  # noqa: BLE001, S110 - draining cancelled tasks; any error here is irrelevant during shutdown
            pass

    if stream_task in done and not stream_task.cancelled():
        exc = stream_task.exception()
        if exc is not None:
            record_error("stream", exc)


async def _cancel_until_done(task: asyncio.Task[Any], state: _StreamState) -> None:
    """Cancel *task* and wait for it to end, cancelling again while it streams.

    A stream that lost the cancel keeps yielding items (``state.items``),
    and only another cancel stops it.  A stream that yields nothing more is
    closing its connection, and is left to finish: another cancel would cut
    that short.
    """
    # On Python 3.11, asyncio.wait_for drops a cancel that arrives in the same
    # loop tick as the result it awaits (CPython gh-86296, fixed in 3.12).  The
    # bundled sources do not wait that way, but a plug-in source may.
    task.cancel()
    while not task.done():
        seen = state.items
        await asyncio.wait({task}, timeout=CANCEL_RETRY_SECONDS)
        if not task.done() and state.items != seen:
            task.cancel()


def _retrieve_outcome(task: asyncio.Task[Any]) -> None:
    """Read how an abandoned task ended, so asyncio does not log it as lost."""
    if not task.cancelled():
        task.exception()


async def _stream(
    capturer: LiveSource,
    config: CaptureConfig,
    sink: CaptureSink,
    state: _StreamState,
) -> None:
    """Pump the capturer's stream into *sink*, updating *state* in place.

    Counts are incremented per write (not returned) so they remain accurate
    when the task is cancelled mid-stream by a signal. For the same reason
    the unconfirmed ids are shrunk in place: an opening-book order id leaves
    them as soon as an order event or a trade names it.
    """
    counts = state.counts
    unconfirmed = state.unconfirmed
    async for kind, event, frame in capturer.stream(config):
        state.items += 1
        if "started" not in state.times:
            state.times["started"] = pd.Timestamp.now(tz="UTC")
            if isinstance(state.streaming, _FirstEvent):
                state.streaming.at = state.times["started"]
            if state.streaming is not None:
                state.streaming.set()
        if kind == "order":
            event.setdefault(ORIGIN_COLUMN, ORIGIN_STREAM)
            sink.write_order(event)
            counts["order"] += 1
            if unconfirmed:
                unconfirmed.discard(event["id"])
        elif kind == "depth":
            event.setdefault(ORIGIN_COLUMN, ORIGIN_STREAM)
            sink.write_depth(event)
            counts["depth"] += 1
        elif kind == "trade":
            sink.write_trade(event)
            counts["trade"] += 1
            if unconfirmed:
                unconfirmed.discard(event.get("buy_order_id"))
                unconfirmed.discard(event.get("sell_order_id"))
        # ``raw`` (heartbeats / subscription_succeeded) bypasses CSV writers
        # but still goes to raw.jsonl below for forensic completeness.
        if frame is not None:
            sink.write_raw(frame)
            counts["raw"] += 1
