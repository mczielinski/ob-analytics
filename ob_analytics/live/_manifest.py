"""The record of a segmented live capture: ``manifest.json``.

A capture made by :func:`~ob_analytics.live.run_capture` is a directory of
segments.  Each segment is a complete capture on its own -- the files
:func:`~ob_analytics.live._runner.run_capturer` writes, with its own
``meta.json`` -- so every segment replays with the existing loaders.  The
manifest at the top says what the segments are, why each one ended, and which
stretches of time no segment covers::

    out/
      manifest.json
      seg-0001/  orders.csv | depth.csv, trades.csv, raw.jsonl, meta.json
      seg-0002/  ...

The capture rewrites the manifest (atomically) whenever a segment starts or
ends, and every few seconds while one runs, so a capture that is killed still
leaves a manifest that says how far it got.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import pandas as pd

from ob_analytics._secrets import redact
from ob_analytics.analytics import QualityCheck, Severity
from ob_analytics.capture_record import read_fields
from ob_analytics.exceptions import ConfigError

MANIFEST_NAME = "manifest.json"
#: What the manifest's ``format`` field says, so a reader knows the file.
CAPTURE_FORMAT = "ob-analytics-capture"
#: Version of the directory layout and manifest fields.  A reader refuses a
#: version it does not know.
CAPTURE_FORMAT_VERSION = 1


class EndReason(str, Enum):
    """Why a segment ended.

    Attributes
    ----------
    FINISHED
        The capture reached the end of its run time.
    STOPPED
        A signal (Ctrl-C, SIGTERM) stopped the capture.
    ROLLED_TIME, ROLLED_SIZE
        The segment reached its time or size limit.  The next segment was
        streaming before this one stopped, so a roll loses nothing.
    FAILED
        The source raised: most often a lost connection.
    ENDED_EARLY
        The source stopped streaming before it was asked to, with no error.
    UNFINISHED
        The capture process died while the segment was open.  The restart
        closed it: see :func:`~ob_analytics.live.run_capture`.
    """

    FINISHED = "finished"
    STOPPED = "stopped"
    ROLLED_TIME = "rolled_time"
    ROLLED_SIZE = "rolled_size"
    FAILED = "failed"
    ENDED_EARLY = "ended_early"
    UNFINISHED = "unfinished"


# What a gap after a segment that ended this way is called.
_GAP_CAUSE = {
    EndReason.FAILED: "disconnect",
    EndReason.ENDED_EARLY: "disconnect",
    EndReason.UNFINISHED: "restart",
    EndReason.ROLLED_TIME: "roll",
    EndReason.ROLLED_SIZE: "roll",
    EndReason.STOPPED: "stopped",
    EndReason.FINISHED: "finished",
}


def _ts(value: Any) -> pd.Timestamp | None:
    return None if value is None else pd.Timestamp(value)


def _count(d: dict[str, Any], key: str) -> int:
    """The whole number under *key* in a manifest entry; 0 when it is absent.

    Raises :class:`TypeError` for any other value, so a count that is a word
    or a list is refused rather than read as 0.
    """
    value = d.get(key)
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"capture manifest {key} {value!r} is not a whole number")
    return value


def _iso(value: pd.Timestamp | None) -> str | None:
    return None if value is None else value.isoformat()


@dataclass
class Segment:
    """One segment of a capture, as the manifest records it.

    Attributes
    ----------
    name : str
        The segment's directory name (``seg-0001``, ...).
    started, ended : pandas.Timestamp or None
        When the segment was opened and closed.  ``ended`` is ``None`` while
        it runs.
    stream_started, stream_ended : pandas.Timestamp or None
        The stretch of time the segment covers: from its first live event to
        the moment it was asked to stop, or its stream stopped by itself.
        ``stream_started`` is ``None`` for a segment that never streamed (a
        snapshot that failed, say), or that was asked to stop before its
        first live event: it wrote rows, but covered nothing.
    heartbeat : pandas.Timestamp or None
        Last time the running capture said the segment was alive.  After a
        crash it is the latest time the segment is known to cover.
    end_reason : EndReason or None
        Why it ended; ``None`` while it runs.
    error : str or None
        The error that ended it, if any.
    n_book_events, n_trade_events : int
        Rows written to the book file and to ``trades.csv``.
    dropped : int
        Messages the source received but could not use (its ``dropped``
        counter in ``meta.json``).
    diagnostics_error : str or None
        Why the source's counters are missing from ``meta.json``, when its
        ``diagnostics()`` raised: ``dropped`` is then not known.
    book_resyncs : int
        Times the source lost the venue's stream and started again from a new
        opening book within the segment (its ``book_resyncs`` counter in
        ``meta.json``).  The changes in between were missed.
    raw_frames_skipped : int
        Frames ``raw.jsonl`` skipped because JSON cannot hold them (its
        ``n_raw_frames_skipped`` in ``meta.json``).
    """

    name: str
    started: pd.Timestamp
    ended: pd.Timestamp | None = None
    stream_started: pd.Timestamp | None = None
    stream_ended: pd.Timestamp | None = None
    heartbeat: pd.Timestamp | None = None
    end_reason: EndReason | None = None
    error: str | None = None
    n_book_events: int = 0
    n_trade_events: int = 0
    dropped: int = 0
    diagnostics_error: str | None = None
    book_resyncs: int = 0
    raw_frames_skipped: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "started": _iso(self.started),
            "ended": _iso(self.ended),
            "stream_started": _iso(self.stream_started),
            "stream_ended": _iso(self.stream_ended),
            "heartbeat": _iso(self.heartbeat),
            "end_reason": None if self.end_reason is None else self.end_reason.value,
            "error": self.error,
            "n_book_events": self.n_book_events,
            "n_trade_events": self.n_trade_events,
            "dropped": self.dropped,
            "diagnostics_error": self.diagnostics_error,
            "book_resyncs": self.book_resyncs,
            "raw_frames_skipped": self.raw_frames_skipped,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Segment:
        """Build a segment from its manifest entry.

        Raises :class:`ValueError` when ``name`` is not one plain directory
        name: a name such as ``..`` or ``a/b`` would make a reader of the
        capture leave its directory.
        """
        name = d["name"]
        # One rule for both platforms: on Windows "C:x" is a path too.
        if (
            not isinstance(name, str)
            or name in ("", ".", "..")
            or "\0" in name
            or PurePosixPath(name).name != name
            or PureWindowsPath(name).name != name
        ):
            raise ValueError(
                f"capture manifest segment name {name!r} is not one plain "
                "directory name"
            )
        reason = d.get("end_reason")
        return cls(
            name=name,
            started=pd.Timestamp(d["started"]),
            ended=_ts(d.get("ended")),
            stream_started=_ts(d.get("stream_started")),
            stream_ended=_ts(d.get("stream_ended")),
            heartbeat=_ts(d.get("heartbeat")),
            end_reason=None if reason is None else EndReason(reason),
            error=d.get("error"),
            n_book_events=_count(d, "n_book_events"),
            n_trade_events=_count(d, "n_trade_events"),
            dropped=_count(d, "dropped"),
            diagnostics_error=d.get("diagnostics_error"),
            book_resyncs=_count(d, "book_resyncs"),
            raw_frames_skipped=_count(d, "raw_frames_skipped"),
        )


@dataclass
class Gap:
    """A stretch of time no segment covers.

    Attributes
    ----------
    after : str
        The segment the gap follows.
    before : str or None
        The segment that ends it; ``None`` when the capture ended inside it.
    start, end : pandas.Timestamp
        When coverage stopped and when it resumed (or the capture ended).
    cause : str
        How the segment before it ended: ``"disconnect"``, ``"restart"`` (the
        capture process died), ``"roll"`` (a roll whose next segment was slow
        to stream, or never did), or ``"stopped"`` / ``"finished"`` (the capture was stopped
        and later started again into the same directory).
    """

    after: str
    before: str | None
    start: pd.Timestamp
    end: pd.Timestamp
    cause: str

    @property
    def seconds(self) -> float:
        return (self.end - self.start).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "after": self.after,
            "before": self.before,
            "start": _iso(self.start),
            "end": _iso(self.end),
            "seconds": self.seconds,
            "cause": self.cause,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Gap:
        return cls(
            after=d["after"],
            before=d.get("before"),
            start=pd.Timestamp(d["start"]),
            end=pd.Timestamp(d["end"]),
            cause=d["cause"],
        )


@dataclass
class CaptureManifest:
    """What ``manifest.json`` records about a segmented capture.

    Read one with :func:`read_manifest`.  ``source``, ``pair`` and ``level``
    identify the capture: a restart into the same directory must match them.
    ``declarations`` is what the source declares about its feed (the same
    fields a segment's ``meta.json`` carries).  ``restarts`` counts the times
    the capture was started again into this directory.
    """

    source: str
    pair: str
    level: str
    started: pd.Timestamp
    declarations: dict[str, Any] = field(default_factory=dict)
    roll_minutes: float | None = None
    roll_mb: float | None = None
    ended: pd.Timestamp | None = None
    restarts: int = 0
    segments: list[Segment] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)

    # -- totals -------------------------------------------------------------

    @property
    def dropped(self) -> int:
        """Messages the sources could not use, across every segment."""
        return sum(s.dropped for s in self.segments)

    @property
    def book_resyncs(self) -> int:
        """New opening books the sources took within segments, across all."""
        return sum(s.book_resyncs for s in self.segments)

    @property
    def raw_frames_skipped(self) -> int:
        """Frames ``raw.jsonl`` skipped across every segment."""
        return sum(s.raw_frames_skipped for s in self.segments)

    @property
    def gap_seconds(self) -> float:
        return sum(g.seconds for g in self.gaps)

    @property
    def unfinished(self) -> list[Segment]:
        """Segments a dead capture process left open (closed by the restart)."""
        return [s for s in self.segments if s.end_reason is EndReason.UNFINISHED]

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": CAPTURE_FORMAT,
            "version": CAPTURE_FORMAT_VERSION,
            "source": self.source,
            "pair": self.pair,
            "level": self.level,
            **self.declarations,
            "roll_minutes": self.roll_minutes,
            "roll_mb": self.roll_mb,
            "started": _iso(self.started),
            "ended": _iso(self.ended),
            "restarts": self.restarts,
            "dropped": self.dropped,
            "book_resyncs": self.book_resyncs,
            "raw_frames_skipped": self.raw_frames_skipped,
            "gap_seconds": self.gap_seconds,
            "segments": [s.to_dict() for s in self.segments],
            "gaps": [g.to_dict() for g in self.gaps],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CaptureManifest:
        if d.get("format") != CAPTURE_FORMAT:
            raise ValueError(
                f"not an ob-analytics capture manifest: {d.get('format')!r}"
            )
        if d.get("version") != CAPTURE_FORMAT_VERSION:
            raise ValueError(
                f"capture manifest version {d.get('version')!r} is not supported "
                f"(this ob-analytics reads version {CAPTURE_FORMAT_VERSION})"
            )
        for key in ("source", "pair", "level"):
            if not isinstance(d[key], str) or not d[key]:
                raise ValueError(f"capture manifest {key} {d[key]!r} is not a name")
        for key in ("segments", "gaps"):
            if not isinstance(d.get(key, []), list):
                raise TypeError(f"capture manifest {key} {d[key]!r} is not a list")
        declarations = {
            k: d[k]
            for k in ("feed_type", "trade_attribution", "sequence_kind")
            if k in d
        }
        return cls(
            source=d["source"],
            pair=d["pair"],
            level=d["level"],
            started=pd.Timestamp(d["started"]),
            declarations=declarations,
            roll_minutes=d.get("roll_minutes"),
            roll_mb=d.get("roll_mb"),
            ended=_ts(d.get("ended")),
            restarts=_count(d, "restarts"),
            segments=[Segment.from_dict(s) for s in d.get("segments") or ()],
            gaps=[Gap.from_dict(g) for g in d.get("gaps") or ()],
        )

    def write(self, root: Path) -> None:
        """Write ``manifest.json`` into *root*, replacing the old one atomically.

        The file is written beside the old one and renamed over it, so a
        reader -- or a restart after a crash -- never sees half a manifest.
        """
        root = Path(root)
        tmp = root / (MANIFEST_NAME + ".tmp")
        tmp.write_text(redact(json.dumps(self.to_dict(), indent=2)))
        os.replace(tmp, root / MANIFEST_NAME)

    # -- building -----------------------------------------------------------

    def next_segment_name(self) -> str:
        return f"seg-{len(self.segments) + 1:04d}"

    def note_streaming(self, segment: Segment) -> None:
        """Record the gap, if any, before *segment*'s first live event.

        The segment before it with coverage either is still streaming (a roll:
        the two overlap, so there is no gap) or stopped before *segment*
        started to stream, and the time between is a gap.
        """
        start = segment.stream_started
        if start is None:
            return
        previous = self._last_covered(before=segment)
        if previous is None or previous.end_reason is None:
            return
        covered_to = previous.stream_ended or previous.heartbeat
        if covered_to is None or covered_to >= start:
            return
        self.gaps.append(
            Gap(
                after=previous.name,
                before=segment.name,
                start=covered_to,
                end=start,
                cause=_GAP_CAUSE[previous.end_reason],
            )
        )

    def note_uncovered_end(self, ended: pd.Timestamp) -> None:
        """Record a gap if the capture ended while no segment was streaming.

        The last segment that streamed is looked at.  If it ran to the end
        (``finished``/``stopped``) nothing is missing.  Otherwise -- a failure,
        or a roll whose next segment never streamed -- the time from its last
        event to the end of the capture is a gap.
        """
        if any(s.end_reason is None for s in self.segments):
            return
        previous = self._last_covered(before=None)
        if previous is None or previous.end_reason is None:
            return
        if previous.end_reason in (EndReason.FINISHED, EndReason.STOPPED):
            return
        covered_to = previous.stream_ended or previous.heartbeat
        if covered_to is None or covered_to >= ended:
            return
        self.gaps.append(
            Gap(
                after=previous.name,
                before=None,
                start=covered_to,
                end=ended,
                cause=_GAP_CAUSE[previous.end_reason],
            )
        )

    def _last_covered(self, before: Segment | None) -> Segment | None:
        found = None
        for s in self.segments:
            if s is before:
                break
            if s.stream_started is not None:
                found = s
        return found

    # -- reading ------------------------------------------------------------

    @property
    def open_segments(self) -> list[Segment]:
        """Segments not yet closed: still being captured, or left by a process
        that died and has not been restarted.  Their files have no closing
        rows yet, and their row counts are not in the manifest."""
        return [s for s in self.segments if s.end_reason is None]

    def segment_dirs(self, root: Path) -> list[Path]:
        """The directories under *root* of closed segments that hold data, in order.

        *root* is the capture directory, or a ``process`` output made from it
        (which keeps the same segment names).  A segment that wrote no book
        rows -- its snapshot failed -- is left out: there is nothing to read.
        So is an open segment (see :attr:`open_segments`): it has no closing
        rows yet, so the checks would read every order still resting as a
        fault.
        """
        root = Path(root)
        return [
            root / s.name
            for s in self.segments
            if s.end_reason is not None
            and s.n_book_events > 0
            and (root / s.name).is_dir()
        ]

    def checks(self) -> tuple[QualityCheck, ...]:
        """The capture-level checks ``ob-analytics audit`` adds to each segment's.

        Each is a :attr:`~ob_analytics.analytics.Severity.WARNING`: a gap, a
        dropped message, an error or a resync is data the capture does not
        have, which the data on either side of it still describes correctly.
        ``dropped_messages`` and ``capture_errors`` are the checks
        :func:`loss_checks` makes, from every segment.
        """
        longest = max(self.gaps, key=lambda g: g.seconds, default=None)
        gap_detail = (
            f"{len(self.gaps)} gap(s), {self.gap_seconds:.1f} s in total; longest "
            f"{longest.seconds:.1f} s after {longest.after} ({longest.cause})"
            if longest is not None
            else "every stretch of the capture is covered by a segment"
        )
        unfinished = self.unfinished
        return (
            QualityCheck(
                name="capture_gaps",
                passed=not self.gaps,
                severity=Severity.WARNING,
                detail=gap_detail,
            ),
            QualityCheck(
                name="unfinished_segments",
                passed=not unfinished,
                severity=Severity.WARNING,
                detail=(
                    f"{len(unfinished)} segment(s) left open by a capture process "
                    "that died; the restart wrote their closing rows ("
                    + ", ".join(s.name for s in unfinished)
                    + ")"
                    if unfinished
                    else "every segment was closed by the process that wrote it"
                ),
            ),
            *loss_checks(
                dropped=self.dropped,
                counters_lost=[s.name for s in self.segments if s.diagnostics_error],
                # A segment a dead process left open is the check above.
                errors=[
                    f"{s.name}: {s.error}"
                    for s in self.segments
                    if s.error and s.end_reason is not EndReason.UNFINISHED
                ],
            ),
            QualityCheck(
                name="book_resyncs",
                passed=self.book_resyncs == 0,
                severity=Severity.WARNING,
                detail=(
                    f"{self.book_resyncs} time(s) a source lost the venue's "
                    "stream and started again from a new opening book; the "
                    "changes in between were missed"
                ),
            ),
        )

    def render(self) -> str:
        """Return a fixed-width, human-readable report block."""
        lines = [
            "Capture summary",
            f"  source / pair / level : {self.source} / {self.pair} / {self.level}",
            f"  started / ended       : {self.started} / {self.ended}",
            f"  segments / restarts   : {len(self.segments)} / {self.restarts}",
            f"  gaps                  : {len(self.gaps)} ({self.gap_seconds:.1f} s)",
            f"  dropped messages      : {self.dropped}",
            f"  book resyncs          : {self.book_resyncs}",
        ]
        if self.raw_frames_skipped:
            # Only when there were some: a capture run with --no-raw has no
            # raw.jsonl, and the manifest does not record that.
            lines.append(f"  raw frames skipped    : {self.raw_frames_skipped}")
        for s in self.segments:
            reason = "running" if s.end_reason is None else s.end_reason.value
            lines.append(
                f"    {s.name}: {s.stream_started} -> {s.stream_ended} [{reason}]"
                + (f" {s.error}" if s.error else "")
            )
        for g in self.gaps:
            lines.append(
                f"    gap after {g.after}: {g.start} -> {g.end} "
                f"({g.seconds:.1f} s, {g.cause})"
            )
        failed = [c for c in self.checks() if not c.passed]
        if failed:
            lines.append(f"Checks: {len(failed)} warning(s)")
            lines += [
                f"  {c.severity.value.upper():<7} {c.name}: {c.detail}" for c in failed
            ]
        else:
            lines.append("Checks: all passed")
        return "\n".join(lines)


def loss_checks(
    *, dropped: int, counters_lost: Sequence[str], errors: Sequence[str]
) -> tuple[QualityCheck, QualityCheck]:
    """The checks for what a capture's sources lost, as ``audit`` reports them.

    ``dropped_messages`` fails when a source left out messages it could not
    use (*dropped*), or when the count is not known because a source's
    counters were lost (*counters_lost* names where).  ``capture_errors``
    fails when the capture recorded an error (*errors*, each one line): one
    that ended a segment, or a segment that was slow to stop.
    Both are :attr:`~ob_analytics.analytics.Severity.WARNING`.
    """
    dropped_detail = f"{dropped} message(s) the sources could not use"
    if counters_lost:
        dropped_detail += (
            "; the counters of "
            + ", ".join(counters_lost)
            + " were lost (diagnostics_error in meta.json), so more may be missing"
        )
    return (
        QualityCheck(
            name="dropped_messages",
            passed=dropped == 0 and not counters_lost,
            severity=Severity.WARNING,
            detail=dropped_detail,
        ),
        QualityCheck(
            name="capture_errors",
            passed=not errors,
            severity=Severity.WARNING,
            detail=(
                f"{len(errors)} error(s) while capturing: " + "; ".join(errors)
                if errors
                else "no error while capturing"
            ),
        ),
    )


def record_checks(path: Path) -> tuple[QualityCheck, ...]:
    """The :func:`loss_checks` of the single capture at *path*, from its ``meta.json``.

    *path* is a capture made by
    :func:`~ob_analytics.live._runner.run_capturer`, or the output of
    ``ob-analytics process`` made from one.  Empty when there is no
    ``meta.json``: the input is not a live capture.
    """
    try:
        fields = read_fields(path)
    except OSError:
        # Reading the capture's record already warned that it cannot be read.
        return ()
    if fields is None:
        return ()
    error = fields.get("capture_error")
    dropped = fields.get("dropped")
    return loss_checks(
        dropped=dropped if isinstance(dropped, int) and dropped > 0 else 0,
        counters_lost=["this capture"] if fields.get("diagnostics_error") else [],
        errors=[f"{fields.get('capture_error_phase')}: {error}"] if error else [],
    )


def read_manifest(path: Path) -> CaptureManifest | None:
    """Read the manifest of the segmented capture at *path*.

    *path* is the capture directory (or its ``manifest.json``).  Returns
    ``None`` when there is no manifest: a single capture made by
    :func:`~ob_analytics.live._runner.run_capturer`, or a processed output.
    Raises :class:`~ob_analytics.exceptions.ConfigError` for a manifest this
    version cannot read: one that is not a JSON object, lacks a key, or holds
    a value of the wrong kind, such as a segment name that is not one plain
    directory name.
    """
    path = Path(path)
    file = path if path.name == MANIFEST_NAME else path / MANIFEST_NAME
    if not file.is_file():
        return None
    try:
        d = json.loads(file.read_text())
        if not isinstance(d, dict):
            raise TypeError("it does not hold a JSON object")
        return CaptureManifest.from_dict(d)
    except (ValueError, KeyError, TypeError) as exc:
        why = f"it has no {exc} key" if isinstance(exc, KeyError) else str(exc)
        raise ConfigError(
            f"{file} is not a capture manifest this version can read: {why}"
        ) from exc
