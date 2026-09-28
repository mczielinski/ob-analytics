"""run_capture: a capture that survives disconnects, rolls, and restarts (#150).

A scripted source stands in for a venue, so every break -- a dropped
connection, a failed snapshot, a crash -- happens on cue and with no network.
"""

from __future__ import annotations

import asyncio
import copy
import csv
import json
import time
from collections.abc import AsyncIterator
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from ob_analytics.config import SourceSettings
from ob_analytics.exceptions import ConfigError
from ob_analytics.live import (
    CaptureConfig,
    CaptureManifest,
    EndReason,
    Segment,
    _runner,
    _supervisor,
    read_manifest,
    run_capture,
)
from ob_analytics.live._base import EventDict
from ob_analytics.protocols import FeedType, Level

# ---------------------------------------------------------------------------
# A scripted L3 source
# ---------------------------------------------------------------------------

OK = "ok"
FAIL_SNAPSHOT = "fail_snapshot"
BIG_SNAPSHOT = "big_snapshot"  # streams like OK, from a 5,000-order book
SILENT = "silent"  # connects, sends nothing for 0.6 s, then drops
# Streams like OK, but carries on after the first cancel, as a stream waiting
# in asyncio.wait_for does on Python 3.11 when a message arrives with it.
DEAF_ONCE = "deaf_once"
# Like DEAF_ONCE, but yields only heartbeats: items the runner writes nowhere.
DEAF_ONCE_RAW = "deaf_once_raw"
# Streams like OK, but never finishes closing its connection once stopped.
HANG_ON_STOP = "hang_on_stop"
# Its opening snapshot never finishes, so it never streams.
STUCK_SNAPSHOT = "stuck_snapshot"


def _disconnect_after(n: int) -> str:
    return f"disconnect:{n}"


def _slow_close(seconds: float) -> str:
    """Streams like OK, but takes *seconds* to close its connection once stopped."""
    return f"slow_close:{seconds}"


def _slow_snapshot(seconds: float) -> str:
    """Streams like OK, after an opening snapshot that takes *seconds*.

    Like Bitstamp, it replays a message it buffered during the snapshot the
    moment its stream starts, before it waits for anything.
    """
    return f"slow_snapshot:{seconds}"


def _stubborn_shutdown(seconds: float) -> str:
    """Streams like OK, then spends *seconds* on its closing rows, deaf to cancels."""
    return f"stubborn_shutdown:{seconds}"


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


class _ScriptedSource:
    """Opens a two-order book, then adds and removes one order every few ms.

    *plan* says what each new instance does, in turn: stream until stopped,
    drop the connection after *n* events, or fail its snapshot.  The last
    entry repeats.
    """

    name = "scripted"
    level = Level.L3
    feed_type = FeedType.MATCHED_BOOK
    settings = SourceSettings()

    def __init__(self, behaviour: str) -> None:
        self._behaviour = behaviour
        self._open: dict[int, tuple[float, str]] = {}
        self._next_id = 1000
        self._closed = False

    async def snapshot(self, config: CaptureConfig) -> AsyncIterator[EventDict]:
        if self._behaviour == FAIL_SNAPSHOT:
            raise ConnectionError("venue unreachable")
        if self._behaviour == STUCK_SNAPSHOT:
            await asyncio.sleep(3600)
        if self._behaviour.startswith("slow_snapshot"):
            await asyncio.sleep(float(self._behaviour.split(":")[1]))
        ts = _now()
        book = [(1, 100.0, "bid"), (2, 101.0, "ask")]
        if self._behaviour == BIG_SNAPSHOT:
            book += [(10 + i, 50.0 + i / 1000, "bid") for i in range(5000)]
        for oid, price, side in book:
            self._open[oid] = (price, side)
            yield {
                "id": oid,
                "timestamp": ts,
                "exchange_timestamp": ts,
                "price": price,
                "volume": 1.0,
                "action": "created",
                "direction": side,
            }

    async def stream(
        self, config: CaptureConfig
    ) -> AsyncIterator[tuple[str, EventDict, Any]]:
        limit = (
            int(self._behaviour.split(":")[1])
            if self._behaviour.startswith("disconnect")
            else None
        )
        if self._behaviour == SILENT:
            await asyncio.sleep(0.6)
            raise ConnectionError("connection closed by venue")
        sent = 0
        ignore_cancel = self._behaviour in (DEAF_ONCE, DEAF_ONCE_RAW)
        replay_first = self._behaviour.startswith("slow_snapshot")
        deadline = time.monotonic() + config.minutes * 60
        try:
            while time.monotonic() < deadline:
                if limit is not None and sent >= limit:
                    raise ConnectionError("connection closed by venue")
                try:
                    if not (replay_first and sent == 0):
                        await asyncio.sleep(0.005)
                except asyncio.CancelledError:
                    if not ignore_cancel:
                        raise
                    ignore_cancel = False
                if self._behaviour == DEAF_ONCE_RAW:
                    yield ("raw", {}, None)
                    continue
                ts = _now()
                oid = self._next_id
                if oid in self._open:
                    price, side = self._open.pop(oid)
                    action = "deleted"
                    self._next_id += 1
                else:
                    price, side = 99.0, "bid"
                    self._open[oid] = (price, side)
                    action = "created"
                sent += 1
                yield (
                    "order",
                    {
                        "id": oid,
                        "timestamp": ts,
                        "exchange_timestamp": ts,
                        "price": price,
                        "volume": 0.5,
                        "action": action,
                        "direction": side,
                    },
                    None,
                )
        finally:
            if self._behaviour.startswith("slow_close"):
                await asyncio.sleep(float(self._behaviour.split(":")[1]))
            elif self._behaviour == HANG_ON_STOP:
                await asyncio.sleep(3600)
            self._closed = True

    async def shutdown_synthetic_events(self) -> AsyncIterator[EventDict]:
        if self._behaviour.startswith("stubborn_shutdown"):
            until = time.monotonic() + float(self._behaviour.split(":")[1])
            while (left := until - time.monotonic()) > 0:
                try:
                    await asyncio.sleep(left)
                except asyncio.CancelledError:
                    pass
        ts = _now()
        for oid, (price, side) in list(self._open.items()):
            yield {
                "id": oid,
                "timestamp": ts,
                "exchange_timestamp": ts,
                "price": price,
                "volume": 0.0,
                "action": "deleted",
                "direction": side,
            }
        self._open.clear()

    def diagnostics(self) -> dict[str, Any]:
        return {"closed_cleanly": self._closed}


def _factory(plan: list[str]):
    remaining = list(plan)

    def make() -> _ScriptedSource:
        behaviour = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return _ScriptedSource(behaviour)

    return make


@pytest.fixture(autouse=True)
def _fast_clock(monkeypatch):
    """Shrink every wait so a capture of a few segments takes a second or two."""
    monkeypatch.setattr(_supervisor, "BACKOFF_START_SECONDS", 0.05)
    monkeypatch.setattr(_supervisor, "BACKOFF_MAX_SECONDS", 0.2)
    monkeypatch.setattr(_supervisor, "POLL_SECONDS", 0.02)
    monkeypatch.setattr(_supervisor, "HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(_supervisor, "HANDOVER_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(_supervisor, "STOP_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(_supervisor, "STOP_GRACE_SECONDS", 0.3)
    monkeypatch.setattr(_runner, "CANCEL_RETRY_SECONDS", 0.02)


def _capture(tmp_path: Path, plan: list[str], seconds: float, **config: Any):
    cfg = CaptureConfig(
        pair="btcusd",
        out_dir=tmp_path / "cap",
        minutes=seconds / 60,
        keep_raw=False,
        **config,
    )
    return asyncio.run(run_capture(_factory(plan), cfg))


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def _every_order_closed(orders: Path) -> bool:
    open_ids: set[str] = set()
    for row in _rows(orders):
        if row["action"] == "deleted":
            open_ids.discard(row["id"])
        else:
            open_ids.add(row["id"])
    return not open_ids


# ---------------------------------------------------------------------------
# Disconnects
# ---------------------------------------------------------------------------


class TestDisconnect:
    def test_a_dropped_connection_starts_a_new_segment_and_records_the_gap(
        self, tmp_path
    ):
        run = _capture(tmp_path, [_disconnect_after(6), OK], seconds=1.0)

        assert run.error is None
        first, second = run.manifest.segments
        assert first.end_reason is EndReason.FAILED
        assert "connection closed" in (first.error or "")
        assert second.end_reason is EndReason.FINISHED

        (gap,) = run.manifest.gaps
        assert (gap.after, gap.before, gap.cause) == (
            "seg-0001",
            "seg-0002",
            "disconnect",
        )
        assert gap.start == first.stream_ended
        assert gap.end == second.stream_started
        assert gap.seconds > 0

    def test_every_segment_replays_on_its_own(self, tmp_path):
        """Each segment opens from a snapshot and closes every order it opened."""
        run = _capture(tmp_path, [_disconnect_after(6), OK], seconds=1.0)
        for name in ("seg-0001", "seg-0002"):
            seg = run.out_dir / name
            rows = _rows(seg / "orders.csv")
            assert rows[0]["origin"] == "snapshot"
            assert _every_order_closed(seg / "orders.csv")
            meta = json.loads((seg / "meta.json").read_text())
            assert meta["source"] == "scripted"
            assert "provisional" not in meta

    def test_repeated_failures_wait_longer_each_time(self, tmp_path, monkeypatch):
        waits: list[float] = []
        real = _supervisor._Supervisor._next_backoff

        def spy(self):
            wait = real(self)
            waits.append(wait)
            return wait

        monkeypatch.setattr(_supervisor._Supervisor, "_next_backoff", spy)
        _capture(tmp_path, [_disconnect_after(1)], seconds=1.0)
        assert waits[:4] == [0.05, 0.1, 0.2, 0.2]

    def test_a_capture_that_cannot_start_stops_at_once(self, tmp_path):
        """A first snapshot that fails means wrong settings: no retrying."""
        started = time.monotonic()
        run = _capture(tmp_path, [FAIL_SNAPSHOT], seconds=30.0)
        assert time.monotonic() - started < 10
        assert run.error is not None and "venue unreachable" in run.error
        (segment,) = run.manifest.segments
        assert segment.end_reason is EndReason.FAILED
        assert run.manifest.gaps == []

    def test_a_stream_that_fails_before_its_first_event_stops_at_once(self, tmp_path):
        """A source with no snapshot (cryptofeed) rejects a bad pair in stream."""
        started = time.monotonic()
        run = _capture(tmp_path, [_disconnect_after(0)], seconds=30.0)
        assert time.monotonic() - started < 10
        assert run.error is not None and "connection closed" in run.error

    def test_a_failed_snapshot_after_streaming_is_waited_out(self, tmp_path):
        run = _capture(
            tmp_path,
            [_disconnect_after(3), FAIL_SNAPSHOT, FAIL_SNAPSHOT, OK],
            seconds=1.5,
        )
        assert run.error is None
        reasons = [s.end_reason for s in run.manifest.segments]
        assert reasons == [
            EndReason.FAILED,
            EndReason.FAILED,
            EndReason.FAILED,
            EndReason.FINISHED,
        ]
        # One gap, from the first segment to the one that streamed again.
        (gap,) = run.manifest.gaps
        assert (gap.after, gap.before) == ("seg-0001", "seg-0004")


# ---------------------------------------------------------------------------
# Rolls
# ---------------------------------------------------------------------------


class TestRoll:
    def test_a_time_roll_overlaps_the_segments_and_leaves_no_gap(self, tmp_path):
        run = _capture(tmp_path, [OK], seconds=1.2, roll_minutes=0.3 / 60)

        segments = run.manifest.segments
        assert len(segments) >= 3
        assert {s.end_reason for s in segments[:-1]} == {EndReason.ROLLED_TIME}
        assert segments[-1].end_reason is EndReason.FINISHED
        assert run.manifest.gaps == []
        for earlier, later in pairwise(segments):
            assert later.stream_started is not None
            assert earlier.stream_ended is not None
            assert later.stream_started <= earlier.stream_ended

    def test_a_stream_that_ignores_one_cancel_still_rolls_and_stops(self, tmp_path):
        # Each source runs its stream a minute past the capture's end, so a
        # segment that is not stopped holds the capture for that minute.
        started = time.monotonic()
        run = _capture(tmp_path, [DEAF_ONCE], seconds=1.2, roll_minutes=0.3 / 60)

        assert time.monotonic() - started < 10
        segments = run.manifest.segments
        assert len(segments) >= 3
        assert {s.end_reason for s in segments[:-1]} == {EndReason.ROLLED_TIME}
        assert segments[-1].end_reason is EndReason.FINISHED
        for earlier, later in pairwise(segments):
            assert earlier.stream_ended is not None
            assert later.stream_started is not None
            assert earlier.stream_ended - later.stream_started < pd.Timedelta(
                seconds=0.5
            )

    def test_a_stream_of_unwritten_items_that_ignores_one_cancel_stops(self, tmp_path):
        # Nothing it yields is written, so only the count of items shows that
        # it is still running and needs another cancel.
        run = _capture(tmp_path, [DEAF_ONCE_RAW], seconds=0.8, roll_minutes=0.3 / 60)

        segments = run.manifest.segments
        assert len(segments) >= 2
        assert all(s.error is None for s in segments)

    def test_a_stream_that_is_closing_is_not_cancelled_again(
        self, tmp_path, monkeypatch
    ):
        # Closing takes 0.1 s, five times the retry interval: cancelling again
        # would cut it short and leave the connection open.  The stop limit is
        # far above it, so only the retry interval is under test.
        monkeypatch.setattr(_supervisor, "STOP_TIMEOUT_SECONDS", 5.0)
        run = _capture(tmp_path, [_slow_close(0.1)], seconds=0.8, roll_minutes=0.3 / 60)

        segments = run.manifest.segments
        assert len(segments) >= 2
        assert all(s.error is None for s in segments)
        for segment in segments:
            meta = json.loads((run.out_dir / segment.name / "meta.json").read_text())
            assert meta["closed_cleanly"] is True, segment.name

    def test_a_segment_that_does_not_stop_is_cancelled_and_closed(self, tmp_path):
        started = time.monotonic()
        run = _capture(tmp_path, [HANG_ON_STOP], seconds=1.0, roll_minutes=0.3 / 60)

        # Without the limit, the first roll would wait for its stream an hour.
        assert time.monotonic() - started < 10
        assert run.error is None
        segments = run.manifest.segments
        assert len(segments) >= 2
        # Each stopped when asked, only late: the reason stays, the lateness
        # is the error, and nothing is recorded as a gap.
        assert {s.end_reason for s in segments[:-1]} == {EndReason.ROLLED_TIME}
        assert segments[-1].end_reason is EndReason.FINISHED
        assert run.manifest.gaps == []
        for segment in segments:
            assert "did not stop within" in (segment.error or "")
            seg_dir = run.out_dir / segment.name
            assert _every_order_closed(seg_dir / "orders.csv"), segment.name
            meta = json.loads((seg_dir / "meta.json").read_text())
            assert "did not stop within" in meta["capture_error"]

    def test_a_segment_covers_the_market_only_until_asked_to_stop(
        self, tmp_path, monkeypatch
    ):
        # Closing takes 0.5 s, well under the limit.  The market is covered by
        # the next segment from the moment it streams, and the old one is
        # asked to stop straight after: its close is not coverage.
        monkeypatch.setattr(_supervisor, "STOP_TIMEOUT_SECONDS", 5.0)
        run = _capture(tmp_path, [_slow_close(0.5)], seconds=1.0, roll_minutes=0.3 / 60)

        segments = run.manifest.segments
        assert len(segments) >= 2
        for earlier, later in pairwise(segments):
            assert earlier.stream_ended is not None
            assert later.stream_started is not None
            assert earlier.stream_ended - later.stream_started < pd.Timedelta(
                seconds=0.25
            )

    def test_a_segment_asked_to_stop_before_it_streams_covers_nothing(
        self, tmp_path, monkeypatch
    ):
        # The roll at 0.3 s starts the next segment, and the capture ends at
        # 0.8 s while that segment is still taking its 1 s snapshot.  Once the
        # snapshot is done, its stream replays a message before it sees the
        # stop, so its first event comes after the stop.  The stop limit is
        # longer than the snapshot, so the segment is not cancelled first.
        monkeypatch.setattr(_supervisor, "STOP_TIMEOUT_SECONDS", 5.0)
        run = _capture(
            tmp_path, [OK, _slow_snapshot(1.0)], seconds=0.8, roll_minutes=0.3 / 60
        )

        first, second = run.manifest.segments
        assert first.stream_started is not None
        assert first.stream_ended is not None
        assert first.stream_ended >= first.stream_started
        # The late one covers nothing, so it neither ends nor starts a gap...
        assert second.stream_started is None and second.stream_ended is None
        assert run.manifest.gaps == []
        # ...but keeps the rows it wrote.
        rows = _rows(run.out_dir / second.name / "orders.csv")
        assert any(r["origin"] == "stream" for r in rows)
        # Each meta.json says the same as the manifest.
        for segment in (first, second):
            meta = json.loads((run.out_dir / segment.name / "meta.json").read_text())
            for key in ("stream_started", "stream_ended"):
                recorded = meta[key]
                expected = getattr(segment, key)
                assert (recorded is None) == (expected is None), (segment.name, key)
                if recorded is not None:
                    assert pd.Timestamp(recorded) == expected, (segment.name, key)

    def test_a_stuck_segment_covers_only_until_asked_to_stop(
        self, tmp_path, monkeypatch
    ):
        # The next segment's snapshot takes 1 s, so the roll gives up waiting
        # after 0.3 s and stops the current one, which then hangs.  Its
        # heartbeat must not run on while it hangs, or the gap before the
        # next segment would look shorter than it is.
        monkeypatch.setattr(_supervisor, "HANDOVER_TIMEOUT_SECONDS", 0.3)
        run = _capture(
            tmp_path,
            [HANG_ON_STOP, _slow_snapshot(1.0), OK],
            seconds=1.8,
            roll_minutes=0.3 / 60,
        )

        first = run.manifest.segments[0]
        assert "did not stop within" in (first.error or "")
        assert first.stream_ended is not None and first.ended is not None
        # It hung for the whole 0.3 s limit after it was asked to stop.
        assert first.ended - first.stream_ended >= pd.Timedelta(seconds=0.25)
        (gap,) = [g for g in run.manifest.gaps if g.after == "seg-0001"]
        assert gap.cause == "roll"
        assert gap.start == first.stream_ended

    def test_a_late_stop_that_ends_by_itself_keeps_its_own_result(
        self, tmp_path, monkeypatch
    ):
        # Its closing rows take 0.45 s: past the 0.3 s limit, and it ignores
        # the cancel, but it is done well within the 2 s grace.
        monkeypatch.setattr(_supervisor, "STOP_GRACE_SECONDS", 2.0)
        run = _capture(tmp_path, [_stubborn_shutdown(0.45)], seconds=0.6)

        (segment,) = run.manifest.segments
        assert segment.end_reason is EndReason.FINISHED
        assert "took more than" in (segment.error or "")
        assert run.manifest.gaps == []
        seg_dir = run.out_dir / segment.name
        assert _every_order_closed(seg_dir / "orders.csv")
        # The runner's own meta.json, not one rebuilt from the files.
        meta = json.loads((seg_dir / "meta.json").read_text())
        assert "unfinished" not in meta
        assert "n_snapshot_unconfirmed" in meta

    def test_a_segment_that_ignores_the_cancel_is_left_behind(self, tmp_path):
        run = _capture(tmp_path, [_stubborn_shutdown(1.5)], seconds=0.6)

        (segment,) = run.manifest.segments
        assert segment.end_reason is EndReason.FINISHED
        assert "even when cancelled" in (segment.error or "")
        # It covers the market only until it was asked to stop, not until it
        # was given up 0.6 s later.
        assert segment.stream_ended is not None and segment.ended is not None
        assert segment.ended - segment.stream_ended >= pd.Timedelta(seconds=0.5)

    def test_a_stuck_segment_that_then_fails_says_how(self, tmp_path, monkeypatch):
        def full_disk(self: Any, result: Any) -> None:
            raise OSError("No space left on device")

        monkeypatch.setattr(_runner.FileCaptureSink, "finalize", full_disk)
        run = _capture(tmp_path, [HANG_ON_STOP], seconds=0.5)

        (segment,) = run.manifest.segments
        assert "did not stop within" in (segment.error or "")
        assert "No space left on device" in (segment.error or "")

    def test_a_capture_goes_on_when_closing_a_stuck_segment_fails(
        self, tmp_path, monkeypatch
    ):
        def disk_full(*args: Any, **kwargs: Any) -> Any:
            raise OSError("No space left on device")

        monkeypatch.setattr(_supervisor, "_close_segment_files", disk_full)
        run = _capture(tmp_path, [HANG_ON_STOP], seconds=1.0, roll_minutes=0.3 / 60)

        assert run.error is None
        segments = run.manifest.segments
        assert len(segments) >= 2
        for segment in segments:
            assert "closing its files failed" in (segment.error or "")
            assert "No space left" in (segment.error or "")

    def test_segments_still_running_at_the_end_stop_together(
        self, tmp_path, monkeypatch
    ):
        # The capture ends while the next segment is still taking its snapshot,
        # so both are running, and neither stops by itself.
        monkeypatch.setattr(_supervisor, "STOP_TIMEOUT_SECONDS", 2.0)
        monkeypatch.setattr(_supervisor, "STOP_GRACE_SECONDS", 0.1)
        started = time.monotonic()
        run = _capture(
            tmp_path,
            [HANG_ON_STOP, STUCK_SNAPSHOT],
            seconds=0.6,
            roll_minutes=0.3 / 60,
        )

        # One limit (2 s) after the end, not one per segment (4 s).
        assert time.monotonic() - started < 3.7
        first, second = run.manifest.segments
        assert first.end_reason is second.end_reason is EndReason.FINISHED
        assert "did not stop within" in (first.error or "")
        assert "did not stop within" in (second.error or "")

    def test_a_size_roll(self, tmp_path):
        run = _capture(tmp_path, [OK], seconds=1.0, roll_mb=0.002)
        reasons = [s.end_reason for s in run.manifest.segments]
        assert EndReason.ROLLED_SIZE in reasons
        assert run.manifest.gaps == []

    def test_the_opening_snapshot_does_not_count_toward_a_size_roll(self, tmp_path):
        """Else a limit below the snapshot's size would roll without end."""
        run = _capture(tmp_path, [BIG_SNAPSHOT], seconds=1.0, roll_mb=0.05)
        (seg,) = run.manifest.segments
        assert seg.end_reason is EndReason.FINISHED
        assert (run.out_dir / "seg-0001" / "orders.csv").stat().st_size > 50_000

    @pytest.mark.parametrize("limit", [{"roll_minutes": 0}, {"roll_mb": -1.0}])
    def test_a_limit_must_be_above_zero(self, tmp_path, limit):
        with pytest.raises(ConfigError, match="greater than 0"):
            _capture(tmp_path, [OK], seconds=0.3, **limit)
        assert not (tmp_path / "cap").exists()

    def test_a_roll_whose_next_segment_never_streams_leaves_a_recorded_gap(
        self, tmp_path, monkeypatch
    ):
        """The venue goes quiet at a roll and never comes back."""
        monkeypatch.setattr(_supervisor, "HANDOVER_TIMEOUT_SECONDS", 0.2)
        run = _capture(
            tmp_path, [OK, SILENT, FAIL_SNAPSHOT], seconds=1.5, roll_minutes=0.3 / 60
        )
        first = run.manifest.segments[0]
        assert first.end_reason is EndReason.ROLLED_TIME
        assert all(s.stream_started is None for s in run.manifest.segments[1:])

        (gap,) = run.manifest.gaps
        assert (gap.after, gap.before, gap.cause) == ("seg-0001", None, "roll")
        assert gap.start == first.stream_ended
        assert gap.end == run.manifest.ended
        assert not next(
            c for c in run.manifest.checks() if c.name == "capture_gaps"
        ).passed

    def test_a_next_segment_that_fails_leaves_the_current_one_running(self, tmp_path):
        run = _capture(
            tmp_path, [OK, FAIL_SNAPSHOT, OK], seconds=1.2, roll_minutes=0.3 / 60
        )
        segments = run.manifest.segments
        assert segments[1].end_reason is EndReason.FAILED
        # The first segment carried on until a later roll took over from it.
        assert segments[0].end_reason is EndReason.ROLLED_TIME
        assert segments[2].stream_started is not None
        assert segments[2].stream_started <= segments[0].stream_ended
        assert run.manifest.gaps == []


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


class TestManifest:
    def test_round_trips_through_the_file(self, tmp_path):
        run = _capture(tmp_path, [_disconnect_after(4), OK], seconds=0.8)
        read = read_manifest(run.out_dir)
        assert read is not None
        assert read.to_dict() == run.manifest.to_dict()
        raw = json.loads((run.out_dir / "manifest.json").read_text())
        assert raw["format"] == "ob-analytics-capture"
        assert raw["version"] == 1
        assert (raw["source"], raw["pair"], raw["level"]) == (
            "scripted",
            "btcusd",
            "L3",
        )
        assert raw["feed_type"] == "matched_book"
        assert raw["dropped"] == 0
        assert raw["gap_seconds"] == pytest.approx(run.manifest.gaps[0].seconds)

    def test_a_running_segment_keeps_a_provisional_meta_on_disk(self, tmp_path):
        cfg = CaptureConfig(
            pair="btcusd", out_dir=tmp_path / "cap", minutes=1.0 / 60, keep_raw=False
        )

        async def peek() -> dict[str, Any]:
            task = asyncio.create_task(run_capture(_factory([OK]), cfg))
            await asyncio.sleep(0.4)
            meta = json.loads((tmp_path / "cap" / "seg-0001" / "meta.json").read_text())
            manifest = json.loads((tmp_path / "cap" / "manifest.json").read_text())
            await task
            return {"meta": meta, "manifest": manifest}

        seen = asyncio.run(peek())
        assert seen["meta"]["provisional"] is True
        assert seen["meta"]["source"] == "scripted"
        assert seen["manifest"]["segments"][0]["heartbeat"] is not None
        assert seen["manifest"]["ended"] is None

    def test_a_capture_will_not_continue_a_manifest_it_cannot_read(self, tmp_path):
        root = tmp_path / "cap"
        root.mkdir()
        (root / "manifest.json").write_text(
            json.dumps({"format": "ob-analytics-capture", "version": 99})
        )
        with pytest.raises(ConfigError, match="version 99"):
            _capture(tmp_path, [OK], seconds=0.3)

    def test_an_unknown_version_is_refused(self, tmp_path):
        (tmp_path / "manifest.json").write_text(
            json.dumps({"format": "ob-analytics-capture", "version": 99})
        )
        with pytest.raises(ValueError, match="version 99"):
            read_manifest(tmp_path)

    def test_checks_flag_gaps_but_pass_a_clean_capture(self, tmp_path):
        clean = _capture(tmp_path / "a", [OK], seconds=0.4)
        assert all(c.passed for c in clean.manifest.checks())

        gappy = _capture(tmp_path / "b", [_disconnect_after(4), OK], seconds=0.8)
        failed = {c.name for c in gappy.manifest.checks() if not c.passed}
        assert failed == {"capture_gaps"}
        assert "Capture summary" in gappy.manifest.render()


# ---------------------------------------------------------------------------
# Restarts
# ---------------------------------------------------------------------------


def _crashed_capture(root: Path) -> None:
    """The state a capture killed mid-segment leaves behind.

    Order 1 is still open, order 2 was deleted, and the last line of
    orders.csv is cut short.  The manifest's last heartbeat is the latest
    word from the dead process.
    """
    seg = root / "seg-0001"
    seg.mkdir(parents=True)
    t0 = pd.Timestamp("2026-09-26 10:00:00", tz="UTC")
    ms = int(t0.value // 1_000_000)
    header = "id,timestamp,exchange_timestamp,price,volume,action,direction,sequence,origin\n"
    (seg / "orders.csv").write_text(
        header
        + f"1,{ms},{ms},100.0,1.0,created,bid,,snapshot\n"
        + f"2,{ms},{ms},101.0,1.0,created,ask,,snapshot\n"
        + f"2,{ms + 500},{ms + 500},101.0,1.0,deleted,ask,,stream\n"
        + f"3,{ms + 900},{ms + 900},99.0,0.5,cre"  # cut off mid-line
    )
    # Killed before its first flush: the trades file holds nothing at all.
    (seg / "trades.csv").write_text("")
    (seg / "meta.json").write_text(
        json.dumps(
            {"source": "scripted", "feed_type": "matched_book", "provisional": True}
        )
    )
    manifest = CaptureManifest(
        source="scripted",
        pair="btcusd",
        level="L3",
        started=t0,
        declarations={"feed_type": "matched_book", "trade_attribution": "both"},
        segments=[
            Segment(
                name="seg-0001",
                started=t0,
                stream_started=t0,
                heartbeat=t0 + pd.Timedelta(seconds=2),
            )
        ],
    )
    manifest.write(root)


def _first_event_recorded(
    tmp_path: Path, first_event_s: float, stop_s: float | None
) -> tuple[pd.Timestamp | None, pd.Timestamp]:
    """What the supervisor records as a segment's start, and the first event.

    The first event came *first_event_s* seconds from now, and the segment was
    asked to stop at *stop_s* (or never).  The stop flag is set whenever a stop
    time is given, as it would be by the time the watcher wakes.
    """

    async def run() -> tuple[pd.Timestamp | None, pd.Timestamp]:
        t0 = _now()
        manifest = CaptureManifest(
            source="scripted", pair="btcusd", level="L3", started=t0
        )
        config = CaptureConfig(pair="btcusd", out_dir=tmp_path, minutes=1.0)
        supervisor = _supervisor._Supervisor(
            _factory([OK]), config, tmp_path, manifest, asyncio.Event()
        )
        segment = Segment(name="seg-0001", started=t0)
        manifest.segments.append(segment)
        running = _supervisor._Running(
            segment=segment,
            source=_ScriptedSource(OK),
            sink=_runner.FileCaptureSink(tmp_path / "seg-0001", keep_raw=False),
            stop=asyncio.Event(),
            streaming=_runner._FirstEvent(),
            # The watcher never reads the task.
            task=cast(Any, asyncio.create_task(asyncio.sleep(0))),
            opened=time.monotonic(),
        )
        if stop_s is not None:
            running.stop_requested = t0 + pd.Timedelta(seconds=stop_s)
            running.stop.set()
        first_event = t0 + pd.Timedelta(seconds=first_event_s)
        running.streaming.at = first_event
        running.streaming.set()
        await supervisor._on_streaming(running)
        await running.task
        return segment.stream_started, first_event

    return asyncio.run(run())


class TestFirstEvent:
    def test_the_runners_time_is_recorded(self, tmp_path):
        # Not the later moment the supervisor's watcher woke.
        recorded, first_event = _first_event_recorded(tmp_path, -5.0, None)
        assert recorded == first_event

    def test_an_event_just_before_the_stop_counts(self, tmp_path):
        # The stop flag is already set when the watcher wakes, but the first
        # event came first: the segment covered the market until the stop.
        recorded, first_event = _first_event_recorded(tmp_path, 0.0, 0.001)
        assert recorded == first_event

    def test_an_event_after_the_stop_does_not_count(self, tmp_path):
        recorded, _ = _first_event_recorded(tmp_path, 0.001, 0.0)
        assert recorded is None


class TestCloseSegmentFiles:
    def test_changes_the_files_but_not_the_segment_or_manifest(self, tmp_path):
        """It runs in a thread, so it must leave what the capture reads alone."""
        root = tmp_path / "cap"
        _crashed_capture(root)
        manifest = read_manifest(root)
        assert manifest is not None
        segment = manifest.segments[0]
        before = (copy.deepcopy(segment), copy.deepcopy(manifest.to_dict()))

        closed = _supervisor._close_segment_files(
            root / segment.name, segment, manifest, "it did not stop"
        )

        assert (segment, manifest.to_dict()) == before
        assert closed.orders_closed == 1
        assert _every_order_closed(root / segment.name / "orders.csv")
        meta = json.loads((root / segment.name / "meta.json").read_text())
        assert meta["capture_error"] == "it did not stop"


class TestRestart:
    def test_continues_the_capture_and_closes_what_the_crash_left_open(self, tmp_path):
        root = tmp_path / "cap"
        _crashed_capture(root)
        run = _capture(tmp_path, [OK], seconds=0.5)

        assert run.error is None
        manifest = run.manifest
        assert manifest.restarts == 1
        first, second = manifest.segments[:2]
        assert first.end_reason is EndReason.UNFINISHED
        assert second.name == "seg-0002" and second.stream_started is not None

        # The cut line is gone and order 1 gets its closing row.
        rows = _rows(root / "seg-0001" / "orders.csv")
        assert [r["id"] for r in rows] == ["1", "2", "2", "1"]
        assert rows[-1]["action"] == "deleted"
        assert rows[-1]["origin"] == "shutdown"
        assert _every_order_closed(root / "seg-0001" / "orders.csv")

        trades = (root / "seg-0001" / "trades.csv").read_text()
        assert trades.startswith("trade_id,timestamp,")

        meta = json.loads((root / "seg-0001" / "meta.json").read_text())
        assert meta["unfinished"] is True
        assert "provisional" not in meta
        assert meta["synthetic_deleted"] == 1

        # The downtime -- from the last heartbeat to the new stream -- is a gap.
        (gap,) = manifest.gaps
        assert (gap.after, gap.before, gap.cause) == ("seg-0001", "seg-0002", "restart")
        assert gap.start == pd.Timestamp("2026-09-26 10:00:02", tz="UTC")

        failed = {c.name for c in manifest.checks() if not c.passed}
        assert failed == {"capture_gaps", "unfinished_segments"}

    def test_a_restart_that_finds_a_clean_capture_just_adds_segments(self, tmp_path):
        first = _capture(tmp_path, [OK], seconds=0.4)
        second = _capture(tmp_path, [OK], seconds=0.4)
        assert second.manifest.restarts == 1
        assert [s.name for s in second.manifest.segments] == ["seg-0001", "seg-0002"]
        assert first.manifest.segments[0].end_reason is EndReason.FINISHED
        # Planned or not, the time between the two runs has no data.
        (gap,) = second.manifest.gaps
        assert (gap.after, gap.before, gap.cause) == (
            "seg-0001",
            "seg-0002",
            "finished",
        )

    def test_refuses_a_directory_holding_another_capture(self, tmp_path):
        _capture(tmp_path, [OK], seconds=0.3)
        cfg = CaptureConfig(pair="ethusd", out_dir=tmp_path / "cap", minutes=0.01)
        with pytest.raises(ConfigError, match="btcusd"):
            asyncio.run(run_capture(_factory([OK]), cfg))

    def test_refuses_a_directory_another_capture_is_writing_to(self, tmp_path):
        root = tmp_path / "cap"
        root.mkdir()
        held = _supervisor._lock_capture_dir(root)  # stands in for a live process
        try:
            with pytest.raises(ConfigError, match="another capture process"):
                _capture(tmp_path, [OK], seconds=0.3)
            assert not (root / "manifest.json").exists()
        finally:
            held.close()
        # Once the other process has gone, the directory is free again.
        assert _capture(tmp_path, [OK], seconds=0.3).error is None

    def test_refuses_a_directory_holding_other_files(self, tmp_path):
        (tmp_path / "cap").mkdir()
        (tmp_path / "cap" / "orders.csv").write_text("id\n")
        cfg = CaptureConfig(pair="btcusd", out_dir=tmp_path / "cap", minutes=0.01)
        with pytest.raises(ConfigError, match="no manifest.json"):
            asyncio.run(run_capture(_factory([OK]), cfg))


# ---------------------------------------------------------------------------
# Reading a segmented capture back
# ---------------------------------------------------------------------------


class TestReadBack:
    @pytest.fixture
    def gappy(self, tmp_path) -> Path:
        """A two-segment capture with one disconnect gap between them."""
        return _capture(tmp_path, [_disconnect_after(20), OK], seconds=0.8).out_dir

    def test_audit_reports_each_segment_and_the_capture(self, cli_runner, gappy):
        r = cli_runner("audit", str(gappy))
        assert r.returncode == 0, r.stderr
        assert "== seg-0001 ==" in r.stdout
        assert "== seg-0002 ==" in r.stdout
        assert "Capture summary" in r.stdout
        assert "capture_gaps" in r.stdout

    def test_audit_strict_fails_on_a_gap(self, cli_runner, gappy):
        r = cli_runner("audit", str(gappy), "--strict")
        assert r.returncode == 1
        assert "capture_gaps" in r.stderr

    def test_audit_json(self, cli_runner, gappy):
        r = cli_runner("audit", str(gappy), "--json")
        assert r.returncode == 0, r.stderr
        report = json.loads(r.stdout)
        assert set(report["segments"]) == {"seg-0001", "seg-0002"}
        assert len(report["capture"]["gaps"]) == 1
        checks = {c["name"]: c["passed"] for c in report["capture"]["checks"]}
        assert checks["capture_gaps"] is False

    def test_a_capture_with_no_data_fails_audit_and_process(self, cli_runner, tmp_path):
        """A capture whose first snapshot failed has nothing that could pass."""
        empty = _capture(tmp_path, [FAIL_SNAPSHOT], seconds=5.0).out_dir
        for command in (["audit", str(empty)], ["audit", str(empty), "--json"]):
            r = cli_runner(*command)
            assert r.returncode == 1
            assert "No segment" in r.stderr
        r = cli_runner("process", str(empty), "--output", str(tmp_path / "out"))
        assert r.returncode == 1
        assert "No segment" in r.stderr

    def test_a_segment_still_being_captured_is_named_and_left_out(
        self, cli_runner, gappy, tmp_path
    ):
        """Auditing a running capture reads its closed segments only."""
        manifest = read_manifest(gappy)
        assert manifest is not None
        # seg-0003 is open: rows on disk, no counts or end in the manifest.
        (gappy / "seg-0003").mkdir()
        for name in ("orders.csv", "trades.csv"):
            (gappy / "seg-0003" / name).write_bytes(
                (gappy / "seg-0002" / name).read_bytes()
            )
        manifest.segments.append(Segment(name="seg-0003", started=_now()))
        manifest.write(gappy)

        r = cli_runner("audit", str(gappy))
        assert r.returncode == 0, r.stderr
        assert "== seg-0002 ==" in r.stdout
        assert "== seg-0003 ==" not in r.stdout
        assert "seg-0003 was not checked: it is still being captured" in r.stderr

        out = tmp_path / "out"
        r = cli_runner("process", str(gappy), "--output", str(out))
        assert r.returncode == 0, r.stderr
        assert not (out / "seg-0003").exists()
        assert "seg-0003 was not processed" in r.stderr

    def test_a_capture_whose_only_data_is_still_being_captured(
        self, cli_runner, tmp_path
    ):
        root = tmp_path / "cap"
        _crashed_capture(root)  # one open segment, rows on disk
        r = cli_runner("audit", str(root))
        assert r.returncode == 1
        assert "No closed segment" in r.stderr
        assert "seg-0001 was not checked" in r.stderr

    def test_process_writes_each_segment_and_the_manifest(
        self, cli_runner, gappy, tmp_path
    ):
        out = tmp_path / "out"
        r = cli_runner("process", str(gappy), "--output", str(out))
        assert r.returncode == 0, r.stderr
        for name in ("seg-0001", "seg-0002"):
            assert (out / name / "events.parquet").is_file()
            assert (out / name / "meta.json").is_file()
        assert (out / "manifest.json").read_text() == (
            gappy / "manifest.json"
        ).read_text()

        r = cli_runner("audit", str(out), "--from-parquet")
        assert r.returncode == 0, r.stderr
        assert "== seg-0002 ==" in r.stdout
        assert "Capture summary" in r.stdout


class TestTrimToLastLine:
    """A crash can cut a line short; the restart cuts back to the last whole one."""

    def test_cuts_a_partial_line_longer_than_one_block(self, tmp_path):
        path = tmp_path / "raw.jsonl"
        path.write_bytes(b'{"a":1}\n' + b"x" * (3 * _supervisor._TRIM_BLOCK + 5))
        _supervisor._trim_to_last_line(path)
        assert path.read_bytes() == b'{"a":1}\n'

    def test_empties_a_file_with_no_whole_line(self, tmp_path):
        path = tmp_path / "raw.jsonl"
        path.write_bytes(b"x" * (_supervisor._TRIM_BLOCK + 1))
        _supervisor._trim_to_last_line(path)
        assert path.read_bytes() == b""

    def test_leaves_a_whole_file_alone(self, tmp_path):
        path = tmp_path / "raw.jsonl"
        path.write_bytes(b"one\ntwo\n")
        _supervisor._trim_to_last_line(path)
        assert path.read_bytes() == b"one\ntwo\n"
