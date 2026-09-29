---
title: Capture live data
---

# Capture live order-book data

ob-analytics ships a small framework for capturing live order-book data
straight into the format the pipeline reads. Install the optional
``[live]`` extra (pulls in ``websockets``) and use the ``capture`` CLI verb:

```bash
pip install "ob-analytics[live]"

ob-analytics capture bitstamp --pair btcusd --minutes 10 --out /tmp/cap
ob-analytics process /tmp/cap --gallery --output /tmp/cap_out
```

A capture is a directory of **segments**. A short run has one; a long run can
have many (see [Running for days](#running-for-days)). Each segment is a
complete capture on its own, so it replays alone:

```text
/tmp/cap/
  manifest.json
  seg-0001/
    orders.csv  trades.csv  raw.jsonl  meta.json
  seg-0002/
    ...
```

| File | Contents |
|------|----------|
| `manifest.json` | The whole capture: its segments, why each one ended, and the gaps between them |
| `seg-NNNN/orders.csv` | BitstampLoader-compatible event log (`created` / `changed` / `deleted`) |
| `seg-NNNN/trades.csv` | Venue-reported trades (informational; pipeline infers fills itself) |
| `seg-NNNN/raw.jsonl` | Every raw WebSocket frame (omit with `--no-raw`) |
| `seg-NNNN/meta.json` | Segment metadata: start/end, counts, per-capturer diagnostics |

`process` and `audit` given the capture directory work through each segment.
`process` writes each segment's results to the folder of the same name under
`--output` and copies `manifest.json` there. To read one segment in Python,
pass its book file: `Pipeline().run("/tmp/cap/seg-0001/orders.csv")`.

The Bitstamp capturer also pulls a REST order-book snapshot at startup
(emitting synthetic `created` events for every resting order) and emits
synthetic `deleted` events at shutdown so every order id in `orders.csv`
has a complete `created -> ... -> deleted` lifecycle.

## Which rows came from the snapshot

Every row of `orders.csv` and `depth.csv` has an `origin` column that says
which part of the capture wrote it:

| `origin` | Written by |
|----------|------------|
| `snapshot` | The opening book, before the stream starts |
| `stream` | A live message from the venue |
| `shutdown` | A synthetic close-out at the end of the run |

The runner fills this in, so it means the same thing for every venue. A source
that delivers its opening book as its first live message, as the cryptofeed
source does, has no `snapshot` rows. Captures made before this column existed
do not have it.

The pipeline keeps the column, so `result.events["origin"]` tells you whether
an order was first seen in the opening book or placed during the run.

## When the snapshot and the stream line up

The Bitstamp capturer subscribes to the stream first, then fetches the REST
book while it holds the live messages in a buffer. That only works if the
stream already covers the moment the REST book describes. Then every change
after the snapshot arrives on the stream, and buffered messages the snapshot
already includes are skipped (`pre_snapshot_skipped` in `meta.json`). Trades
from before the snapshot are skipped too (`pre_snapshot_trades_skipped`). The
orders they filled are not in the capture, so `audit` would count them as
unmatched. At a roll, the previous segment already has these trades.

Bitstamp's REST book can be older than that. In a live test the first order
message on the stream came 0.7 seconds after the snapshot's `microtimestamp`.
An order deleted in that gap is in the snapshot, but the stream never reports
its delete. The bundled sample has the same gap, and two of its asks are
stale for this reason. A second fetch 2 seconds later no longer listed any of
them.

So the capturer checks for overlap: at least one buffered order message must
be at or before the snapshot's `microtimestamp`. If none is, it waits a second
and fetches the book again, up to 10 times. `meta.json` records how many
fetches it took (`snapshot_fetches`) and whether the snapshot it used overlaps
the stream (`snapshot_overlap`). If `snapshot_overlap` is `false`, orders
removed just before the stream started may stay on the book until shutdown.

## Opening orders the stream never confirmed

The opening book is not checked against the stream after the capture starts.
An order that the snapshot lists but that had already gone rests in the
capture until the synthetic `deleted` at shutdown. It then looks like a normal
order with a complete lifecycle.

`meta.json` reports `n_snapshot_unconfirmed`: how many orders in the opening
book were never named again by an order event or a trade. Most of these are
far from the touch and did not trade during the run, so a large number is
normal. Compare it with the size of the opening book, then look at the
unconfirmed orders near the touch. An unconfirmed order that trades print
through was not really on the book; [`ob-analytics audit`](audit.md) reports
these as `stale_orders`. L2 captures report `null`, because a price
level has no id that the stream could confirm.

## Running for days

A capture can run unattended. Every break in it is handled the same way: the
current segment is closed, with its closing rows, and a new segment starts from
a fresh snapshot. `manifest.json` records each break.

**Rolling to a new segment.** `--roll-minutes 60` starts a new segment every
hour, and `--roll-mb 500` starts one when a segment has written 500 MB since
its first live event (the opening snapshot does not count). Both must be above
0. The
new segment starts before the old one stops, and the old one stops only when
the new one is streaming. The two segments overlap, so a roll loses nothing.

**A lost connection.** The segment ends with the error in its `meta.json`. The
capture waits 1 second, then starts a new segment. The wait doubles after each
failure in a row, up to 60 seconds, and goes back to 1 second after a segment
that ran for a minute. The time between the two segments is recorded as a gap.
Book changes made during a gap cannot be recovered: no venue replays them. The
new snapshot means the book is correct again from the start of the next
segment.

**A restart.** Run the same command again with the same `--out` and the
capture continues in the next segment. If the old process died without
closing its last segment, the restart closes it: it cuts off a half-written
last line, writes a `deleted` row for each order still open at the last
recorded time, and completes `meta.json` with `"unfinished": true`. The time
the capture was down is recorded as a gap. The capture refuses an `--out`
that holds a different venue or pair, files but no `manifest.json`, or a
`manifest.json` this version cannot read. While it runs, the capture holds a
lock on `--out` (the `.capture.lock` file), so a second capture into the same
directory stops with an error instead of rewriting the first one's files.

**A segment that does not stop.** A segment asked to stop, at a roll, at the
end or on a signal, has 20 seconds to close its connection and write its
closing rows. If it takes longer, it is cancelled and closed from its files,
the same way a restart closes a segment a crash left open. The manifest keeps
why it was stopped (`rolled_time`, `finished`, ...) and records the delay as
its error. The delay is not a gap: the segment had already stopped streaming.
If closing its files fails, the error says so and the capture carries on.
So a connection that hangs cannot stop the rolls, and SIGTERM ends a capture
in under a minute: at most two of these limits, if it arrives while a roll is
waiting for a segment that hangs.

This lets a service manager restart the capture. For example, a systemd unit
with `Restart=always` and
`ExecStart=ob-analytics capture bitstamp --pair btcusd --minutes 10080 --roll-minutes 60 --out /data/btcusd`
captures for a week (counted from each start) and continues after a crash or
reboot.

`manifest.json` records:

| Field | Meaning |
|-------|---------|
| `source`, `pair`, `level` | What was captured. A restart must match them |
| `feed_type`, `trade_attribution`, `sequence_kind` | What the source declares about its feed |
| `version` | The layout version (currently `1`) |
| `started`, `ended` | When the capture started, and when it last stopped |
| `restarts` | How many times the capture was started again in this directory |
| `segments` | Each segment: when it streamed from and to (until it was asked to stop; none for a segment asked to stop before its first live event), why it ended (`rolled_time`, `rolled_size`, `failed`, `ended_early`, `unfinished`, `stopped`, `finished`), its error, row counts, and the messages its source dropped |
| `gaps` | Each stretch with no segment streaming: start, end, length, and cause (`disconnect`, `restart`, `roll`, `stopped`, `finished`) |
| `dropped`, `gap_seconds` | Totals over the whole capture |

The manifest is rewritten after every change and every 10 seconds while the
capture runs, so it is never more than 10 seconds out of date.

`ob-analytics audit /tmp/cap` audits each segment, then prints the capture's
own checks: gaps, segments a dead process left open, and dropped messages.
These are warnings, so `audit --strict` fails a capture that has any.
You can audit or process a capture while it runs. The segment still being
captured is left out, and the log names it: it has no closing rows yet, so
every order still resting would read as a fault. Only closed segments are
read.

## A capture that fails

`ob-analytics capture` checks the source before it starts. If a source needs
an optional extra that is not installed, or the venue name is unknown, it
prints the reason, exits with status 1, and creates no output directory.

If the first segment of a new capture fails before its first event, the
settings are probably wrong (a pair the venue does not list, a location it
refuses), so the capture stops and exits with status 1 instead of retrying. It also exits with status 1
if no segment streamed at all. A failure after that ends the segment and the
capture carries on, as described above.

A segment keeps the rows it wrote before an error. Its `meta.json` records the
error as `capture_error`, the step that failed as `capture_error_phase`
(`snapshot`, `stream` or `shutdown`), and counts it in `errors`. After a failed
snapshot the stream does not run.

## Adding a new venue

Give your source the live capability -- the three async-iterator methods of a
`LiveSource` -- alongside its `level` / `feed_type` / `settings`, and register
it:

```python
from ob_analytics import FeedType, Level, SourceSettings, register_source


class CoinbaseSource:
    name = "coinbase"
    level = Level.L3
    feed_type = FeedType.MATCHED_BOOK
    settings = SourceSettings()

    async def snapshot(self, config):
        # yield synthetic "created" events from a REST snapshot
        ...

    async def stream(self, config):
        # yield (kind, event, raw_frame) tuples for each live message
        ...

    async def shutdown_synthetic_events(self):
        # yield "deleted" events for everything still resting
        ...


register_source("coinbase", CoinbaseSource)
```

That's enough to make `ob-analytics capture coinbase` work. Persistence,
raw-frame archival, signal handling, segments, and `meta.json` all live in the
generic runner -- you only write the per-venue parser. A source can also add
the offline-replay factories and be both.

To wait for the next message with a time limit, use
`async with asyncio.timeout(...)`, not `asyncio.wait_for`. On Python 3.11,
`wait_for` can drop the cancel that stops a segment when a message arrives at
the same moment, and the stream then carries on. Keep the cleanup in `stream`'s
`finally` short: a segment that takes more than 20 seconds to stop is
cancelled.

Do not reconnect inside `stream`. When the connection drops, let `stream`
raise: the capture then starts a new segment from a fresh snapshot and records
the gap. A source that reconnects by itself carries its book across the gap,
and every order changed while it was disconnected stays wrong.

## Related

- [Command-line interface](cli.md) — all `capture` flags
- [Extending ob-analytics](../extending.md) — the `Source` / `LiveSource` protocols in depth
