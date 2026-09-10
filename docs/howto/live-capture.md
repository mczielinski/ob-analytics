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
ob-analytics process /tmp/cap/orders.csv --gallery --output /tmp/cap_out
```

Each capture run produces a self-contained directory:

| File | Contents |
|------|----------|
| `orders.csv` | BitstampLoader-compatible event log (`created` / `changed` / `deleted`) |
| `trades.csv` | Venue-reported trades (informational; pipeline infers fills itself) |
| `raw.jsonl` | Every raw WebSocket frame (omit with `--no-raw`) |
| `meta.json` | Run metadata: start/end, counts, per-capturer diagnostics |

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
already includes are skipped (`pre_snapshot_skipped` in `meta.json`).

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
through was not really on the book. L2 captures report `null`, because a price
level has no id that the stream could confirm.

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
raw-frame archival, signal handling, and `meta.json` all live in the
generic runner -- you only write the per-venue parser. A source can also add
the offline-replay factories and be both.

## Related

- [Command-line interface](cli.md) — all `capture` flags
- [Extending ob-analytics](../extending.md) — the `Source` / `LiveSource` protocols in depth
