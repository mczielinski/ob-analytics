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
