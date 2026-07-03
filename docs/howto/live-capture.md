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

Implement the `LiveCapturer` protocol -- three async-iterator methods --
and register your class:

```python
from ob_analytics.live import LiveCapturer, register_capturer

class CoinbaseCapturer(LiveCapturer):
    name = "coinbase"

    async def snapshot(self, config):
        # yield synthetic "created" events from a REST snapshot
        ...

    async def stream(self, config):
        # yield (kind, event, raw_frame) tuples for each live message
        ...

    async def shutdown_synthetic_events(self):
        # yield "deleted" events for everything still resting
        ...

register_capturer("coinbase", CoinbaseCapturer)
```

That's enough to make `ob-analytics capture coinbase` work. Persistence,
raw-frame archival, signal handling, and `meta.json` all live in the
generic runner -- you only write the per-venue parser.

## Related

- [Command-line interface](cli.md) — all `capture` flags
- [Extending ob-analytics](../extending.md) — the `LiveCapturer` protocol in depth
