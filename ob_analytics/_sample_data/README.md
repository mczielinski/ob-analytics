# Bundled sample data

A 30-minute Bitstamp BTC/USD capture (UTC 2026-05-02 02:36 → 03:06)
shipped alongside the package so `Pipeline()` runs out of the box.

| File | What it is |
|------|-----------|
| `orders.csv.gz` | Order events (gzip-compressed; `order_created` / `order_changed` / `order_deleted`), one row per WebSocket message. Includes a synthetic snapshot at `t=0` (REST `order_book/btcusd?group=2`) and synthetic deletes at `t=end` for everything still resting, so every order has a complete lifecycle. |
| `trades.csv` | Live trades (`live_trades_btcusd`), one row per match. Read by `BitstampTradeReader` to produce the canonical trades DataFrame. |
| `meta.json` | Capture metadata (start/end, channel list, snapshot microtimestamp, counters, reconnects). |

Headline numbers (see `meta.json` for the full set):

- `live_orders` 301 027, `synthetic_created` 6 512, `synthetic_deleted` 6 518
  → `total_order_rows` 314 057.
- `trades` 284, `dropped` 0, `reconnects` 0.

## Loading the sample

```python
from ob_analytics import Pipeline, sample_csv_path, sample_data_dir

# orders.csv.gz path; the pipeline auto-resolves the sibling trades.csv:
result = Pipeline().run(sample_csv_path())

# Or hand the directory directly to the reader:
from ob_analytics.bitstamp import BitstampLoader, BitstampTradeReader
events = BitstampLoader().load(sample_data_dir() / "orders.csv.gz")
trades = BitstampTradeReader().load(events, sample_data_dir())
```

## Regenerating the sample

The `scripts/collect_bitstamp_btcusd.py` collector produces a directory
matching this layout (it also emits a `raw.jsonl` frame log that is
intentionally **not bundled** — far larger than orders + trades combined,
no extra value for pipeline users).

```bash
./scripts/collect_bitstamp_btcusd.py --minutes 30 --out /tmp/sample-capture

RUN=$(ls -dt /tmp/sample-capture/bitstamp_btcusd_* | head -1)
gzip -9 -c "$RUN/orders.csv" > ob_analytics/_sample_data/orders.csv.gz
cp "$RUN/trades.csv" "$RUN/meta.json" ob_analytics/_sample_data/
```

Run the demo to confirm the bundled capture flows through end-to-end:

```bash
uv run python scripts/bitstamp_demo.py --output /tmp/bitstamp_demo
open /tmp/bitstamp_demo/gallery/gallery.html
```

## Packaging note

`orders.csv.gz` is gzip-compressed (~23 MB → ~2.9 MB), so it ships in the wheel
without bloating installs; pandas reads `.csv.gz` transparently. `trades.csv`
(~27 KB) and `meta.json` stay uncompressed. Both the sdist and the wheel bundle
all three.
