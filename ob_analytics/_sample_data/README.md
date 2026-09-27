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

## Known problems in this capture

The sample is kept as it was captured, faults included, because the
documentation uses it to show what `ob-analytics audit` finds. Three of them
change what you see:

- **Two stale orders keep the book crossed.** The capture fetched the opening
  REST book 0.74 s before its first stream event, and two asks deleted in that
  gap stayed in the rebuilt book: `2002347646152704` at 78,333 and
  `2002347642003458` at 78,374. The book that `order_book()` rebuilds is
  crossed for 91.6% of the session; with those two orders removed when a trade
  first proves them gone, it is crossed for 1.3%. `audit` names both. The
  depth summary already drops crossed levels, so they change it much less:
  the transaction costs and the count of hidden trades come out the same
  without them. Captures made now fetch the book again until it overlaps the
  stream, so they do not have this fault.
- **72 of the 284 trades have a negative effective spread.** This is not the
  stale orders. Bitstamp sends order messages and trade prints separately, and
  in 40 of those trades the taker's order reaches the order stream before the
  print does. See the transaction-costs how-to.
- **30 minutes is short for VPIN.** The default VPIN bucket is a fiftieth of a
  day's volume, which is more than this whole capture traded, so
  `compute_vpin` fills one bucket and warns. Pass a smaller `bucket_volume`.

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
