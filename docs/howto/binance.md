---
title: Capture Binance spot
---

# Capture Binance spot

ob-analytics captures Binance spot through the [ccxt source](ccxt.md) and
replays it through the [L2 path](l2-depth.md). You do not need a Binance
account or an API key: the order book and the trades are public.

```bash
pip install "ob-analytics[ccxt]"

ob-analytics capture ccxt --exchange binance --pair BTC/USDT --minutes 10 --out /tmp/binance
ob-analytics process /tmp/binance --source depth_csv --gallery --output /tmp/binance_out
```

`--pair` takes the ccxt symbol, `BTC/USDT`. Binance calls the same market
`BTCUSDT`, and ccxt converts between the two.

## If Binance refuses your location

Binance does not serve some countries, the United States among them. From
those, the capture stops with this error:

```text
binance refused this location (HTTP 451). For Binance US, use --exchange binanceus. ...
```

You can do one of two things:

- **Capture Binance US** with `--exchange binanceus`. It is a separate
  exchange with its own book, and much less trading: 21 BTC/USDT trades in
  five minutes when we tested it, against about 7,000 on Binance.
- **Read from Binance's market-data mirror** with `--market-data-mirror`.
  Binance publishes public market data at `data-api.binance.vision` and
  `data-stream.binance.vision`. The mirror serves spot markets only. Before
  you use it, check that Binance's terms let you do so from where you are.

```bash
ob-analytics capture ccxt --exchange binance --market-data-mirror --pair BTC/USDT --minutes 10 --out /tmp/binance
```

From Python, set `CcxtSettings(exchange="binance", market_data_mirror=True)`.

## What the capture records

The book is price-level (L2): Binance publishes a total size at each price
and no order IDs. So queue position and order types do not apply.

The capture checks the book against Binance's REST snapshot. Each snapshot
carries an update ID, and each captured book update carries the same ID in
the `sequence` column of `depth.csv`. In a five-minute test on BTC/USDT, 251
REST snapshots had a matching update in the capture, and all 100 levels a side
matched exactly in every one.

| Field | Where it comes from |
|-------|--------------------|
| Tick size | Binance's market list: 0.01 for BTC/USDT. The capture writes it to `meta.json` as `tick_size`, and `process` and `audit` read it from there |
| Trade side | Binance's `m` flag (the buyer was the maker). `m` true means the taker sold, so ccxt writes `sell`. All 7,203 trades in the test agreed with `m` |
| Trade time | `exchange_timestamp` is Binance's time. `timestamp` is the time the capture received the trade |
| Book time | The same two clocks. Replay sorts on `timestamp`, the order the updates arrived |

Every trade carries its side, so ob-analytics does not have to guess sides
with [trade-sign classification](l2-depth.md#trade-signs).

## Depth: request more than 100 levels

By default the capture keeps 100 levels a side. The BTC/USDT book is dense,
with an order at almost every cent near the price. In the test, the levels
reached this far from the mid-price:

| Levels a side | Reach |
|---------------|-------|
| 100 (default) | about 0.02% (about $20) |
| 1,000 | about 0.2 to 0.3% |
| 5,000 (the most Binance sends) | about 1.0 to 1.4% |

Binance publishes no more than 5,000 levels a side, so no capture can see the
book past about 1% from the price. The depth summary adds up the size in
rings 25 basis points (0.25%) wide, out to 5%. With 100 levels, all the size
falls in the first ring.

Record more levels with `--depth-limit`, and make the rings narrower:

```bash
ob-analytics capture ccxt --exchange binance --pair BTC/USDT --depth-limit 5000 --out /tmp/binance
```

```python
from ob_analytics import DepthCsvSource, Pipeline, PipelineConfig

config = PipelineConfig(tick_size=0.01, depth_bps=1)
result = Pipeline(config, source=DepthCsvSource()).run("/tmp/binance")
```

ccxt starts its Binance book from a REST snapshot, and past that snapshot it
learns a level only when the level changes. So the capture makes the snapshot
as deep as `--depth-limit`, and never less than 1,000 levels. A
`--depth-limit` above 5,000 is refused, because the bottom of the window would
have gaps. (Binance's futures venues, `binanceusdm` and `binancecoinm`, send at
most 1,000.)

A level that leaves the recorded window, because the price moved away from it,
is written with size `0`, the same as a cancelled level. With 5,000 levels the
window reaches about 1% from the price, so this happens only after a large
move.

!!! note "Why ccxt gets the whole book"
    If ccxt is given a depth, it deletes the Binance levels past it. Binance
    sends a level again only when it changes. So when the price moves away and
    back, a level that left the window does not come back, and the book has
    gaps near the top. The capture asks ccxt for the whole Binance book, keeps
    the top `--depth-limit` levels itself, and records a level again when it
    comes back into the window.

## Checking the capture

```bash
ob-analytics audit /tmp/binance --source depth_csv
```

The `sequence` of a Binance capture is Binance's update ID. One Binance
message covers a range of IDs, and ccxt can apply several messages before it
hands back a book, so the IDs in `depth.csv` skip many numbers. A skip is not
a lost message. The capture records `"sequence_kind": "monotonic"` in
`meta.json`, and `audit` checks only that the ID never goes back. The report
says `gaps not checked (sequence only rises)`.

ccxt itself watches for lost messages. If a Binance message does not follow
on from the one before, ccxt drops its book and reports an error. The capture
then asks for the book again, and ccxt starts from a new snapshot. Levels that
changed during the gap are recorded with their new size. `meta.json` counts
these restarts as `book_resyncs`; after 10 in one capture, the book stops and
`errors` counts 1.

## See also

- [Capture CCXT venues](ccxt.md): the source this capture uses
- [Process L2 (price-level) feeds](l2-depth.md): what the captured `depth.csv`
  goes through
- [Check data quality](audit.md): run `audit` on the capture
