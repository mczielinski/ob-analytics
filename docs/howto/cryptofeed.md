---
title: Capture cryptofeed venues
---

# Capture native per-order (L3) crypto data (cryptofeed)

[cryptofeed](https://github.com/bmoscon/cryptofeed) streams normalised market
data over websockets and maintains the order book for you, applying each
venue's snapshot and deltas. It is the **per-order complement** to the
[CCXT source](ccxt.md): where CCXT covers the widest venue list at
price-level detail, cryptofeed can also deliver **market-by-order (L3)** on the
venues that publish it — the feed the reconstruction engine was built for.

Install the optional `[cryptofeed]` extra and use the `capture` verb:

```bash
pip install "ob-analytics[cryptofeed]"
```

```bash
ob-analytics capture cryptofeed --exchange bitstamp --pair BTC-USD --minutes 10 --out /tmp/cap
```

## The level comes from the venue, not from a list

A cryptofeed exchange declares the channels it supports. The source reads that
declaration: a venue offering a per-order book is captured at **L3**, and every
other venue at **L2**. Nothing is hardcoded, so the choice tracks cryptofeed's
own coverage as it changes.

In cryptofeed 2.4 and 2.5 the venues publishing a per-order book are
**bitstamp**, **bitfinex**, **blockchain**, and **independent_reserve**.
Coinbase and Bitso publish price-level data only.

| Level | Output | Replays through |
|-------|--------|-----------------|
| L3 | `orders.csv` (real venue order IDs) | the full reconstruction pipeline |
| L2 | `depth.csv` (absolute size per level) | the [L2 path](l2-depth.md) |

`--level` forces the choice. Forcing **L2** on an L3-capable venue is allowed —
it is a coarser view of the same book. Forcing **L3** on a venue that does not
publish one fails, because the only way to satisfy it would be to invent order
IDs that the feed never had.

```bash
# A per-order venue, captured at price level instead:
ob-analytics capture cryptofeed --exchange bitstamp --pair BTC-USD --level L2 --out /tmp/cap

# An L2-only venue:
ob-analytics capture cryptofeed --exchange binance --pair BTC-USDT --out /tmp/cap
```

Each run produces a self-contained directory:

| File | Contents |
|------|----------|
| `orders.csv` | L3 only: per-order `created` / `changed` / `deleted` events |
| `depth.csv` | L2 only: price-level updates (`volume` = new absolute size, `0` removes the level) |
| `trades.csv` | The trade tape (taker side; feeds trade-sign) |
| `raw.jsonl` | Raw frames as cryptofeed passed them on (omit with `--no-raw`). cryptofeed reads prices and sizes as `Decimal`, and these are written as strings so no digits are lost: a venue's `0.011` is stored as `"0.011"` |
| `meta.json` | Counts + per-run diagnostics (venue, level, book updates, sequence gaps, fills from the trade tape, errors), and what the source declares about its feed (`source`, `feed_type`, `trade_attribution`) |

## How the per-order events are derived

cryptofeed's L3 venues do not agree on what a book callback carries, so the
source handles both shapes:

- **A populated delta** — `(order_id, price, quantity)` triples, with a
  quantity of `0` meaning the order is gone — is mapped entry by entry. This
  is the venue's own account of what changed, so a venue that models a price
  move as a removal followed by an add (bitfinex) keeps that meaning: a move
  loses queue position, and recording it as one order quietly changing price
  would misstate the queue.
- **No delta, or an empty one** — bitstamp resends the whole book on every
  message, and bitfinex, blockchain and independent_reserve all open that way.
  The maintained book is diffed against the tracked orders to recover the same
  `created` / `changed` / `deleted` vocabulary.

### Bitstamp: a top-100 window, not every order

cryptofeed does not read Bitstamp's order channel (`live_orders`). Its Bitstamp
L3 book is the `detail_order_book` channel: about 10 times a second, a picture
of the **top 100 bids and the top 100 asks**. Three things follow.

- **An order can leave the picture without leaving the book.** When the book
  shows 100 orders on a side, an order missing past the last price shown, or at
  that price, may just have dropped below the 100th place. The source keeps it
  at its last known size and records nothing. When it comes back it is the same
  order. It is recorded `deleted` only once a picture shows its price again and
  it is not there, or at the end of the capture. So an order cancelled while out
  of view stays in the rebuilt book, below the 100th order, until then.
- **Takers never appear.** A taker trades the moment it arrives, so it is never
  resting when a picture is taken. `trades.csv` names both orders of every
  trade (Bitstamp sends them), but only the maker can be found in `orders.csv`.
  The source declares this as `trade_attribution = maker_only`, and `audit`
  counts only makers (see [Check data quality](audit.md)). No order can be
  labelled a market order.
- **Fills come from the trade tape.** Between two pictures, a fill and a cancel
  look the same: the order's size drops, or the order is gone. So when a trade
  names an order the source is tracking, the source records the fill straight
  away as a `changed` event, and an order filled completely is later recorded
  `deleted` at size 0, as the native Bitstamp feed reports a fill. The pictures
  and the trades come on separate channels, in either order:
  - A trade that a picture already shows (the picture is as late as the trade,
    on the venue's clock) is not applied twice; `meta.json` counts these as
    `tape_fills_already_shown`.
  - A picture that shows an order gone before its trade arrives holds the
    `deleted` for up to 2 seconds of venue time, so the late trade can still
    report its fill.

  An order first seen after it was partly filled cannot be linked to that
  trade. On a 90-second capture, 40 of 42 trades linked to their maker.

For analysis of Bitstamp orders, use the native [`bitstamp`](live-capture.md)
source: it reports every order, takers included, and every fill. The cryptofeed
capture is still useful as a check on the top of the native book. Each picture
states the top 100 in full, so an error there is corrected within a tenth of a
second, where a stream of changes keeps a lost message until the capture ends.

Order IDs are the venue's own throughout, exactly as published — integers on
bitstamp, bitfinex and blockchain, UUID strings on independent_reserve. The
shared schema keys orders by identity rather than by integer, so nothing is
re-labelled on the way through.

At shutdown every order still resting is closed out with a synthetic
`deleted`, so each ID in `orders.csv` has a complete lifecycle.

## Dropped messages and reconnects

cryptofeed owns reconnection — it re-establishes a dropped connection itself,
so a capture cannot count reconnects directly. What it can see is the
discontinuity a reconnect or a dropped message leaves in the venue's own
sequence numbers. Those are recorded per row in both `orders.csv` and
`depth.csv`, and each run reports `sequence_gaps` (how many breaks) and
`sequence_missing` (how many numbers went by unseen) in `meta.json`.

For the authoritative check, replay the capture and score it:

```python
from ob_analytics.analytics import detect_sequence_gaps
from ob_analytics.bitstamp import BitstampLoader
from ob_analytics.config import PipelineConfig

events = BitstampLoader(config=PipelineConfig(track_sequence=True)).load("orders.csv")
print(detect_sequence_gaps(events))
```

A venue that publishes no sequence number is never scored, and reports zero.

## Flags

| Flag | Meaning |
|------|---------|
| `--exchange` | cryptofeed venue id (`bitstamp`, `binance`, `coinbase`, …) |
| `--pair` | Symbol in cryptofeed notation (e.g. `BTC-USD`) |
| `--level` | Force `L2` or `L3`; omit to discover it from the venue |

## Which source for which venue

Prefer **one source per venue** rather than two ways to capture the same thing:

- **cryptofeed** when you want L3 on bitfinex, blockchain or
  independent_reserve, or when its websocket handling for a venue is the more
  robust path;
- **[native `bitstamp`](live-capture.md)** for Bitstamp L3: it shows every
  order and every fill, where cryptofeed's Bitstamp book shows the top 100
  orders a side and no takers (see
  [above](#bitstamp-a-top-100-window-not-every-order));
- **[CCXT](ccxt.md)** for breadth and for the prediction markets (Kalshi,
  Polymarket), which cryptofeed does not cover.

## See also

- [Capture CCXT venues](ccxt.md) — the price-level source covering the widest venue list
- [Capture live data](live-capture.md) — the capture framework and writing a bespoke venue
- [Process L2 (price-level) feeds](l2-depth.md) — what a captured `depth.csv` flows through
- [Check data quality](audit.md) — run `audit` on the captured output. A
  capture records which source made it, so `audit --source bitstamp` on a
  cryptofeed L3 capture checks it against what cryptofeed can show
