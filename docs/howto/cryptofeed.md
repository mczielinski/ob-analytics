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

!!! info "What this feed shows"

    | Property | Bitstamp L3 | Bitfinex L3 | Blockchain.com L3 | Independent Reserve L3 | L2 (Bitstamp, Kraken) |
    |---|---|---|---|---|---|
    | Level | L3 | L3 | L3 | L3 | L2 |
    | Depth shown | top 100 orders a side | top 100 orders a side | not checked (the book held 1 to 34 orders) | whole book | Bitstamp: whole book. Kraken: top 1,000 levels a side, and levels that leave it are kept |
    | Orders shown | resting only | resting only | resting only | resting only | — |
    | Update form | snapshots, about 10 a second | opening book, then every change | opening book, then every change | REST book newer than the stream, then every change | Bitstamp: REST book after 5 s, then changes. Kraken: changes |
    | What can be missed | orders that come and go between snapshots | nothing | nothing | any change or cancel after an order's first change, which cryptofeed ignores; the trades restore the fills | a lost message |
    | After a lost message | the next snapshot corrects it | reconnects; new opening book | reconnects; new opening book | reconnects; new opening book | drifts until the level changes again |
    | Sequence | none | counts every message on the connection, so book rows skip numbers; only rises | counts every message on the connection, so book rows skip numbers; only rises | skips the changes cryptofeed ignores; only rises | none |
    | Clocks | venue + receive | receive only | receive only | venue + receive; the opening book receive only | Bitstamp: venue + receive. Kraken: receive only |
    | Crossing | matched book; failed the check in testing (0.125% crossed): Bitstamp's own snapshots can briefly show a bid at a resting ask's price | matched book; failed the check in testing (58% crossed; cause not found) | matched book | matched book; failed the check in testing (58% crossed): orders that stayed in the book after they had gone | price levels |
    | Trade sides named | maker only, from the tape | neither | neither | maker only, from the tape | — |
    | Taker side | venue | venue | venue (not checked) | venue | venue |
    | Fills | from the tape | not linked | not linked | from the tape | — |
    | Trade tape gaps | a few trades from before the first snapshot | starts with the last 30 trades before the capture | not checked | none found | Bitstamp: a few trades from before the REST book |
    | Price grid | fixed (0.01 on BTC/USD) | five significant figures | fixed (0.01 on BTC/USD) | fixed (0.01 on BTC/AUD); some trade prices are off it | fixed (Bitstamp 0.01, Kraken 0.1 on BTC/USD) |
    | What the book means | normal | normal | normal | normal | normal |
    | Access | public | public | public | public | public; Coinbase needs an API key in cryptofeed 2.4.1; other venues not checked |

    Use it for an independent check of the top of the Bitstamp book, and for the
    per-order book on Independent Reserve. There, the book can keep orders that
    have gone, and the trades include other markets' trades, at prices in
    another currency ([see below](#independent-reserve-both-orders-of-a-trade)).
    Don't use it for Bitstamp order analysis (use the native
    [`bitstamp`](live-capture.md) source) or for Bitfinex order lifetimes. On
    Bitfinex, Blockchain.com and Independent Reserve the book rows skip sequence
    numbers without losing any, so `audit` checks only that the numbers never
    go back; a lost message shows as a `book_resyncs` warning instead (see
    [Dropped messages and reconnects](#dropped-messages-and-reconnects)).
    [What each feed shows](../feeds.md) explains each property and compares
    every source.

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

A capture is a directory of segments with a `manifest.json` (see
[Running for days](live-capture.md#running-for-days)). Each segment holds:

| File | Contents |
|------|----------|
| `orders.csv` | L3 only: per-order `created` / `changed` / `deleted` events |
| `depth.csv` | L2 only: price-level updates (`volume` = new absolute size, `0` removes the level) |
| `trades.csv` | The trade tape (taker side; feeds trade-sign) |
| `raw.jsonl` | Raw frames as cryptofeed passed them on (omit with `--no-raw`). cryptofeed reads prices and sizes as `Decimal`, and these are written as strings so no digits are lost: a venue's `0.011` is stored as `"0.011"`. For some venues (independent_reserve, blockchain) cryptofeed also reads date and time strings as dates and times; these are written back as ISO 8601 strings, such as `"2026-09-28T08:30:15.123456+00:00"`. These strings come from cryptofeed's parsed value, not the venue's text: `Z` becomes `+00:00`, and digits past the microsecond are lost. Any other value or dict key that JSON cannot hold is written as text, and a frame that JSON cannot hold at all is skipped; `meta.json` records both (see [Capture live order-book data](live-capture.md)) |
| `meta.json` | Counts + per-run diagnostics (venue, level, book updates, sequence gaps, fills from the trade tape, errors), what the source declares about its feed (`source`, `feed_type`, `trade_attribution`), and what `raw.jsonl` wrote as text or skipped (`raw_text_types`, `n_raw_frames_skipped`) |

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
L3 book is the `detail_order_book` channel: about 10 times a second, a snapshot
of the **top 100 bids and the top 100 asks**. Three things follow.

- **An order can leave the snapshot without leaving the book.** When the book
  shows 100 orders on a side, an order missing past the last price shown, or at
  that price, may just have dropped below the 100th place. The source keeps it
  at its last known size and records nothing. When it comes back it is the same
  order. It is recorded `deleted` only once a snapshot shows its price again and
  it is not there, or at the end of the capture. So an order cancelled while out
  of view stays in the rebuilt book, below the 100th order, until then.
- **Takers never appear.** A taker trades the moment it arrives, so it is never
  resting when a snapshot is taken. `trades.csv` names both orders of every
  trade (Bitstamp sends them), but only the maker can be found in `orders.csv`.
  The source declares this as `trade_attribution = maker_only`, and `audit`
  counts only makers (see [Check data quality](audit.md)). No order can be
  labelled a market order.
- **Fills come from the trade tape.** Between two snapshots, a fill and a cancel
  look the same: the order's size drops, or the order is gone. So when a trade
  names an order the source is tracking, the source records the fill straight
  away as a `changed` event, and an order filled completely is later recorded
  `deleted` at size 0, as the native Bitstamp feed reports a fill. The snapshots
  and the trades come on separate channels, in either order:
  - A trade that a snapshot already shows (the snapshot is as late as the trade,
    on the venue's clock) is not applied twice; `meta.json` counts these as
    `tape_fills_already_shown`.
  - A snapshot that shows an order gone before its trade arrives holds the
    `deleted` for up to 2 seconds of venue time, so the late trade can still
    report its fill.

  An order first seen after it was partly filled cannot be linked to that
  trade. On a 90-second capture, 40 of 42 trades linked to their maker.

For analysis of Bitstamp orders, use the native [`bitstamp`](live-capture.md)
source: it reports every order, takers included, and every fill. The cryptofeed
capture is still useful as a check on the top of the native book. Each snapshot
states the top 100 in full, so an error there is corrected within a tenth of a
second, where a stream of changes keeps a lost message until the capture ends.

### Independent Reserve: both orders of a trade

Independent Reserve's trade messages name the bid and the offer of each trade
(`BidGuid` and `OfferGuid`), and the source writes them to `trades.csv` as
`buy_order_id` and `sell_order_id`. Its book reports every change, a fill
included, so a fill reaches the capture twice: from the trade and from the
book. The source records each fill once. The one exception is a full fill that
the book reports before its trade: it reads as a cancel (see below). The two
messages are compared on the time Independent Reserve sent them, which is the
same for a trade and the book change it causes.

- **The maker is always named.** It was resting, so the book shows it.
- **A full fill that the book reports before its trade reads as a cancel.**
  The book's removal is written at once, at the order's last size. Unlike on
  Bitstamp, it is not held for the trade, because that would write it after
  rows from later messages. In every capture so far, each trade arrived before
  the book change it caused.
- **The taker is named only when the book shows it.** Some takers appear in
  the book as new limit orders, and most never do: in 40 minutes, 6 of 26
  takers appeared. So the source declares `trade_attribution = maker_only`. In
  a 20-minute capture, all 13 trades were linked to their maker.
- **Later fills are not lost.** cryptofeed forgets an order after its first
  change (see [What each feed shows](../feeds.md#timing-and-integrity)), so the
  book does not report a second fill. The trade does, and the source records
  it. A cancel after a change is still lost: the order stays in the book until
  the capture ends. Once a trade happens at a price past it, `audit` reports
  it as a [stale resting order](../data-quality.md#stale-resting-orders).
- **Some trades come from the venue's other markets.** A BTC-AUD capture
  also gets BTC-NZD and BTC-SGD trades, at prices in those currencies: 3 of 26
  trades in 40 minutes. They name orders in the capture, since the venue keeps
  one book for all currencies, but their prices are far from the AUD price, so
  `audit` reports orders that are still resting as stale. See [What each feed
  shows](../feeds.md#trades).

Order IDs are the venue's own throughout, exactly as published — integers on
bitstamp, bitfinex and blockchain, UUID strings on independent_reserve. The
shared schema keys orders by identity rather than by integer, so nothing is
re-labelled on the way through.

At shutdown every order still resting is closed out with a synthetic
`deleted`, so each ID in `orders.csv` has a complete lifecycle.

## Dropped messages and reconnects

cryptofeed owns reconnection. It checks the venue's sequence numbers on every
message, and on a lost one it reconnects and takes a new opening book. The
capture cannot see the reconnect, but it can see the new opening book after
changes to the old one. It brings its book into line with the new one and
counts it as `book_resyncs` in `meta.json`. The changes between the two
connections were missed. A capture of several segments adds the counts up in
`manifest.json`, and `audit` warns about them in the `book_resyncs` check.
Changes that come before the first opening book, such as on Bitstamp L2, are
not a resync. Bitstamp's L3 channel sends the whole book every time, so it
never counts one.

On Independent Reserve, cryptofeed takes the opening book from the venue's REST
interface when the first stream message arrives. The venue serves that book
from a cache, so it can be a few seconds older than the stream, and an order
cancelled in between would stay in the book until the capture ended. The
source fetches the book again, once a second, until it was created at least
3 s after the first message, at the start and after each reconnect. The stream
waits meanwhile and loses nothing, so the first rows arrive 3 to 6 s later than
on other venues. If no new enough book comes in 15 tries, the source uses the
last one and logs a warning.

The sequence numbers are recorded per row in both `orders.csv` and
`depth.csv`, but they skip although no message was lost: on Bitfinex and
Blockchain.com they count every message on the connection, trades and
heartbeats too, and on Independent Reserve cryptofeed passes on no change to an
order it does not hold. So the source declares them as only rising
(`sequence_kind` is `monotonic` in `meta.json`), and `audit` checks only that
they never go back. Each run also reports `sequence_out_of_order`, how many
times the number went back, in `meta.json`. On Bitfinex and Blockchain.com the
count starts again on a new connection. The capture counts that step back as a
`sequence_restarts`, not as out of order, and `audit` leaves it out of
`sequence_out_of_order`: the `book_resyncs` warning already reports the
reconnect.

For the authoritative check, replay the capture and score it:

```python
from ob_analytics import SequenceKind
from ob_analytics.analytics import detect_sequence_gaps
from ob_analytics.bitstamp import BitstampLoader
from ob_analytics.config import PipelineConfig

events = BitstampLoader(config=PipelineConfig(track_sequence=True)).load("orders.csv")
print(detect_sequence_gaps(events, kind=SequenceKind.MONOTONIC))
```

A venue that publishes no sequence number is never scored, and reports zero.

## Flags

| Flag | Meaning |
|------|---------|
| `--exchange` | cryptofeed venue id (`bitstamp`, `binance`, `coinbase`, …) |
| `--pair` | Symbol in cryptofeed notation (e.g. `BTC-USD`) |
| `--level` | Force `L2` or `L3`; omit to discover it from the venue |

## Which source for which venue

- **Bitstamp orders:** use the [native `bitstamp`](live-capture.md) source for
  analysis. It shows every order and every fill, where cryptofeed's Bitstamp
  book shows the top 100 orders a side and no takers (see
  [above](#bitstamp-a-top-100-window-not-every-order)). A cryptofeed capture of
  the same period is still worth making as an independent check of the top of
  the native book: each snapshot states the top 100 in full, so it does not keep
  an order the native stream lost.
- **bitfinex, blockchain or independent_reserve orders:** use cryptofeed. It is
  the only per-order source for these venues. Read [What each feed
  shows](../feeds.md) first: Bitfinex's book is a top-100 window too, and on all
  three the book rows skip sequence numbers without losing messages.
- **Price levels and prediction markets:** use [CCXT](ccxt.md). It covers the
  most venues, and Kalshi and Polymarket, which cryptofeed does not.

## See also

- [Capture CCXT venues](ccxt.md) — the price-level source covering the widest venue list
- [Capture live data](live-capture.md) — the capture framework and writing a bespoke venue
- [Process L2 (price-level) feeds](l2-depth.md) — what a captured `depth.csv` flows through
- [Check data quality](audit.md) — run `audit` on the captured output. A
  capture records which source made it, so `audit --source bitstamp` on a
  cryptofeed L3 capture checks it against what cryptofeed can show
