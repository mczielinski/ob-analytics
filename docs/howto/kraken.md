---
title: Capture Kraken order by order
---

# Capture Kraken order by order

The `kraken` source reads Kraken's `level3` WebSocket channel. It records each
order in the book, with Kraken's order id, and every change to it. It needs a
Kraken account and an API key. A normal account can make a key at no cost.

!!! info "What this feed shows"

    | Property | Value |
    |---|---|
    | Level | L3 |
    | Depth shown | top 1,000 price levels a side (`--depth` sets 10, 100 or 1000); an order that leaves them is kept |
    | Orders shown | resting only |
    | Update form | opening book, then every change |
    | What can be missed | when an order left the book while it was outside the window |
    | After a lost message | the checksum fails; a new segment from a new opening book |
    | Sequence | none; a checksum of the top 10 levels on every message |
    | Clocks | venue + receive |
    | Crossing | matched book |
    | Trade sides named | neither (Kraken's trades carry no order ids) |
    | Taker side | venue |
    | Fills | not linked |
    | Trade tape gaps | trades from before the opening book dropped |
    | Price grid | fixed (0.1 on BTC/USD) |
    | What the book means | normal |
    | Access | API key, free with a Kraken account |

    Use it for the per-order book of a large crypto market: order lifetimes,
    queue position and cancel rates. Don't use it to find the order behind a
    trade: Kraken's trades name no orders. [What each feed shows](../feeds.md)
    explains each property and compares every source.

The values were checked on 19 November 2026 with a key from a normal Kraken
account, in captures of BTC/USD of up to 3 minutes. [What the live captures
showed](#what-the-live-captures-showed) gives the details.

## Get a key

1. Log in to Kraken and open **Settings**, then **API**
   (<https://pro.kraken.com/app/settings/api>).
2. Create a key and turn on **WebSocket interface**. Kraken counts that as a
   setting, not a permission, and asks for at least one permission too: tick
   one that only reads, such as **Query Open Orders & Trades**. The capture
   uses neither; it never trades or moves funds. Leave every permission that
   creates, changes or cancels orders, or deposits or withdraws funds, off.
3. Kraken shows the key and its private key (the "secret") once. Save each in
   a file that only you can read (`chmod 600`).

## Capture

Set the key and its private key, then capture. [Use an API key](api-keys.md)
explains how keys are kept out of the shell history and out of the capture's
files.

```bash
pip install "ob-analytics[live]"

export KRAKEN_API_KEY="$(cat ~/.config/kraken/api-key)"
export KRAKEN_API_SECRET="$(cat ~/.config/kraken/api-secret)"

ob-analytics capture kraken --pair BTC/USD --minutes 10 -o /tmp/kraken
ob-analytics audit /tmp/kraken
ob-analytics process /tmp/kraken --gallery --output /tmp/kraken_out
```

`--pair` takes Kraken's WebSocket symbol, such as `BTC/USD` or `ETH/EUR`.

The capture writes `orders.csv`, `trades.csv`, `raw.jsonl` and `meta.json`,
the same files as a [Bitstamp capture](live-capture.md), and replays through
the same pipeline. The order ids are Kraken's, such as `OTCFZG-YOE2Q-LQKNM3`.

| Kraken sends | `orders.csv` holds |
|---|---|
| an order in the opening book | `created` |
| `add` | `created` |
| `modify` (a fill, or a change to the size) | `changed`, with the size left |
| `delete` (a full fill or a cancel) | `deleted` |
| `delete`, then `add` of the same id in the next message (an amend) | see [Amends](#amends) |

Each row carries Kraken's time in `exchange_timestamp`: the time Kraken made
the message. `timestamp` is the time the capture received it.

The capture gets a token from Kraken's REST API to open the `level3` channel.
The token must be used within 15 minutes, so each segment gets its own. The
token never reaches the capture's files.

## The checksum

Kraken numbers nothing on this channel, so a lost message cannot be seen from a
missing number. Instead, each message carries a checksum: a CRC32 of the top 10
price levels a side, order by order. The capture works out the same checksum
from its own book after every message and compares the two.

If they differ, the capture's book no longer matches Kraken's. By default the
capture ends the segment and starts a new one from a new opening book, and
`manifest.json` records the gap. `--on-checksum-failure count` keeps the
capture going instead: it counts each failure in `meta.json` and warns once.
A clean run has `checksum_failures: 0`.

## Amends

Kraken lets a trader change an order's price without a new order id. The
`level3` channel shows this as a `delete` and then, in the next message, an
`add` with the same id at the new price. In a 3-minute BTC/USD capture, 15,183
of 61,066 adds were amends like this.

So the capture holds each `delete` for one message before it writes it. If the
next message adds the same id, the capture writes:

- one `changed` row at the new price, when the size stayed the same or grew.
  The pipeline reads this as a move: the order leaves its old level for the
  back of the new one.
- a `deleted` row, then a `created` row under a new id (the old id with `#1`
  added), when the size went down. A replay would read a smaller size as a
  fill at the old price and leave the rest of the order there, so this amend
  is written as a cancel and a new order instead. `meta.json` counts these as
  `amends_split`: 2,534 in the same capture.

If the next message does not add the id, the `delete` is written as it came,
with its own time.

## The queue within a price

The checksum covers the orders at each price in queue order, so it shows how
Kraken orders its queue. An order whose size goes down (a fill) keeps its
place. An order whose size goes up, or whose price changes, goes to the back
of its level. The capture keeps its book the same way.

## The depth window

Kraken sends the top `--depth` price levels a side: 10, 100 or 1000. The
default is 1000, the most Kraken sends. When the price moves, a level can leave
this window while its orders still rest. Kraken sends no `delete` for them.

The capture does not write a level that leaves the window as a cancel. It keeps
its orders open and writes nothing. When the window reaches that price again,
Kraken sends every order resting there as an `add`:

- An order that comes back unchanged gets no row.
- An order that comes back with another size gets a `changed` row.
- An order that does not come back left the book while it was outside the
  window. It gets a `deleted` row then, at the time of that message, which is
  later than the order really left.

So inside the window, the capture's book matches Kraken's after every message.
Outside it, the capture's book holds orders as they were when they left, and
some may be gone. `meta.json` counts `window_exits`, `window_returns` and
`window_unseen_deletes`.

## What `meta.json` records

As well as the fields every capture writes, a Kraken capture records:

| Field | What it counts |
|---|---|
| `depth` | the price levels a side Kraken was asked for |
| `snapshot_orders` | orders in the opening book |
| `adds`, `modifies`, `deletes` | Kraken's order events |
| `checksum_checks`, `checksum_failures` | messages checked, and how many failed |
| `window_exits` | orders that left the window and were kept open |
| `window_returns` | orders that came back into the window |
| `window_unseen_deletes` | orders that left the book while outside the window |
| `amends`, `amends_split` | amends written as one `changed` row or as a cancel and a new order (see [Amends](#amends)) |
| `readded_after_unseen_delete` | orders written as gone that Kraken sent again; each gets a new id, the old one with `#1` added |
| `readded_after_delete` | orders Kraken deleted and added again later than the next message; each gets a new id the same way |
| `parked_at_end` | orders outside the window when the capture ended |
| `messages_without_time` | book messages that carried no time |
| `trade_events`, `pre_snapshot_trades_skipped`, `dropped` | trades written, left out as too early, and left out as unreadable |

## From Python

Build the settings once and give them to each segment:

<!-- docs-test: skip needs a Kraken API key -->

```python
import asyncio
from pathlib import Path

from ob_analytics.live import CaptureConfig, KrakenSettings, KrakenSource, run_capture

settings = KrakenSettings()  # reads KRAKEN_API_KEY and KRAKEN_API_SECRET
config = CaptureConfig(pair="BTC/USD", out_dir=Path("/tmp/kraken"), minutes=10)
asyncio.run(run_capture(lambda: KrakenSource(settings=settings), config))
```

If a variable is not set, the capture stops before it creates the output
directory. If Kraken refuses the key, the capture stops at once and says why:
a wrong key, a wrong private key, or a key without the "WebSocket interface"
permission.

## What the live captures showed

Captures of BTC/USD on 19 November 2026, with a key from a normal Kraken
account that had "WebSocket interface" on and one read-only permission:

- **Access.** The key got the `level3` channel. Kraken stated no rule about
  account tier, and none applied.
- **Checksum.** In a 3-minute capture, all 55,802 checksums matched. The
  first captures failed after 20 to 30 seconds, because the capture kept an
  order whose size went up in its place in the queue; Kraken moves it to the
  back.
- **`modify` can raise a size.** It did so 7 times in 38,917 messages.
- **Amends** are a `delete` and an `add` of the same id, always in two
  messages one after the other (7,965 of 7,965 in one segment), and always at
  a new price. See [Amends](#amends).
- **The depth window** at 1000 levels moves a lot: 32,426 orders left it and
  32,309 came back in 3 minutes. 13 orders did not come back. 2 of those, and
  2 deleted orders, were added again later; each got a new id. The ones seen
  were orders that a trader moved to a new price while they were outside the
  window.
- **Time.** Every book message carried its time. A `modify` carries Kraken's
  time of the change, not only the order's insert time.
- **Trades** carry no order ids, so `audit` warns that 100% of trades have no
  maker order. This is expected for this feed.
- **`audit`** passed on every capture made with the finished source: no
  crossed book, no duplicate ids and no orphan orders.
- **Trades arrive late.** Kraken serves trades only on its public connection,
  not on the `level3` one, and they reach the capture 40 to 170 ms after
  Kraken made them. The book arrives sooner. `audit` orders trades and book
  rows by the time the capture received them, so it can see a trade "print
  through" an order that by Kraken's clock rested only after the trade: a
  taker order that sweeps the book and rests its remainder looks like this.
  `audit` then counts the order as stale. A 2-minute capture had 71 such
  orders, the worst at the touch for 7.1 s. The checksum shows the book was
  right at the time, so treat `stale_orders` on a Kraken capture as an upper
  bound.

## See also

- [Use an API key](api-keys.md)
- [Capture live data](live-capture.md)
- [What each feed shows](../feeds.md)
- [CCXT's Kraken capture](ccxt.md), price levels with no key
