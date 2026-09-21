---
title: Find hidden liquidity
---

# Find iceberg orders and trades against hidden orders

Some of the size a venue will trade does not show in the visible book. An L3
stream still records two footprints of it:

- **An iceberg order** shows a small displayed peak and keeps the rest in
  reserve. When the peak is filled, the venue shows the next slice as a new
  order at the same price, a moment later. `detect_icebergs` finds these
  refills and chains them into one suspected iceberg.
- **A hidden order** does not show at all. When a trade prints strictly
  inside the visible spread, no visible order rested at that price, so the
  maker was a hidden order. `hidden_trades` finds these trades.

Both read the tables of a finished L3 run.

## Find icebergs

```python
from ob_analytics import LobsterSource, Pipeline, RunContext, detect_icebergs

result = Pipeline(
    source=LobsterSource(),
    ctx=RunContext(trading_date="2012-06-21"),
).run("/path/to/AAPL_2012-06-21_10")

found = detect_icebergs(result.events, result.trades)
found.icebergs      # one row per suspected iceberg
found.slices        # one row per visible order in each iceberg
```

A slice is *filled out* when a trade fills it as the maker and leaves nothing
outstanding. A *refill* is the first new order at the same side and price, at
most `max_delay` later. Refills chain: a refilled slice that is filled out and
refilled again stays in the same iceberg.

Each row of `found.icebergs` gives the side and price, the start and end, the
number of `slices` and `refills`, the displayed `peak`, the size `executed`
across all slices, and the median delay from a filled-out slice to its refill.
The `confidence` column is `high` when there are at least two refills and every
one has the same displayed size as the slice it replaced, `medium` when at least
one does, and `low` otherwise.

```python
found.icebergs[found.icebergs["confidence"] == "high"]
```

### Choose `max_delay`

The default is one millisecond. On one day of AAPL on Nasdaq (LOBSTER,
2012-06-21), the delay from a filled-out peak to a new order at the same price
has a sharp peak between 0.2 and 0.3 ms. After a cancellation, a new order at
the same price follows within 0.4 ms only about 1% of the time. So one
millisecond keeps nearly all refills and lets few unrelated orders in.

Use a longer delay when the venue refills more slowly, or when the timestamps
are taken on receipt far from the venue:

```python
detect_icebergs(result.events, result.trades, max_delay="50ms")
```

A longer delay finds more refills, and more of them are orders that only
happened to arrive at the same price. Watch the share of `low` confidence rows
grow as you raise it.

### What it cannot see

- An iceberg whose first peak is cancelled before it fills. It never refills.
- The last part of the reserve when a single trade takes the peak and the rest
  of the reserve together. The venue has nothing left to show.
- The difference between a venue iceberg and a trader who re-sends the same
  order by hand within `max_delay`. Both look the same in the stream.

## Find trades against hidden orders

```python
from ob_analytics import hidden_trades

inside = hidden_trades(result.events, result.trades, result.depth_summary)
```

This returns the trades that printed strictly inside the visible spread, with
that spread's `best_bid_price` and `best_ask_price` added.

The spread is the one standing just before the maker's fill, not the one at
the trade's own timestamp. Some feeds report the fill on the order stream before
the trade print arrives. On the bundled Bitstamp sample the gap is about 20 ms,
and by the print the maker has already left the book. The book at the fill's
own instant is not used either. If it were, a trade that empties a price level
would read the level as already empty and look like a trade inside the
spread.

A hidden order that rests at the touch, or behind it, prints at a visible price.
`hidden_trades` does not flag it.

## How well it works

**On synthetic data.** The [synthetic generator](synthetic-data.md) labels every
slice of every iceberg it places, in `session.icebergs`:

```python
from ob_analytics.synth import generate_session

session = generate_session(seed=3, duration=600, iceberg_fraction=0.1)
found = detect_icebergs(session.events, session.trades)
```

Over five seeds with Poisson arrivals, the detector finds every refill, and
99.4–100% of the refills it reports are real. With Hawkes arrivals, which
cluster orders in time, it still finds every refill, and 98.5–99.3% are real.
The false refills are unrelated orders placed at the same price within the
delay, and they get `low` confidence because their size differs.

**On LOBSTER data.** LOBSTER does not label icebergs, but it does label every
execution against a hidden order: event type 5. On the AAPL day there are
11,332 of them:

| Where the type-5 execution printed | Executions | Share |
|---|---:|---:|
| Strictly inside the visible spread | 9,645 | 85.1% |
| At a visible peak filled out at the same price and instant | 1,399 | 12.3% |
| Anywhere else | 288 | 2.5% |

- `hidden_trades` flags all 9,645 in the first row and nothing else, so its
  precision is 100% and its recall 85%.
- The second row is an iceberg's reserve: the trade took the visible peak, then
  continued into the hidden size behind it. Of those 1,399, 408 (29%) belong to
  an iceberg that `detect_icebergs` found. For the rest, no new order at that
  price followed within one millisecond.
- A filled-out peak that was refilled has a type-5 execution at the same price
  and instant 24% of the time. A peak that was not refilled has one 9% of the
  time.

## Diff feeds need care

Both functions trust the book rebuilt from the events. That holds for a
matched book such as LOBSTER or Databento. A diff feed such as Bitstamp can
cross, and the depth summary then drops the resting levels a new quote crosses.
On the bundled Bitstamp sample, `hidden_trades` flags 40 of 284 trades. Every
one of them has a visible maker order at the trade price, so none is a trade
against a hidden order. In 39 of them the depth table still holds volume at the
maker's price, but the depth summary had dropped that level after a crossing
quote.

On a diff feed, run the [data-quality audit](audit.md) first, and read what
`hidden_trades` returns as trades to look at, not as hidden orders.

## Related

- [Generate synthetic L3 data](synthetic-data.md) — icebergs with known labels
- [Process LOBSTER files](lobster.md) — the event types, type 5 included
- [Check data quality](audit.md) — stale orders and crossed books on a diff feed
- [API: hidden liquidity](../api/hidden_liquidity.md)
