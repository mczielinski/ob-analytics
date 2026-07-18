# Data quality: matched book vs diff feed

Before you trust a reconstructed order book, you need to know **what kind of
feed produced it**. The single most important property is the book's
*crossing invariant* — whether a resting buy order can ever sit at a higher
price than a resting sell order. It can, in some feeds, and that fact changes
how you read every downstream plot and metric.

This page explains the distinction, why a crossed book is often *faithful*
rather than a bug, and the two tools `ob-analytics` gives you to work with it:
the [`validate`](howto/validate.md) command that **measures** data quality,
and the `uncross=` option that **cleans the book up for display**.

## Two kinds of L3 feed

Every Level-3 (market-by-order) feed falls into one of two families:

| Feed family | Source | Crossing invariant | Examples |
|---|---|---|---|
| **Matched book** | The venue's own matching engine | Bids can **never** rest above asks — uncrossed is guaranteed by the data | LOBSTER, exchange MBO (Databento) |
| **Diff feed** | A public placement/cancellation stream, reassembled client-side | Can contain **genuinely crossed resting orders** | Bitstamp public feed |

A matched book is emitted *after* the engine has paired every marketable
order, so a crossing is impossible by construction. A diff feed is a running
log of "order added / changed / cancelled" messages with no authoritative
matched state; reassembling it can leave a bid resting above an ask because
the feed never told us the two met.

Each format **declares** its family, so downstream code reasons about crossing
*by coordinate, not by format name*:

```python
from ob_analytics import BitstampFormat, LobsterFormat, FeedType

BitstampFormat().feed_type   # FeedType.DIFF_FEED
LobsterFormat().feed_type    # FeedType.MATCHED_BOOK
FeedType.DIFF_FEED == "diff_feed"   # True — the enum mixes in str
```

A format that predates the attribute reads back as `FeedType.UNKNOWN`, so you
can always do `getattr(fmt, "feed_type", FeedType.UNKNOWN)` without special-casing.

## A crossed book is faithful, not a bug

We have **verified** a bid resting above an ask on the bundled Bitstamp
sample for about 1.5 minutes, neither order ever filling. That is a real
property of the public feed, not a defect in the reconstruction.

[`order_book()`](api/analytics.md) therefore replays a diff feed **as-is**: a
crossed book in the output is a property of the feed, not a reconstruction
error. The tutorial meets this pitfall twice — once in
[Chapter 2 · L1 → L2 → L3](tutorial/02_three_resolutions.md) and again in
[Chapter 3 · Loading order data](tutorial/03_loading_data.md) — and the
[glossary](glossary.md#data-formats) defines both terms.

!!! warning "Don't silently uncross data you're going to analyse"
    The crossing *is the signal* for a diff feed — it tells you the feed is
    unmatched and that spreads, mid-prices, and depth near the touch need care.
    Uncrossing (below) is a **display** convenience; never bake it into the
    data an analysis runs on, or you erase the very property `validate` is
    there to surface.

## Measuring it: `validate`

The [`ob-analytics validate`](howto/validate.md) command runs the pipeline and
prints a per-run data-quality summary. On the bundled Bitstamp sample:

```text
Data quality summary
  feed type             : diff_feed
  events / orders       : 314,057 / 156,902
  trades                : 284
  crossed resting book  : 92.02% of session (6348 episode(s)) [expected for a diff feed — faithful replay, not a bug]
  unmatched trades      : 0.70%
  duplicate event ids   : 0
  duplicate created ids : 0
  pre-existing orders   : 13
```

A matched book (LOBSTER) reports **~0%** crossed on the same metric — the
number is the cleanest single discriminator between the two families.

The four headline metrics:

| Metric | What it measures | Matched book | Diff feed |
|---|---|---|---|
| **crossed resting book %** | Share of session *time* the faithful book has `best_bid > best_ask` | ~0% | often high (92% here) |
| **unmatched trades %** | Trades with no resolvable maker/taker resting order | low | low–moderate |
| **duplicate ids** | `event_id`s seen twice, or order ids created twice | 0 | 0 (else a feed defect) |
| **pre-existing orders** | Orders already resting when the capture began (no `created` row) | a few | a few |

The crossing figure is measured from the **faithful** resting book (via
[`price_level_volume`](api/depth.md)), *not* from `depth_summary` — the depth
engine already evicts crossed levels, so it would always report ~0%. It is
duration-weighted, so "crossed for 90 seconds out of a 10-minute capture"
reads as ~15%, matching intuition.

## Cleaning it up for display: `uncross=`

When you want a clean `best_bid < best_ask` ladder from a diff feed — for a
figure, not for analysis — pass `uncross=True`:

```python
from ob_analytics.analytics import order_book

faithful = order_book(events)                 # default: crossed if the feed is
display  = order_book(events, uncross=True)   # best_bid < best_ask everywhere
```

The same flag threads through the visualization prepares that feed the
`book_snapshot` ladder and the `depth_chart` curve:

```python
from ob_analytics.visualization import prepare, plot

fig = plot("book_snapshot", level="L3",
           **prepare.book_snapshot(order_book=book, uncross=True))
```

Uncrossing mirrors the depth engine's crossed-level eviction: at the crossed
(or locked) touch it **trusts the fresher quote** and evicts the older, stale
opposing order, repeating until the sides no longer overlap. It never
fabricates liquidity — it only removes orders — and it recomputes each side's
cumulative depth against the surviving touch.

| Question | Use |
|---|---|
| "What did the feed actually contain?" | **`uncross=False`** (default) — faithful |
| "Draw me a clean ladder / depth curve for a slide" | `uncross=True` — display only |
| "How crossed is this feed?" | [`validate`](howto/validate.md) — never uncross first |

On a matched book `uncross=True` is a no-op (there is nothing to evict), so it
is always safe to leave on in a display helper that must handle both families.

## See also

- **How-to:** [Check data quality with `validate`](howto/validate.md)
- **Tutorial:** [Chapter 2 · L1 → L2 → L3](tutorial/02_three_resolutions.md),
  [Chapter 3 · Loading order data](tutorial/03_loading_data.md)
- **Reference:** [`FeedType`](api/protocols.md),
  [`data_quality_summary` / `order_book`](api/analytics.md),
  [Glossary → Data formats](glossary.md#data-formats)
