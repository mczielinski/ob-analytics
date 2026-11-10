# Data quality: matched book, diff feed and price levels

Before you trust a reconstructed order book, you need to know **what kind of
feed produced it**. The single most important property is the book's
*crossing invariant* — whether a resting buy order can ever sit at a higher
price than a resting sell order. It can, in some feeds, and that fact changes
how you read every downstream plot and metric.

This page explains the distinction, why a crossed book is often *faithful*
rather than a bug, and the two tools `ob-analytics` gives you to work with it:
the [`audit`](howto/audit.md) command that **measures** data quality,
and the `uncross=` option that **cleans the book up for display**.

Crossing is one of several properties that decide what a capture can tell you.
[What each feed shows](feeds.md) sets them all out, source by source.

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
from ob_analytics import BitstampSource, LobsterSource, FeedType

BitstampSource().feed_type   # FeedType.DIFF_FEED
LobsterSource().feed_type    # FeedType.MATCHED_BOOK
FeedType.DIFF_FEED == "diff_feed"   # True — the enum mixes in str
```

A format that predates the attribute reads back as `FeedType.UNKNOWN`, so you
can always do `getattr(fmt, "feed_type", FeedType.UNKNOWN)` without special-casing.

## Price-level (L2) feeds

An L2 feed has no orders, only the venue's total size at each price. Every L2
source declares `FeedType.PRICE_LEVELS`: the ccxt source, cryptofeed at L2, and
`depth_csv`. The venue does not publish a crossed book, so a crossed capture
means the capture's copy is wrong, most often because it kept a level that the
venue removed. `audit` scores it as an error, as for a matched book.

In test captures of nine L2 feeds (Binance, Coinbase, Kraken, two Kalshi and two
Polymarket markets, and cryptofeed's Bitstamp and Kraken books), none was
crossed for any measurable time. Some showed crossings that lasted no time at
all: one venue update is written as several rows, and between those rows the
book can briefly be half old, half new.

Captures made before this value existed record `matched_book` in `meta.json`.
`audit` reads the record, so it scores them the same way.

## A crossed book is faithful, not a bug

A diff feed does carry real crossings. An aggressive order can rest for a few
events until the venue reports its execution, and on the bundled Bitstamp
sample there are about 100,000 such crossings, each shorter than a
millisecond. That is a property of the public feed, not a defect in the
reconstruction.

A crossing that lasts is a different matter. On the same sample, almost all of
the crossed time comes from one order: an ask at $78,333.00 that the opening
snapshot reported and the venue never reported again. Trades print above it
for 27 minutes while it holds the ask touch. It is a
[stale resting order](#stale-resting-orders), and `audit` names it.

[`order_book()`](api/book.md) therefore replays a diff feed **as-is**: a
crossed book in the output is a property of the feed, not a reconstruction
error. The tutorial meets this pitfall twice — once in
[Chapter 2 · L1 → L2 → L3](tutorial/02_three_resolutions.md) and again in
[Chapter 3 · Loading order data](tutorial/03_loading_data.md) — and the
[glossary](glossary.md#data-formats) defines both terms.

!!! warning "Don't silently uncross data you're going to analyse"
    The crossing *is the signal* for a diff feed — it tells you the feed is
    unmatched and that spreads, mid-prices, and depth near the touch need care.
    Uncrossing (below) is a **display** convenience; never bake it into the
    data an analysis runs on, or you erase the very property `audit` is
    there to surface.

## Measuring it: `audit`

The [`ob-analytics audit`](howto/audit.md) command runs the pipeline, prints a
per-run data-quality summary, and exits non-zero when a check fails. On the
bundled Bitstamp sample:

<!-- docs-test
import subprocess
import sys

audit_report = subprocess.run(
    [sys.executable, "-m", "ob_analytics", "audit", "orders.csv"],
    stdout=subprocess.PIPE, text=True, check=False,
).stdout
-->

<!-- docs-test: output audit_report -->

```text
Data quality summary
  feed type             : diff_feed
  events / orders       : 314,057 / 156,902
  trades                : 284
  crossed resting book  : 91.61% of session (8886 episode(s)) [diff feed, but 2 stale resting order(s) stay in the book — see stale resting orders]
  per-order book crossed: 91.61% of session (8886 episode(s)) [no more than the depth table — see the crossed resting book]
  venue book            : not checked: no venue book to compare with; only a LOBSTER run from its files has one
  stale resting orders  : 2 (worst: ask 2002347646152704 at 78,333 held the ask touch for 27.4 min after a trade printed through it)
  unmatched trades      : 0.00% [maker and taker]
  matched by order      : 2 trade(s) [taker created with size 0]
  duplicate event ids   : 0
  duplicate created ids : 0
  pre-existing orders   : 13
  orphan orders         : 13 (13 event(s), no created row)
  impossible values     : 49 non-positive price(s) / 0 negative volume(s) / 0 level row(s) below zero
  clock order           : 0 venue-after-receive / 11 reordered
  records after a gap   : not checked: the events carry no count of records flagged after a gap (only Databento flags them)
  venue sequence        : 0 missing / 0 out-of-order (0 row(s) numbered)
Checks: 0 error(s), 4 warning(s)
  WARNING orphan_orders: 13 order(s) are changed or deleted with no created event ...
  WARNING stale_orders: 2 resting order(s) a trade printed through and the venue did not report again within 1 s ...
  WARNING nonpositive_price: 49 row(s) are priced at or below zero: not a tradeable level
  WARNING exchange_time_reordered: 11 message(s) arrived out of venue order ...
```

A matched book (LOBSTER) reports **~0%** crossed on the same metric — the
number is the cleanest single discriminator between the two families. On a
diff feed, read it together with the stale-order line below it.

## Stale resting orders

A matching engine fills the better price first. So a trade above a resting
ask, or below a resting bid, shows that the order has already left the book.
The venue normally reports it a few milliseconds later: on the bundled sample,
a median of 23 ms after the trade and 234 ms at the 95th percentile.

An order the venue does not report again within one second is **stale**. The
rebuilt book goes on holding it, and it distorts the spread, the depth and the
queue from then on. On the bundled sample two orders are stale, and both came
from the opening REST snapshot. Removing each one at the moment a trade proved
it gone takes the crossed share of session time from 91.6% to 1.3%.

Stale orders distort the book that `order_book()` rebuilds, and everything
read from it, such as the crossed share above. They reach the depth summary
much less, because the depth summary leaves a level out of its touch while a
newer quote on the other side crosses it. On the bundled sample, without the
two stale orders the depth summary's best ask changes on 0.5% of rows, the
trades with a negative effective spread go from 48 to 45, no trade is flagged
as hidden either way, and the table the
[feature-table how-to](howto/feature-table.md) builds changes by a small
amount on 19 of its 127 rows.

`audit` reports stale orders as a warning and names the worst one: its id,
side, price, and how long it held the touch after the trade. From Python,
[`detect_stale_orders`](api/quality.md) returns every one. Neither removes
anything:

- The test needs to know what the feed said *after* the trade. A live capture
  cannot know that in time, so a repair would make offline and live
  reconstruction differ.
- Without the grace period the test is wrong most of the time: it catches
  orders whose delete is already on its way.
- `order_book()` replays what the feed said. Changing that to what we think the
  feed meant is a different promise.

Stale orders are not dropped messages: this capture records none. Nor can
sequence-gap detection find them: Bitstamp publishes a timestamp, not a
sequence number.

Each metric is scored by a named check with a severity, and the exit code
follows the severities: an **error** fails the run, a **warning** only fails it
under `--strict`. Which severity crossing gets is decided by the feed type —
the whole point of this page. The [`audit` how-to](howto/audit.md) lists every
check and what trips it.

The headline metrics:

| Metric | What it measures | Matched book | Diff feed |
|---|---|---|---|
| **crossed resting book %** | Share of session *time* the faithful book has `best_bid > best_ask` | ~0% | often high (~92% here, almost all of it from one stale order) |
| **stale resting orders** | Resting orders a trade printed through that the venue did not report again within 1 s | 0 | 0 (else a feed defect) |
| **unmatched trades %** | Trades with no resolvable maker/taker resting order | low | low–moderate |
| **duplicate ids** | `event_id`s seen twice, or order ids created twice | 0 | 0 (else a feed defect) |
| **pre-existing orders** | Orders already resting when the capture began (no `created` row) | a few | a few |

The crossing figure is measured from the **faithful** resting book (via
[`price_level_volume`](api/depth.md)), *not* from `depth_summary` — the depth
summary leaves crossed levels out of its touch, so it would always report ~0%. It is
duration-weighted, so "crossed for 90 seconds out of a 10-minute capture"
reads as ~15%, matching intuition.

## Cleaning it up for display: `uncross=`

When you want a clean `best_bid < best_ask` ladder from a diff feed — for a
figure, not for analysis — pass `uncross=True`:

```python
from ob_analytics import Pipeline, sample_csv_path
from ob_analytics.book import order_book

events = Pipeline().run(sample_csv_path()).events

faithful = order_book(events)                 # default: crossed if the feed is
display  = order_book(events, uncross=True)   # best_bid < best_ask everywhere
```

The `book_snapshot` ladder and the `depth_chart` curve draw the book they are
given, so pass them the uncrossed book:

```python
from ob_analytics.visualization import prepare, plot

fig = plot("book_snapshot", level="L3",
           **prepare.book_snapshot(order_book=display, per_order=True))
```

Uncrossing applies to orders the rule the depth summary applies to price
levels: at the crossed (or locked) touch it **trusts the fresher quote** and
evicts the older, stale opposing order, repeating until the sides no longer
overlap. It never
fabricates liquidity — it only removes orders — and it recomputes each side's
cumulative depth against the surviving touch.

| Question | Use |
|---|---|
| "What did the feed actually contain?" | **`uncross=False`** (default) — faithful |
| "Draw me a clean ladder / depth curve for a slide" | `uncross=True` — display only |
| "How crossed is this feed?" | [`audit`](howto/audit.md) — never uncross first |

On a matched book `uncross=True` is a no-op (there is nothing to evict), so it
is always safe to leave on in a display helper that must handle both families.

## Checked against other engines

A golden file catches a change, but not a mistake that was already there when
the file was recorded. So the rebuilt book is also checked against two engines
written by other people. `tests/test_backtest_parity.py` exports a session,
replays it through hftbacktest's own L3 reconstruction and Nautilus Trader's
own order book, and compares the best bid and ask with `depth_summary` on every
row.

The result, on five synthetic L3 sessions (seeds 224, 143, 7, 99 and 1234):
both engines agree with `depth_summary` on every row. Before a fix to the
export writers (they exported orders that never rest as resting orders), they
disagreed on up to 78 rows a seed. The check first ran with hftbacktest
2.4.2 and Nautilus Trader 1.231.0.

The bundled Bitstamp sample is not compared. It is a diff feed whose book
crosses (see [A crossed book is faithful](#a-crossed-book-is-faithful-not-a-bug)),
which this package keeps on purpose. An engine that uncrosses or refuses a
crossed book would disagree for that reason alone, so a mismatch there would
not show which side is wrong.

To run it, install the `backtest-engines` dependency group:
`uv sync --group dev --group backtest-engines`, then
`uv run pytest tests/test_backtest_parity.py`. Nautilus needs Python 3.12 or
newer; on 3.11 its half skips.

## Checked against Databento's own BBO

The engines above check the rebuild on synthetic sessions. A real Databento
file can be checked against Databento itself. Databento publishes the best bid
and offer (BBO) for each instrument, built from the same feed as its
market-by-order (MBO) records. `scripts/databento_bbo_check.py` rebuilds the
book from an MBO file with `Pipeline(source=DatabentoSource())` and compares
its best bid and ask with Databento's.

The check needs two files for the same instrument and time range:

- the **MBO** file, which the package rebuilds the book from;
- a **top-of-book** file in the `mbp-1` schema (one record each time the top
  of the book changes) or the `bbo-1s` or `bbo-1m` schema (one record a second
  or a minute). `mbp-1` is the stronger check, because every change is
  compared. `bbo-1s` is smaller, but it does not see a wrong state that lasts
  less than a second.

The MBO file must start at the start of the venue's day. The rebuilt book
starts empty, so an order resting before the file starts is missing from it.

How the two are lined up:

- Both files stamp each record with `ts_recv`, Databento's receive time. For
  each top-of-book record, the check reads the rebuilt book after every MBO
  record received at or before that time.
- Records received at the same instant are one update. On both sides, only the
  last record of each instant is compared.
- A record **matches** when the best bid price and the best ask price are both
  the same. Sizes are compared too, and reported apart.
- Mismatches are grouped into episodes: runs of consecutive mismatched
  records. For the largest ones, the script prints both books and the MBO
  events just before the first mismatch.

To run it, install the `databento` extra and pass the two files:

```bash
uv run --extra databento python scripts/databento_bbo_check.py \
    XNAS.ITCH-AAPL-20240212.mbo.dbn.zst XNAS.ITCH-AAPL-20240212.mbp-1.dbn.zst
```

For a file of several million records, add `--cut TIME` once or more. The book
is then rebuilt one window at a time with `run_windows`, which carries the book
across each cut. `--csv PATH` writes every mismatched record to a file.
`tests/test_databento_bbo_check.py` runs the check on a short hand-built MBO
stream and its known top of book. It needs no key.

**Result:** not run on a real file yet. The result, with the dataset, symbol
and date range, will be recorded here after the first run.

## See also

- **How-to:** [Check data quality with `audit`](howto/audit.md)
- **Tutorial:** [Chapter 2 · L1 → L2 → L3](tutorial/02_three_resolutions.md),
  [Chapter 3 · Loading order data](tutorial/03_loading_data.md)
- **Reference:** [`FeedType`](api/protocols.md),
  [`data_quality_summary`](api/quality.md), [`order_book`](api/book.md),
  [Glossary → Data formats](glossary.md#data-formats)
