---
title: Check data quality with audit
---

# Check data quality with `audit`

`ob-analytics audit <source>` runs the pipeline, prints a per-run data-quality
summary, and **exits non-zero when a check fails** — so it works both as a
thing to read and as a gate in a script. Point it at the same source you would
pass to `process`:

```bash
ob-analytics audit orders.csv
ob-analytics audit data/ --source lobster --trading-date 2012-06-21
ob-analytics audit orders.csv --json           # machine-readable, for CI
ob-analytics audit results/ --from-parquet     # a saved 'process' output
ob-analytics audit orders.csv --strict         # warnings fail too
```

`validate` is the old name for this verb and still works.

Text output looks like this (bundled Bitstamp sample):

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
  crossed resting book  : 91.61% of session (8883 episode(s)) [diff feed, but 2 stale resting order(s) stay in the book — see stale resting orders]
  per-order book crossed: 91.61% of session (8883 episode(s)) [no more than the depth table — see the crossed resting book]
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

## Reading the metrics

| Field | Read it as |
|---|---|
| **feed type** | `matched_book` (LOBSTER/MBO), `diff_feed` (Bitstamp) or `price_levels` (any L2 feed) — sets expectations for the next line |
| **crossed resting book** | Share of session *time* with `best_bid > best_ask`. ~0% for a matched book or price levels; can be high and faithful for a diff feed, unless stale resting orders cause it |
| **per-order book crossed** | The same share for the book rebuilt from the per-order events. On most runs that book is the depth table, so the two lines agree. A LOBSTER run with an orderbook file reads its depth table from that file, and the per-order book can then be crossed when the depth table is not: see [LOBSTER: too few levels](lobster.md#request-enough-levels) |
| **venue book** | Whether the book rebuilt from the per-order events matches the venue's own book, row by row, on every price and size of the levels the venue states. Only LOBSTER supplies one, its orderbook file. Gives the number of rows that differ and the first difference. `not checked` for other sources and for a saved run (`--from-parquet`), which has no orderbook file: see [What fails a run](#what-fails-a-run) |
| **stale resting orders** | Resting orders a trade printed through that the venue did not report again within 1 s. The worst is named with its side, price and how long it held the touch |
| **unmatched trades** | Trades whose maker or taker order could not be found among the order events. Only the orders the feed can show are looked for: the note in brackets says which |
| **matched by order** | Trades whose taker is an order created with size 0. Bitstamp sends an instant order bought by quote amount that way and never reports what it bought, so a trade with no row of its size is matched to the order by its id. The order and its type are right; its rows show less fill than its trades. Information only; `not checked` when the feed does not show takers |
| **duplicate event ids / created ids** | Should be `0`; anything else is a feed defect worth chasing |
| **pre-existing orders** | Orders already resting when the capture began (no `created` row) — structurally unclassifiable, not errors |
| **orphan orders** | Orders changed or deleted with no `created` row at all. The opening book is the honest source of these; a rise mid-session is the stream losing messages |
| **impossible values** | Levels priced at or below zero, negative volumes or fills, and depth rows where the events take more off a price level than they put on (the depth table holds such a level at zero) |
| **clock order** | Rows the venue stamped *after* we received them, and messages that reached the capture out of venue order. The opening book's rows are left out; `not checked` when the data has one clock |
| **records after a gap** | Records the venue flagged as coming after a gap in its feed that it could not repair, so the book may be wrong from the first of them on. Only Databento flags these (`F_MAYBE_BAD_BOOK`); `not checked` for every other source |
| **venue sequence** | Skipped and non-advancing sequence numbers: dropped and reordered messages ([gap detection](../api/analytics.md)) |

A high **crossed resting book** number on a `diff_feed` can be expected — see
[Data quality: matched book, diff feed and price levels](../data-quality.md) for why, and for
the `uncross=` option that cleans the book up *for display* without touching
the data you analyse. On a `matched_book` or `price_levels`, a non-zero figure
is a red flag.

Read it together with **stale resting orders**. On the bundled sample one
stale order holds the ask touch for 27 minutes, and it causes almost all of the
91.61%. When the run has stale orders, the crossing note says so instead of
calling the crossing normal.

## What fails a run

Every metric above is scored by a named check carrying a severity, and the exit
code follows the severities rather than the numbers:

| Severity | Meaning | Exit code |
|---|---|---|
| **error** | The data contradicts something that must hold | non-zero |
| **warning** | Worth reading, but a sound capture can show it | `0`, or non-zero with `--strict` |
| **info** | Context; never fails a run | `0` |

A check that cannot run on the data is **skipped**: it says `not checked` and
why, its severity is info, and in `--json` it has `"skipped": true` and
`"passed": false`, so it is not counted as a pass. The clock checks on data
with one clock, `venue_book` with no venue book to compare (any source but
LOBSTER, and a saved run), and `maybe_bad_book` on events with no count of
records flagged after a gap (any source but Databento) are skipped.

Errors: `duplicate_event_ids`, `duplicate_created_ids`, `sequence_gaps`,
`sequence_out_of_order`, `negative_volume`, `exchange_time_after_receive`, and
`crossed_book` **on a matched book or price levels only**.

Warnings: `orphan_orders`, `stale_orders`, `nonpositive_price`,
`negative_level`, `exchange_time_reordered`, `unmatched_trades` (above 5%),
`per_order_book_crossed` when the per-order book is crossed for longer than the
depth table, `venue_book` when the rebuilt book differs from the venue's own
book in any row, `maybe_bad_book` when the venue flagged records after a gap,
and `crossed_book` when no feed type was declared.

Two of these are judgement calls worth stating plainly:

- **A crossed book is scored by feed type, not by size.** A crossed book is
  always a defect in a matched book or a price-level book, but can be a
  faithful replay of a diff feed. Only the source's declared [`FeedType`](../api/protocols.md) can tell
  them apart. Size alone says little: on the bundled sample, almost all of the
  92% comes from two stale orders, which `stale_orders` reports. A saved
  output records the feed type of the run that made it. One written by an
  older version does not: with `--from-parquet` and no `--source`, its feed
  type is undeclared and crossing drops to a warning rather than being
  guessed.
- **A dropped `created` message cannot be told apart from an order that was
  already resting** when the capture began — both leave an order that is only
  ever changed or deleted. So `orphan_orders` is a warning, and the hard
  evidence for dropped messages is `sequence_gaps`, which needs a feed that
  carries a venue sequence. `audit` always loads with sequence tracking on.
  A skipped number is a dropped message only when the venue adds one per
  message. A [ccxt](ccxt.md) or [cryptofeed](cryptofeed.md) capture records
  in `meta.json` that its sequence only rises, and `audit` then checks only
  that it never goes back, and prints `gaps not checked`. When a cryptofeed
  capture reconnects, some venues start the count again; the capture records
  each such step back as `sequence_restarts`, and `audit` leaves those out of
  `sequence_out_of_order`.

  Most sources carry no sequence that can prove a message was lost. On those,
  `audit` prints `0 row(s) numbered` for the venue sequence, and a dropped
  message shows only indirectly, as orphan or stale orders:

  | Source | Venue sequence | What `audit` can check |
  |---|---|---|
  | `bitstamp` (files and live capture) | none: the feed has a timestamp only | nothing |
  | `lobster` | none | nothing |
  | `cryptofeed` | one number per message, on venues that publish one | gaps and order |
  | `ccxt`, including Binance | rises, but skips on its own | order only |
  | `ccxt` for Kalshi and Polymarket | none | nothing |
  | `databento` | rises, but skips on its own | order only |

  Databento numbers every message on the venue's channel. A file usually holds
  one instrument of that channel, and its trade and fill records become
  trades rather than book events, so the numbers left in the events skip even
  when nothing was lost. The source declares this, and `audit` then prints
  `gaps not checked`. A saved Databento output records this too. For one
  written by an older version, pass `--from-parquet --source databento`:
  without `--source`, nothing says what the numbers promise, and every skip
  counts as a lost message.

  So Databento's own flag is its only gap signal. When Databento finds a gap
  it cannot repair, it marks the records after it with `F_MAYBE_BAD_BOOK`.
  The loader keeps those records, warns, and counts them, and `audit` reports
  the count as the `maybe_bad_book` warning. It is a warning, not an error,
  as `negative_level` is: the records are what the venue sent, and dropping
  them would not make the book right.
- **Unmatched trades count only the orders the feed can show.** Every trade
  has a maker, the order that was resting, and a taker, the order that
  arrived and traded against it. Only a feed that reports every order shows
  the taker. Each source declares which it can show, as its
  [`TradeAttribution`](../api/protocols.md):

  | Source | Shows | What `unmatched_trades` counts |
  |---|---|---|
  | `bitstamp` | maker and taker | trades missing either |
  | `lobster`, `databento`, `cryptofeed` at L3 | maker only: a taker trades on arrival and never rests | trades missing the maker |
  | `ccxt`, `depth_csv`, `cryptofeed` at L2 | neither: price levels have no order identity | nothing |

  LOBSTER's trades do carry a taker, but it is a guess: Nasdaq ITCH names only
  the resting order of an execution, and the reader picks the most recent new
  order on the other side that could have traded. So the check does not count
  it.
- **The clock checks need two clocks.** They compare the venue's time,
  `exchange_timestamp`, with the receive time, `timestamp`. Some data has only
  one, and the schema copies it into both columns: LOBSTER has the venue's
  time only, and some live venues send no time with their book. Each source
  declares its [`Clocks`](../api/protocols.md), and a live capture records
  what the venue sent in `meta.json`. With one clock, `audit` prints
  `clock order : not checked` and the reason, and the two checks do not run.
  With two clocks, the checks still leave out a capture's opening book (rows
  whose `origin` is `snapshot`): a REST book's receive time is when the
  capture asked for it, and some venues send it with no venue time. See
  [Clocks](../feeds.md#clocks) for which feeds have which.
- **A capture is checked against the source that made it.** A live capture
  records its source and that source's declarations in `meta.json`, and
  `ob-analytics process` copies `meta.json` into its output. `audit` uses the
  record even when `--source` names another source. `--source` also says how
  to read the files, and a [cryptofeed](cryptofeed.md) L3 capture can only be
  read as `bitstamp`, whose feed shows more. The log says when the record
  overrides `--source`.
- **A capture of several segments is audited one segment at a time.** Given a
  capture directory (or the `process` output made from one), `audit` prints a
  report for each segment and then the capture's own checks from
  `manifest.json`: `capture_gaps` (time no segment covered),
  `unfinished_segments` (segments a dead capture process left open),
  `dropped_messages` (messages a source could not use, or a count lost with
  the source's counters), `capture_errors` (a segment that ended on an error, or was slow to stop)
  and `book_resyncs` (times a source lost the venue's stream and started again
  from a new opening book). All five are warnings, so `--strict` fails on
  them. A single capture gets `dropped_messages` and `capture_errors` from its
  `meta.json`. See [Running for days](live-capture.md#running-for-days).
- **The rebuilt book against the venue's.** A LOBSTER orderbook file states
  the best N price levels of each side after every message. `venue_book`
  rebuilds the book from the messages and compares it with the file, row by
  row, on every price and size of those N price levels. Each event is matched with
  the file's row of its own message (its `original_number`), not the row at
  its position: the file also has rows for cross trades and halts, which are
  not events. The rebuilt book keeps every price level its orders rest at, crossed
  or not. The check is a warning because the depth table, which a LOBSTER
  run reads from the file, is right either way; the per-order views read the
  rebuilt book. On LOBSTER's own AAPL samples it fails: the message file does
  not cover every order in the book. See
  [LOBSTER: request enough levels](lobster.md#request-enough-levels).
- **A stale order is reported, not removed.** A trade above a resting ask (or
  below a resting bid) shows the order has gone, because a matching engine
  fills the better price first. The venue normally reports that order within
  milliseconds, so `stale_orders` waits one second before it counts one. The
  test needs to know what the feed said after the trade, which a live capture
  cannot know in time, so `order_book()` keeps the order.
  [`detect_stale_orders`](../api/analytics.md) runs the same test from Python.

## In CI

```bash
ob-analytics audit orders.csv --json > quality.json || exit 1
```

`--json` writes the whole summary, including `ok` and every check, so a build
can read the verdict instead of parsing the text block:

```json
{
  "feed_type": "diff_feed",
  "orphan_orders": 13,
  "ok": true,
  "checks": [
    {
      "name": "duplicate_event_ids",
      "passed": true,
      "skipped": false,
      "severity": "error",
      "detail": "0 event_id value(s) occur more than once; ..."
    }
  ]
}
```

## From Python

A run's result carries what its data is declared to show (its
`declarations`: feed type, trade attribution, sequence kind and clocks), and
`result.audit()` holds the run to them, as `ob-analytics audit` does:

```python
from ob_analytics import Pipeline

result = Pipeline().run("orders.csv")
summary = result.audit()
print(summary.render())
summary.ok            # False when an error-severity check failed
summary.errors        # the failed error checks, each with a one-line detail
summary.warnings      # the failed warning checks
summary.stale_orders  # StaleOrder records, worst first
summary.to_dict()     # JSON-serialisable, including every check
```

The declarations come from the run's source. For a live capture, what its
`meta.json` records replaces them, since the capture's own source declared
those. A result saved with `save_data` keeps them, so
`load_result(folder).audit()` checks it the same way.

The sequence checks need the venue's numbers, which a run keeps only with
`PipelineConfig(track_sequence=True)`; `audit` turns this on for you:

<!-- docs-test: skip needs a Databento file -->

```python
from ob_analytics import DatabentoSource, Pipeline, PipelineConfig

source = DatabentoSource()
result = Pipeline(PipelineConfig(track_sequence=True), source=source).run("aapl.mbo.dbn.zst")
summary = result.audit()
```

To compare a LOBSTER run with its orderbook file, pass the venue's book as
`venue_book`. `LobsterSource().venue_book(folder, config)` reads it, or
returns `None` when the folder has no orderbook file:

<!-- docs-test: skip needs LOBSTER files -->

```python
from ob_analytics import LobsterSource, Pipeline, RunContext

source = LobsterSource()
result = Pipeline(source=source, ctx=RunContext(trading_date="2012-06-21")).run("data/")
summary = result.audit(venue_book=source.venue_book("data/", result.config))
summary.venue_book_differing_rows    # rows where the two books differ
summary.venue_book_first_difference  # a BookDifference, or None
```

For tables that are not a run's, call `data_quality_summary(events, trades,
...)` and pass the declarations yourself.

!!! note "Pass `depth`, not `depth_summary`"
    Crossing is measured from the *faithful* resting book. `depth_summary` is
    already uncrossed by the depth engine, so passing it would always report
    ~0%. Omit `depth` and it is recomputed from `events`.

## Related

- [Data quality: matched book, diff feed and price levels](../data-quality.md) — the concepts behind these numbers
- [What each feed shows](../feeds.md) — what each source can and cannot show, so which checks can fire on it
- [Run from the command line](cli.md) — every CLI verb
- [`data_quality_summary` reference](../api/analytics.md)
