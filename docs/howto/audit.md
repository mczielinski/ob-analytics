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
  orphan orders         : 13 (13 event(s), no created row)
  impossible values     : 49 non-positive price(s) / 0 negative volume(s)
  clock order           : 0 venue-after-receive / 11 reordered
  venue sequence        : 0 missing / 0 out-of-order (0 row(s) numbered)
Checks: 0 error(s), 3 warning(s)
  WARNING orphan_orders: 13 order(s) are changed or deleted with no created event ...
  WARNING nonpositive_price: 49 row(s) are priced at or below zero: not a tradeable level
  WARNING exchange_time_reordered: 11 message(s) arrived out of venue order ...
```

## Reading the metrics

| Field | Read it as |
|---|---|
| **feed type** | `matched_book` (LOBSTER/MBO) or `diff_feed` (Bitstamp) — sets expectations for the next line |
| **crossed resting book** | Share of session *time* with `best_bid > best_ask`. ~0% for a matched book; high is normal and faithful for a diff feed |
| **unmatched trades** | Trades with no resolvable maker/taker resting order |
| **duplicate event ids / created ids** | Should be `0`; anything else is a feed defect worth chasing |
| **pre-existing orders** | Orders already resting when the capture began (no `created` row) — structurally unclassifiable, not errors |
| **orphan orders** | Orders changed or deleted with no `created` row at all. The opening book is the honest source of these; a rise mid-session is the stream losing messages |
| **impossible values** | Levels priced at or below zero, and negative volumes or fills |
| **clock order** | Rows the venue stamped *after* we received them, and messages that reached the capture out of venue order |
| **venue sequence** | Skipped and non-advancing sequence numbers: dropped and reordered messages ([gap detection](../api/analytics.md)) |

A high **crossed resting book** number on a `diff_feed` is expected — see
[Data quality: matched book vs diff feed](../data-quality.md) for why, and for
the `uncross=` option that cleans the book up *for display* without touching
the data you analyse. On a `matched_book`, a non-zero figure is a red flag.

## What fails a run

Every metric above is scored by a named check carrying a severity, and the exit
code follows the severities rather than the numbers:

| Severity | Meaning | Exit code |
|---|---|---|
| **error** | The data contradicts something that must hold | non-zero |
| **warning** | Worth reading, but a sound capture can show it | `0`, or non-zero with `--strict` |
| **info** | Context; never fails a run | `0` |

Errors: `duplicate_event_ids`, `duplicate_created_ids`, `sequence_gaps`,
`sequence_out_of_order`, `negative_volume`, `exchange_time_after_receive`, and
`crossed_book` **on a matched book only**.

Warnings: `orphan_orders`, `nonpositive_price`, `exchange_time_reordered`,
`unmatched_trades` (above 5%), and `crossed_book` when no feed type was
declared.

Two of these are judgement calls worth stating plainly:

- **A crossed book is scored by feed type, not by size.** The same 92% is a
  defect in a matched book and a faithful replay of a diff feed, and only the
  source's declared [`FeedType`](../api/protocols.md) can tell them apart. With
  `--from-parquet` and no `--source`, the feed type is undeclared and crossing
  drops to a warning rather than being guessed.
- **A dropped `created` message cannot be told apart from an order that was
  already resting** when the capture began — both leave an order that is only
  ever changed or deleted. So `orphan_orders` is a warning, and the hard
  evidence for dropped messages is `sequence_gaps`, which needs a feed that
  carries a venue sequence. `audit` always loads with sequence tracking on.

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
      "severity": "error",
      "detail": "0 event_id value(s) occur more than once; ..."
    }
  ]
}
```

## From Python

```python
from ob_analytics import Pipeline, BitstampSource, FeedType, data_quality_summary

result = Pipeline().run("orders.csv")
summary = data_quality_summary(
    result.events, result.trades,
    feed_type=BitstampSource().feed_type,   # or getattr(source, "feed_type", FeedType.UNKNOWN)
    depth=result.depth,                      # faithful depth; not depth_summary
)
print(summary.render())
summary.ok            # False when an error-severity check failed
summary.errors        # the failed error checks, each with a one-line detail
summary.warnings      # the failed warning checks
summary.to_dict()     # JSON-serialisable, including every check
```

!!! note "Pass `depth`, not `depth_summary`"
    Crossing is measured from the *faithful* resting book. `depth_summary` is
    already uncrossed by the depth engine, so passing it would always report
    ~0%. Omit `depth` and it is recomputed from `events`.

## Related

- [Data quality: matched book vs diff feed](../data-quality.md) — the concepts behind these numbers
- [Run from the command line](cli.md) — every CLI verb
- [`data_quality_summary` reference](../api/analytics.md)
