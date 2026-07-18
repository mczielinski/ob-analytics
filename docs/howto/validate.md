---
title: Check data quality with validate
---

# Check data quality with `validate`

`ob-analytics validate <source>` runs the pipeline and prints a per-run
data-quality summary — the health signals worth checking before you trust a
feed. Point it at the same source you would pass to `process`:

```bash
ob-analytics validate orders.csv
ob-analytics validate data/ --format lobster --trading-date 2012-06-21
ob-analytics validate orders.csv --json      # machine-readable, for CI
```

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
```

## Reading the metrics

| Field | Read it as |
|---|---|
| **feed type** | `matched_book` (LOBSTER/MBO) or `diff_feed` (Bitstamp) — sets expectations for the next line |
| **crossed resting book** | Share of session *time* with `best_bid > best_ask`. ~0% for a matched book; high is normal and faithful for a diff feed |
| **unmatched trades** | Trades with no resolvable maker/taker resting order |
| **duplicate event ids / created ids** | Should be `0`; anything else is a feed defect worth chasing |
| **pre-existing orders** | Orders already resting when the capture began (no `created` row) — structurally unclassifiable, not errors |

A high **crossed resting book** number on a `diff_feed` is expected — see
[Data quality: matched book vs diff feed](../data-quality.md) for why, and for
the `uncross=` option that cleans the book up *for display* without touching
the data you analyse. On a `matched_book`, a non-zero figure is a red flag.

## From Python

```python
from ob_analytics import Pipeline, BitstampFormat, FeedType, data_quality_summary

result = Pipeline().run("orders.csv")
summary = data_quality_summary(
    result.events, result.trades,
    feed_type=BitstampFormat().feed_type,   # or getattr(fmt, "feed_type", FeedType.UNKNOWN)
    depth=result.depth,                      # faithful depth; not depth_summary
)
print(summary.render())
summary.to_dict()   # JSON-serialisable
```

!!! note "Pass `depth`, not `depth_summary`"
    Crossing is measured from the *faithful* resting book. `depth_summary` is
    already uncrossed by the depth engine, so passing it would always report
    ~0%. Omit `depth` and it is recomputed from `events`.

## Related

- [Data quality: matched book vs diff feed](../data-quality.md) — the concepts behind these numbers
- [Run from the command line](cli.md) — every CLI verb
- [`data_quality_summary` reference](../api/analytics.md)
