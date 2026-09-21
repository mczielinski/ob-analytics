---
title: Build a feature table
---

# Build a feature table

A model or a study wants one tidy table: a point in time on each row and a
microstructure feature in each column. The library measures all of those
already, but each in its own table on its own clock.
[`features()`](../api/features.md) does the join once.

```python
from ob_analytics import Pipeline, features, sample_csv_path
from ob_analytics.visualization import display_result

result = Pipeline().run(sample_csv_path())

# Pipeline prices are whole ticks and sizes whole lots; display_result
# converts both to the quote currency and the base asset.
shown = display_result(result)
trades, quotes = shown.trades, shown.depth_summary

table = features(trades, quotes, "volume", 0.05)
print(table[["timestamp", "close", "spread_bps", "obi", "trade_imbalance", "vpin"]])
```

```text
                       timestamp   close  spread_bps       obi  trade_imbalance  vpin
2026-05-02 02:36:23.889000+00:00 78319.0    0.127684  0.772174              1.0   1.0
2026-05-02 02:36:23.918000+00:00 78319.0    1.915085 -0.216647              1.0   1.0
2026-05-02 02:36:23.924000+00:00 78319.0    1.915085 -0.216647              1.0   1.0
2026-05-02 02:36:23.936000+00:00 78320.0    1.915085 -0.216647              1.0   1.0
2026-05-02 02:36:23.936000+00:00 78321.0    1.915085 -0.216647              1.0   1.0
```

Two decisions make the table, and they are separate: where the rows fall, and
what each column measures.

## Where the rows fall

The rows are [bars](bars.md), and `features()` takes the same sampling
arguments `bars()` does — the rule, the threshold, and `target_bars` when you
leave the threshold out. The same arguments give the same cut in both.

```python
features(trades, quotes, "time", "1min")      # a time grid
features(trades, quotes, "tick", 100)         # every 100 trades
features(trades, quotes, "volume", 0.5)       # every 0.5 BTC traded
features(trades, quotes, "dollar", 50_000)    # every $50,000 of turnover
features(trades, quotes, "imbalance", 0.5)    # every 0.5 BTC of net flow
features(trades, quotes, target_bars=200)     # let the rule choose
```

Activity rules are usually the better sampling for a model: a quiet hour and a
busy minute produce the same number of rows, so the rows come much closer to
being independent and identically distributed. The
[bars how-to](bars.md#activity-bars) explains the five rules.

The cut is recorded, so a table built with a defaulted threshold still says
what it was built on:

```python
table.attrs["bar_rule"]        # 'volume'
table.attrs["bar_threshold"]   # 0.05
table.attrs["features"]        # the feature names measured
```

## What each column measures

Each column comes from a **feature**. Ten ship with the package:

| Feature | Columns | Reads |
|---|---|---|
| `price` | `open`, `high`, `low`, `close`, `vwap` | trades |
| `returns` | `log_return`, `realized_vol` | trades |
| `flow` | `volume`, `turnover`, `n_trades`, `signed_volume`, `trade_imbalance` | trades |
| `spread` | `spread`, `spread_bps` | book |
| `mid_price` | `mid_price` | book |
| `micro_price` | `micro_price`, `micro_price_offset` | book |
| `imbalance` | `obi`, `obi_depth` | book |
| `depth` | `best_bid_vol`, `best_ask_vol`, `bid_depth`, `ask_depth` | book |
| `vpin` | `vpin` | trades |
| `kyle_lambda` | `kyle_lambda` | trades |

Name the ones you want, in the order you want their columns:

```python
features(trades, quotes, "volume", 0.5, include=["spread", "imbalance"])
# bar, timestamp_start, timestamp, spread, spread_bps, obi, obi_depth
```

Leave `include` out and every registered feature the inputs support is
measured. Without a quotes frame the five book features are skipped and the
table holds the trade features alone:

```python
table = features(trades, None, "volume", 0.5)
table.attrs["features_skipped"]
# ['spread', 'mid_price', 'micro_price', 'imbalance', 'depth']
```

Naming a book feature without quotes raises instead: an explicit request that
cannot be met is an error, while a default that cannot be met is a smaller
table.

## No look-ahead

Each row is stated as of the close of its bar, which is the `timestamp`
column. The trade columns hold what happened between `timestamp_start` and
that instant, and the book columns hold the book as it stood at it — a
backward as-of join, so a row takes the last quote published at or before its
close. Nothing from later reaches the row.

The trailing-window features — `realized_vol`, `vpin`, `kyle_lambda` — look
back over the last 20 rows, the row's own included, and are `NaN` until there
is enough history behind them. A row that closed before the first quote has no
book to read, and is `NaN` too.

One thing is not settled by the table: the *choice* of threshold. Left to
default it is worked out from the whole trades frame, so where the boundaries
fall depends on the whole capture. Pass a threshold when the cut itself has to
be something the rows could have been given at the time.

## Build a target

The table is features only. A target looks forward, and building one is the
one place look-ahead belongs — so it is yours to write, deliberately:

```python
table["target"] = table["log_return"].shift(-1)   # the next bar's return
```

The last row then has no target, and rows whose trailing windows are still
filling have no features. Drop both together with `.dropna()`.

## A baseline model

Enough to see the shape. Six features, ordinary least squares, and a
chronological split — never a random one, because shuffling rows lets the
model learn from the future:

```python
import numpy as np

table = features(trades, quotes, "volume", 0.05)
table["target"] = table["log_return"].shift(-1)

columns = [
    "obi",
    "obi_depth",
    "micro_price_offset",
    "trade_imbalance",
    "spread_bps",
    "vpin",
]
fit = table[[*columns, "target"]].dropna()
split = int(len(fit) * 0.7)
train, test = fit.iloc[:split], fit.iloc[split:]


def design(frame):
    return np.column_stack([np.ones(len(frame)), frame[columns].to_numpy(float)])


beta, *_ = np.linalg.lstsq(design(train), train["target"].to_numpy(float), rcond=None)


def r_squared(frame):
    actual = frame["target"].to_numpy(float)
    predicted = design(frame) @ beta
    return 1 - ((actual - predicted) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()


print(f"in sample      R² {r_squared(train):+.3f}   ({len(train)} rows)")
print(f"out of sample  R² {r_squared(test):+.3f}   ({len(test)} rows)")
```

```text
in sample      R² +0.186   (88 rows)
out of sample  R² +0.079   (38 rows)
```

Read that as a demonstration of the shape, not as a finding. The bundled
sample is ten minutes of one instrument, so 38 test rows can say almost
anything; and the split is one split, not a walk forward. What it does show is
the workflow: sample on activity, measure as of each row's close, shift a
target back by one row, and split in time.

## A feature of your own

A feature says what one column measures and nothing else. Register one and
`features()` puts it in the table — see
[Extending](../extending.md#6-a-new-feature).

## Somewhere else

The table is a plain pandas frame, so it goes wherever one does:

```python
table.to_parquet("features.parquet")

import polars as pl
pl.from_pandas(table)
```

Polars is not a dependency of ob-analytics; install it yourself to run that
second line. See the [schema page](../schema.md) for what the package
guarantees about frames and files.
