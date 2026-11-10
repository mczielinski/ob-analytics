---
title: Compute flow toxicity
---

# Compute VPIN, Kyle's λ, and order-flow imbalance

Detect informed trading and measure price impact. These work on any trades
DataFrame — run the pipeline first, then compute metrics on `result.trades`.
Every result is in the instrument's own units: prices in the quote currency
and sizes in BTC on the Bitstamp sample. The functions read `tick_size` and
`lot_size` from the trades' `attrs`, so the run's integer ticks and lots and a
display-unit frame give the same numbers.

## VPIN

```python
from ob_analytics import Pipeline, compute_vpin, sample_csv_path
from ob_analytics.visualization import plot, save_figure, prepare

result = Pipeline().run(sample_csv_path())

# About 60 buckets: enough to fill the 50-bucket trailing window.
# The trades store whole lots; lot_size turns their total into BTC.
total = result.trades["volume"].sum() * result.config.lot_size
vpin = compute_vpin(result.trades, bucket_volume=total / 60)
fig = plot("vpin", **prepare.vpin(vpin, threshold=0.7))
save_figure(fig, "vpin.png")
```

`bucket_volume` is in the instrument's own units: BTC on the Bitstamp sample,
shares or contracts elsewhere. A pipeline result stores sizes as whole lots
(the size is `lots × lot_size`), and `compute_vpin` reads the lot size from
the trades' `attrs`, so `bucket_volume=0.5` means 0.5 BTC on `result.trades`
and on [`display_result(result).trades`](../api/gallery.md#ob_analytics.visualization.gallery.display_result) alike.
Trades with integer sizes and no `attrs["lot_size"]` (a frame you built
yourself) raise `ConfigError`: set the lot size on the frame, or pass float
sizes. [Short captures](#short-captures) explains the 60. `compute_vpin`
raises `ConfigError` when the bucket would cut the trades into more than
[`MAX_VOLUME_BUCKETS`](../api/trade_sign.md#ob_analytics.trade_sign.MAX_VOLUME_BUCKETS)
(one million) buckets.

Leave `bucket_volume` out to pick it from the trades with the common rule,
average daily volume ÷ 50. `vpin_bucket_volume` applies the same rule on its
own. A session shorter than a day is scaled up to a day at the rate it traded.
When the market does not trade around the clock and the capture falls inside
one session, pass `trading_day="6.5h"` (or your venue's session length). Keep
the 24-hour default for a capture that runs over several days, since the span
it divides by then includes the closed hours:

```python
from ob_analytics import compute_vpin, vpin_bucket_volume

vpin = compute_vpin(result.trades)
print(vpin.attrs["bucket_volume"], vpin.attrs["bucket_volume_rule"])

bucket = vpin_bucket_volume(result.trades, trading_day="6.5h")
vpin = compute_vpin(result.trades, bucket_volume=bucket)
```

The frame's `attrs` also hold `n_buckets` and `diagnostics`. When there are
fewer complete buckets than `n_buckets`, `vpin_avg` is never a full trailing
average, `diagnostics` says so, and `compute_vpin` raises a `UserWarning`. An
empty `diagnostics` tuple means no problem was found.

### Short captures

The default `bucket_volume` (average daily volume ÷ 50) assumes a capture
that runs close to a full day. The bundled sample covers about 30 minutes, so
the default rule scales that up to a day and picks a bucket almost as large
as the whole capture (96% of its volume) — `compute_vpin` fills one bucket and
stops:

```python
from ob_analytics import Pipeline, compute_vpin, sample_csv_path

result = Pipeline().run(sample_csv_path())
vpin = compute_vpin(result.trades)
len(vpin)                      # 1, with a UserWarning explaining why
```

Pass a `bucket_volume` sized to the capture instead of the default. Dividing
by somewhat more than `n_buckets` (buckets rarely divide the volume evenly,
so dividing by exactly `n_buckets` can still fall one bucket short) reliably
fills a whole trailing window:

```python
total = result.trades["volume"].sum() * result.config.lot_size
vpin = compute_vpin(result.trades, bucket_volume=total / 60)
len(vpin)                      # a full series, no warning
```

Or pass `bucket_volume="fit"`, as the gallery does. It keeps the default when
that fills `n_buckets` buckets. Otherwise each bucket is the total volume ÷
(2 × `n_buckets`), and `attrs["bucket_volume_rule"]` says `"total/100"`:

```python
vpin = compute_vpin(result.trades, bucket_volume="fit")
len(vpin)                      # about 100 buckets on the sample, no warning
```

## Kyle's lambda

```python
from ob_analytics import compute_kyle_lambda
from ob_analytics.visualization import plot, save_figure, prepare

kyle = compute_kyle_lambda(result.trades, window="5min")
print(f"λ={kyle.lambda_:.6f}, t={kyle.t_stat:.2f}, R²={kyle.r_squared:.3f}")
print(f"95% interval: [{kyle.ci_low:.6f}, {kyle.ci_high:.6f}]")
if not kyle.significant:
    print("not reliable:", "; ".join(kyle.diagnostics))

fig = plot("kyle_lambda", **prepare.kyle_lambda(kyle))
save_figure(fig, "kyle_lambda.png")
```

λ is in dollars per BTC here: the price move, in the quote currency, per
unit of net order flow. `significant` is `False` when there are fewer than 30 windows
(`KYLE_MIN_WINDOWS`), when `|t|` is below 2 (`KYLE_MIN_T_STAT`), or when the
fit is undefined; `diagnostics` lists which. The interval `ci_low` / `ci_high`
comes from a block bootstrap over the regression windows: 1000 resamples by
default, seeded so the same trades give the same interval. Pass `seed=` to
change it, `ci_level=` for a different coverage, or `n_boot=0` to skip it.

## Order flow imbalance

```python
from ob_analytics import order_flow_imbalance
from ob_analytics.visualization import save_figure

ofi = order_flow_imbalance(result.trades, window="1min")
# The same table, drawn with each window's last trade price in dollars.
fig = result.plot("order_flow_imbalance", window="1min")
save_figure(fig, "ofi.png")
```

## Feeds without a native aggressor side

L3 crypto (Bitstamp) labels each trade's taker side, so `result.trades` has a
real `direction` and the metrics above just work. L2 / aggregated feeds (and
many CCXT sources) don't — so VPIN, Kyle's λ and OFI infer the buy/sell
split with a [trade-sign classifier](../api/trade_sign.md) when `direction`
is absent, and all three take the same `sign_method` and `quotes`:

```python
from ob_analytics import compute_kyle_lambda, compute_vpin, order_flow_imbalance

# A trades frame with no aggressor side, such as an L2 capture's.
l2_trades = result.trades.drop(columns=["direction"])

# About 60 buckets, in BTC: the trades store whole lots.
bucket = l2_trades["volume"].sum() * result.config.lot_size / 60

# No `direction` column → tick rule by default.
vpin = compute_vpin(l2_trades, bucket_volume=bucket)

# Lee–Ready: pass quotes (e.g. the pipeline's depth_summary — it carries
# best_bid_price / best_ask_price).
vpin = compute_vpin(
    l2_trades, bucket_volume=bucket,
    sign_method="lee_ready", quotes=result.depth_summary,
)

# BVC (bulk volume classification) — the VPIN-native estimator; needs no
# per-trade sign at all.
vpin = compute_vpin(l2_trades, bucket_volume=bucket, sign_method="bvc")

ofi = order_flow_imbalance(l2_trades, window="1min", sign_method="tick")
kyle = compute_kyle_lambda(
    l2_trades, window="5min", sign_method="lee_ready", quotes=result.depth_summary
)
```

Lee–Ready reads the quote each trade arrived into: the last quote strictly
before the trade that has both sides and is not crossed. The quote stamped at
the trade's own instant is not used. On a feed whose trades and book events
share one clock, such as LOBSTER, that quote is the book after the trade took
the touch, and reading it flips signs: on a LOBSTER AAPL hour (30 levels),
Lee–Ready agrees with the feed's own side on 95.9% of trades this way, against
85.2% with the quote at the trade's instant.

A native `direction` is always honored as-is (`sign_method=None`, the
default). You can also call
[`classify_trade_sign`](../api/trade_sign.md#ob_analytics.trade_sign.classify_trade_sign)
directly to attach a `direction` column yourself.

## What the flow cost

Toxic flow is flow the liquidity provider loses money to, and the transaction
cost measures put a number on that loss in the currency a taker pays. The
realized spread is what the provider kept after the trade's information
reached the price; when it is negative, the flow was informed — the same
finding VPIN and λ report, measured differently. On a diff feed, check the
effective spread before reading the split: when order messages and trade prints
arrive out of step, the mid a trade is measured from may already have moved
(see [Trust the book](transaction-costs.md#trust-the-book-before-you-trust-the-cost)):

```python
from ob_analytics import transaction_costs, cost_summary

costs = transaction_costs(result.trades, result.depth_summary, horizon="5s")
print(cost_summary(costs))
```

See [Measure transaction costs](transaction-costs.md) for the full
decomposition, and for Amihud illiquidity and Roll's implied spread, which
need no quotes at all.

## Adding your own metric

A flow-toxicity metric can be a plain function over a trades DataFrame. Write
one and call it on `result.trades`:

```python
import numpy as np
import pandas as pd

def signed_volume_skew(trades: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """How lopsided each window's flow was, beyond its net direction."""
    t = trades.copy()
    t["signed"] = t["volume"] * np.where(t["direction"] == "buy", 1.0, -1.0)
    skew = t.set_index("timestamp")["signed"].resample(freq).apply(pd.Series.skew)
    return skew.rename("signed_volume_skew").reset_index()

skew_df = signed_volume_skew(result.trades)
```

To have it run and draw from a result like the built-in measures, wrap it in
a small `Metric` object and register it with `register_metric`. It then works
with `result.metric(name)` and `result.plot(name)`, and gets its own gallery
card. [A new metric](../extending.md#4-a-new-metric) shows how.

## Related

- [Flow Toxicity API](../api/flow_toxicity.md) — parameters and return types
- [Measure transaction costs](transaction-costs.md) — effective spread, price impact, Amihud, Roll
- [Glossary: flow toxicity](../glossary.md#flow-toxicity) — what the metrics mean, with citations
