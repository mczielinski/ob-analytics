---
title: Compute flow toxicity
---

# Compute VPIN, Kyle's λ, and order-flow imbalance

Detect informed trading and measure price impact. These work on any trades
DataFrame — run the pipeline first, then compute metrics on `result.trades`.

## VPIN

```python
from ob_analytics import compute_vpin
from ob_analytics.visualization import plot, save_figure, prepare

vpin = compute_vpin(result.trades, bucket_volume=5.0)
fig = plot("vpin", **prepare.vpin(vpin, threshold=0.7))
save_figure(fig, "vpin.png")
```

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
average, and `diagnostics` says so. An empty tuple means no problem was found.

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

`significant` is `False` when there are fewer than 30 windows
(`KYLE_MIN_WINDOWS`), when `|t|` is below 2 (`KYLE_MIN_T_STAT`), or when the
fit is undefined; `diagnostics` lists which. The interval `ci_low` / `ci_high`
comes from a block bootstrap over the regression windows: 1000 resamples by
default, seeded so the same trades give the same interval. Pass `seed=` to
change it, `ci_level=` for a different coverage, or `n_boot=0` to skip it.

## Order flow imbalance

```python
from ob_analytics import order_flow_imbalance
from ob_analytics.visualization import plot, save_figure, prepare

ofi = order_flow_imbalance(result.trades, window="1min")
fig = plot("order_flow_imbalance", **prepare.ofi(ofi, trades=result.trades))
save_figure(fig, "ofi.png")
```

## Feeds without a native aggressor side

L3 crypto (Bitstamp) labels each trade's taker side, so `result.trades` has a
real `direction` and the metrics above just work. L2 / aggregated feeds (and
many CCXT sources) don't — so VPIN and OFI infer the buy/sell split with a
[trade-sign classifier](../api/trade_sign.md) when `direction` is absent:

```python
from ob_analytics import compute_vpin, order_flow_imbalance

# No `direction` column → tick rule by default.
vpin = compute_vpin(l2_trades, bucket_volume=5.0)

# Lee–Ready: pass quotes (e.g. the pipeline's depth_summary — it carries
# best_bid_price / best_ask_price).
vpin = compute_vpin(
    l2_trades, bucket_volume=5.0,
    sign_method="lee_ready", quotes=result.depth_summary,
)

# BVC (bulk volume classification) — the VPIN-native estimator; needs no
# per-trade sign at all.
vpin = compute_vpin(l2_trades, bucket_volume=5.0, sign_method="bvc")

ofi = order_flow_imbalance(l2_trades, window="1min", sign_method="tick")
```

A native `direction` is always honored as-is (`sign_method=None`, the
default). You can also call
[`classify_trade_sign`](../api/trade_sign.md#ob_analytics.trade_sign.classify_trade_sign)
directly to attach a `direction` column yourself.

## What the flow cost

Toxic flow is flow the liquidity provider loses money to, and the transaction
cost measures put a number on that loss in the currency a taker pays. The
realized spread is what the provider kept after the trade's information
reached the price; when it is negative, the flow was informed — the same
finding VPIN and λ report, measured differently:

```python
from ob_analytics import transaction_costs, cost_summary

costs = transaction_costs(result.trades, result.depth_summary, horizon="5s")
print(cost_summary(costs))
```

See [Measure transaction costs](transaction-costs.md) for the full
decomposition, and for Amihud illiquidity and Roll's implied spread, which
need no quotes at all.

## Adding your own metric

There is no metrics plugin registry — a flow-toxicity metric is just a
function over a trades DataFrame. Write one and call it on `result.trades`:

```python
import numpy as np
import pandas as pd

def signed_volume_skew(trades: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """How lopsided each window's flow was, beyond its net direction."""
    t = trades.copy()
    t["signed"] = t["volume"] * np.where(t["direction"] == "buy", 1.0, -1.0)
    skew = t.set_index("timestamp")["signed"].resample(freq).skew()
    return skew.rename("signed_volume_skew").reset_index()

skew_df = signed_volume_skew(result.trades)
```

To fold a metric into the HTML gallery, wrap it in a panel builder and pass it
via `extra_panels=` — see [Extending ob-analytics](../extending.md) for the
full walkthrough (new data source, export format, plot, metric, or live
capturer).

## Related

- [Flow Toxicity API](../api/flow_toxicity.md) — parameters and return types
- [Measure transaction costs](transaction-costs.md) — effective spread, price impact, Amihud, Roll
- [Glossary: flow toxicity](../glossary.md#flow-toxicity) — what the metrics mean, with citations
