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

## Kyle's lambda

```python
from ob_analytics import compute_kyle_lambda
from ob_analytics.visualization import plot, save_figure, prepare

kyle = compute_kyle_lambda(result.trades, window="5min")
print(f"λ={kyle.lambda_:.6f}, t={kyle.t_stat:.2f}, R²={kyle.r_squared:.3f}")

fig = plot("kyle_lambda", **prepare.kyle_lambda(kyle))
save_figure(fig, "kyle_lambda.png")
```

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

## Adding your own metric

There is no metrics plugin registry — a flow-toxicity metric is just a
function over a trades DataFrame. Write one and call it on `result.trades`:

```python
import pandas as pd

def amihud(trades: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """Amihud (2002) illiquidity = |return| / volume."""
    t = trades.set_index("timestamp").sort_index()
    ret = t["price"].pct_change().abs()
    illiq = (ret / t["volume"]).resample(freq).mean()
    return illiq.rename("amihud").reset_index()

amihud_df = amihud(result.trades)
```

To fold a metric into the HTML gallery, wrap it in a panel builder and pass it
via `extra_panels=` — see [Extending ob-analytics](../extending.md) for the
full walkthrough (new data source, export format, plot, metric, or live
capturer).

## Related

- [Flow Toxicity API](../api/flow_toxicity.md) — parameters and return types
- [Glossary: flow toxicity](../glossary.md#flow-toxicity) — what the metrics mean, with citations
