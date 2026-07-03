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
