---
title: Step-by-step control
---

# Run the pipeline step by step

`Pipeline().run()` is one line, but every stage is an ordinary class you can
call yourself — useful when you need intermediate results, a point-in-time
book snapshot, or full control over a figure.

## Step-by-step (full control)

Use individual classes when you need access to intermediate results:

```python
from pathlib import Path
from ob_analytics import sample_csv_path
from ob_analytics.analytics import order_aggressiveness, set_order_types
from ob_analytics.bitstamp import BitstampLoader, BitstampTradeReader
from ob_analytics.depth import depth_metrics, price_level_volume

orders_path = sample_csv_path()
run_dir = Path(orders_path).parent

loader = BitstampLoader()
events = loader.load(orders_path)

reader = BitstampTradeReader()
trades = reader.load(events, run_dir)

events = set_order_types(events, trades)             # market, flashed-limit, market-limit, …
depth = price_level_volume(events)
depth_summary = depth_metrics(depth)                 # best bid/ask, BPS bins
events = order_aggressiveness(events, depth_summary)
```

## Point-in-time order book snapshot

Once you have processed events, reconstruct the full bid/ask ladder at any
timestamp:

```python
import pandas as pd
from ob_analytics.analytics import order_book

# Snapshot ten minutes into the run (events.timestamp is timezone-naive UTC).
tp = events["timestamp"].iloc[0] + pd.Timedelta(minutes=10)
snapshot = order_book(events, tp=tp, max_levels=5)
print(snapshot["timestamp"])
print(snapshot["bids"].head())
print(snapshot["asks"].head())
```

Use `bps_range` to filter to a basis-point band around the mid-price,
or `max_levels` to cap the number of price levels returned.

## Compose figures with `plot()` and `prepare`

For full control over a figure, call the `plot()` dispatcher directly:
prepare the payload with the matching helper in the `prepare` namespace,
then render it. `plot()` also takes an `ax=` for multi-panel figures. Note
that **comparable** concepts (those with both an L2 and L3 face, e.g.
`trade_tape`) require a `level=`.

```python
import matplotlib.pyplot as plt
import pandas as pd
from ob_analytics.visualization import plot, prepare

# Pick a 10-minute window inside the sample (it spans 30 minutes).
t_start = result.trades["timestamp"].min()
t3 = t_start + pd.Timedelta(minutes=10)
t4 = t3 + pd.Timedelta(minutes=10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

plot(
    "trade_tape",
    level="L2",
    ax=ax1,
    **prepare.trades(result.trades, start_time=t3, end_time=t4),
)
ax1.set_title(f"Trades {t3:%H:%M}–{t4:%H:%M}")

hist_data = result.events[["timestamp", "direction", "price", "volume"]].copy()
# Drop sentinel-priced events (market orders / deletions) before binning.
q01, q99 = hist_data["price"].quantile([0.01, 0.99])
hist_data = hist_data[hist_data["price"].between(q01, q99)]
bw = max(0.01, round((q99 - q01) / 100, 2))
plot("events_histogram", ax=ax2, **prepare.events_histogram(hist_data, val="price", bw=bw))
ax2.set_title("Price distribution")

fig.tight_layout()
fig.savefig("combined.png", dpi=150)
```

## Related

- [Visualization API](../api/visualization.md) — `plot`, `prepare`, concepts and levels
- [Analytics API](../api/analytics.md) — `order_book`, `set_order_types`, `order_aggressiveness`
- [Plug in custom components](custom-components.md) — swap any of these stages for your own
