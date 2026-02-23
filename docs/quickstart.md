---
title: Quickstart
---

# Quickstart Guide

This guide walks through the three main ways to use ob-analytics, from the
simplest one-liner to writing your own exchange adapter.

---

## 1. Bitstamp — Using the Bundled Sample Data

The package ships with ~5 hours of Bitstamp BTC/USD limit order events from
2015-05-01.  This is the fastest way to see what the package does.

### One-line pipeline

```python
from ob_analytics import Pipeline

result = Pipeline().run("inst/extdata/orders.csv")

print(f"Events:        {result.events.shape}")
print(f"Trades:        {result.trades.shape}")
print(f"Depth:         {result.depth.shape}")
print(f"Depth summary: {result.depth_summary.shape}")
```

```
Events:        (50393, 13)
Trades:        (482, 10)
Depth:         (49376, 4)
Depth summary: (49376, 45)
```

### Visualize

```python
from ob_analytics import (
    get_spread, plot_price_levels, plot_trades, save_figure,
)
import matplotlib.pyplot as plt

spread = get_spread(result.depth_summary)

# Depth heatmap with midprice overlay
fig = plot_price_levels(
    result.depth, spread,
    volume_scale=1e-8,
    col_bias=0.1,
)
save_figure(fig, "price_levels.png")

# Trade price chart
fig = plot_trades(result.trades)
save_figure(fig, "trades.png")
```

### Subplot composition

All plot functions accept an `ax` parameter, so you can compose them into
multi-panel figures:

```python
import matplotlib.pyplot as plt
from ob_analytics import plot_trades, plot_events_histogram
import pandas as pd

t3 = pd.Timestamp("2015-05-01 03:00:00")
t4 = pd.Timestamp("2015-05-01 04:00:00")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

plot_trades(result.trades, start_time=t3, end_time=t4, ax=ax1)
ax1.set_title("Trades 03:00–04:00")

hist_data = result.events[["timestamp", "direction", "price", "volume"]].copy()
hist_data["volume"] *= 1e-8
plot_events_histogram(hist_data, val="price", bw=0.25, ax=ax2)
ax2.set_title("Price distribution")

fig.tight_layout()
fig.savefig("combined.png", dpi=150)
```

### Step-by-step (full control)

If you need access to intermediate results at each stage:

```python
from ob_analytics import (
    load_event_data, event_match, match_trades, set_order_types,
    get_zombie_ids, price_level_volume, depth_metrics,
    order_aggressiveness, get_spread,
)

# 1. Load raw CSV
events = load_event_data("inst/extdata/orders.csv")

# 2. Match bid/ask fills (Needleman-Wunsch for ambiguous cases)
events = event_match(events)

# 3. Infer trades
trades = match_trades(events)

# 4. Classify: market, flashed-limit, pacman, resting-limit, etc.
events = set_order_types(events, trades)

# 5. Remove zombie orders
zombie_ids = get_zombie_ids(events, trades)
events = events[~events["id"].isin(zombie_ids)]

# 6. Price-level volume
depth = price_level_volume(events)

# 7. Depth metrics (best bid/ask, BPS bins)
depth_summary = depth_metrics(depth)

# 8. Order aggressiveness (distance from best price)
events = order_aggressiveness(events, depth_summary)
```

---

## 2. Custom Bitstamp-Format CSV

If you have your own Bitstamp-format data, simply point the pipeline at it.
Adjust `PipelineConfig` to match your instrument:

```python
from ob_analytics import Pipeline, PipelineConfig

config = PipelineConfig(
    price_decimals=2,           # BTC/USD: 2 decimal places
    match_cutoff_ms=5000,       # 5 second matching window
    price_jump_threshold=10.0,  # $10 price jump heuristic
)

result = Pipeline(config=config).run("my_bitstamp_data.csv")
```

### Configuration for different instruments

=== "BTC/USD (default)"

    ```python
    config = PipelineConfig(
        price_decimals=2,
        match_cutoff_ms=5000,
        price_jump_threshold=10.0,
    )
    ```

=== "ETH/USD"

    ```python
    config = PipelineConfig(
        price_decimals=2,
        match_cutoff_ms=3000,
        price_jump_threshold=5.0,
    )
    ```

=== "FX (EUR/USD)"

    ```python
    config = PipelineConfig(
        price_decimals=4,           # Pips: 4 decimal places
        match_cutoff_ms=500,        # Much tighter matching
        price_jump_threshold=0.01,  # 1 pip jump threshold
        depth_bps=5,                # Finer 5-bps bins
        depth_bins=100,             # 100 bins × 5 bps = 500 bps
    )
    ```

=== "High-Price Equity"

    ```python
    config = PipelineConfig(
        price_decimals=2,
        match_cutoff_ms=100,        # Sub-second matching
        price_jump_threshold=50.0,  # $50 for high-price stocks
    )
    ```

---

## 3. Custom Exchange Adapter

For non-Bitstamp data, implement the `EventLoader` protocol. Your class just
needs a `load` method that returns a DataFrame with the expected columns.

### Example: Generic CSV loader

```python
from pathlib import Path
import pandas as pd
from ob_analytics import Pipeline, PipelineConfig

class GenericCsvLoader:
    """Load events from a generic CSV with different column names."""

    COLUMN_MAP = {
        "order_id": "id",
        "event_time": "timestamp",
        "server_time": "exchange_timestamp",
        "side": "direction",
        "type": "action",
    }

    ACTION_MAP = {
        "new": "created",
        "partial_fill": "changed",
        "fill": "changed",
        "cancel": "deleted",
    }

    DIRECTION_MAP = {
        "buy": "bid",
        "sell": "ask",
    }

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def load(self, source: str | Path) -> pd.DataFrame:
        df = pd.read_csv(source)

        # Rename columns to match expected schema
        df = df.rename(columns=self.COLUMN_MAP)

        # Map action and direction values
        df["action"] = df["action"].map(self.ACTION_MAP)
        df["direction"] = df["direction"].map(self.DIRECTION_MAP)

        # Convert timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["exchange_timestamp"] = pd.to_datetime(df["exchange_timestamp"])

        # Compute fill deltas and event IDs
        # (your logic here — see BitstampLoader source for reference)
        ...

        return df


# Use with the pipeline
pipeline = Pipeline(loader=GenericCsvLoader())
result = pipeline.run("my_exchange_data.csv")
```

### Example: Cryptofeed adapter (conceptual)

[cryptofeed](https://github.com/bmoscon/cryptofeed) is the dominant Python
library for real-time crypto market data (40+ exchanges). Here's what a
`CryptofeedLoader` would look like:

```python
from collections import defaultdict
from pathlib import Path
import pandas as pd
from ob_analytics import PipelineConfig

class CryptofeedLoader:
    """Load events from a cryptofeed L3 order book log.

    Cryptofeed L3 deltas provide (order_id, price, size) tuples.
    Size of 0 means the order was deleted. This adapter maintains
    state to infer actions and compute fills.
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def load(self, source: str | Path) -> pd.DataFrame:
        raw = pd.read_parquet(source)  # or however you stored the feed

        # Track known orders to infer action type
        known_orders: dict[str, float] = {}
        rows = []

        for _, row in raw.iterrows():
            oid = row["order_id"]
            size = row["size"]
            prev_size = known_orders.get(oid)

            if prev_size is None:
                action = "created"
                fill = 0.0
            elif size == 0:
                action = "deleted"
                fill = prev_size  # fully consumed
            else:
                action = "changed"
                fill = prev_size - size  # partial fill

            if size > 0:
                known_orders[oid] = size
            elif oid in known_orders:
                del known_orders[oid]

            rows.append({
                "id": oid,
                "timestamp": row["receipt_timestamp"],
                "exchange_timestamp": row["exchange_timestamp"],
                "price": row["price"],
                "volume": size,
                "action": action,
                "direction": "bid" if row["side"] == "BID" else "ask",
                "fill": fill,
            })

        events = pd.DataFrame(rows)
        events["event_id"] = range(1, len(events) + 1)
        events["original_number"] = events["event_id"]
        return events
```

!!! note
    This is a conceptual example. A production adapter would handle edge cases
    like out-of-order messages, reconnection gaps, and exchange-specific
    quirks.  The key point is that **any object with a `load` method returning
    the right DataFrame works** — no subclassing required.

### Example: LOBSTER format adapter (conceptual)

[LOBSTER](https://lobsterdata.com/) is the academic standard for limit order
book research, used widely in market microstructure papers.

```python
from pathlib import Path
import pandas as pd
from ob_analytics import PipelineConfig

class LobsterLoader:
    """Load events from LOBSTER message file format.

    LOBSTER provides two files per trading day:
    - Message file: Time, EventType, OrderID, Size, Price, Direction
    - Orderbook file: snapshots (not used here)

    EventType mapping:
        1 = new limit order → "created"
        2 = partial cancellation → "changed"
        3 = full deletion → "deleted"
        4 = visible execution → "changed"
        5 = hidden execution → "changed"
    """

    EVENT_TYPE_MAP = {
        1: "created",
        2: "changed",
        3: "deleted",
        4: "changed",
        5: "changed",
    }

    def __init__(
        self,
        config: PipelineConfig | None = None,
        trading_date: str = "2023-01-03",
    ):
        self.config = config or PipelineConfig()
        self.trading_date = pd.Timestamp(trading_date)

    def load(self, source: str | Path) -> pd.DataFrame:
        df = pd.read_csv(
            source,
            header=None,
            names=["time", "event_type", "id", "volume", "price", "direction"],
        )

        # Convert LOBSTER price (integer, dollars × 10000)
        df["price"] = df["price"] / 10_000

        # Convert time (seconds after midnight) to datetime
        df["timestamp"] = self.trading_date + pd.to_timedelta(df["time"], unit="s")
        df["exchange_timestamp"] = df["timestamp"]

        # Map event types and directions
        df["action"] = df["event_type"].map(self.EVENT_TYPE_MAP)
        df["direction"] = df["direction"].map({1: "bid", -1: "ask"})

        # Drop unknown event types (6=cross, 7=halt)
        df = df.dropna(subset=["action"])

        # Compute fills (volume consumed between events for the same order)
        df = df.sort_values("timestamp", kind="stable")
        df["fill"] = 0.0
        df["event_id"] = range(1, len(df) + 1)
        df["original_number"] = df["event_id"]

        return df
```

!!! tip
    LOBSTER already distinguishes cancellations (type 2) from executions
    (type 4). You can preserve this in `OrderEvent.raw_event_type` and
    potentially skip the Needleman-Wunsch matching step entirely for
    LOBSTER data, since trade attribution is already known.

---

## 4. Custom Matching Algorithm

Replace the default Needleman-Wunsch matcher with your own logic:

```python
import pandas as pd
from ob_analytics import Pipeline, PipelineConfig

class SimpleTimeMatcher:
    """Match fills by closest timestamp only (no NW alignment)."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def match(self, events: pd.DataFrame) -> pd.DataFrame:
        # Your matching logic here — must add 'matching_event' column
        events["matching_event"] = float("nan")

        bids = events[
            (events["direction"] == "bid") & (events["fill"] > 0)
        ].sort_values("timestamp")
        asks = events[
            (events["direction"] == "ask") & (events["fill"] > 0)
        ].sort_values("timestamp")

        # Simple: pair each bid fill with the closest ask fill
        # (this is a toy example — real matching is more complex)
        ...

        return events


pipeline = Pipeline(matcher=SimpleTimeMatcher())
result = pipeline.run("inst/extdata/orders.csv")
```

---

## 5. Theming and Saving

### Change the default theme

```python
from ob_analytics import PlotTheme, set_plot_theme

# Light theme for presentations
set_plot_theme(PlotTheme(
    style="whitegrid",
    context="talk",
    font_scale=1.2,
    rc={"axes.facecolor": "#f8f9fa", "figure.facecolor": "#ffffff"},
))
```

### Save with custom DPI

```python
from ob_analytics import plot_trades, save_figure

fig = plot_trades(result.trades)
save_figure(fig, "trades_hires.png", dpi=300)
```

---

## 6. Serialization

### Save pipeline results to Parquet (recommended)

```python
from ob_analytics import save_data, load_data

# Save all DataFrames as Parquet files in a directory
save_data(
    {
        "events": result.events,
        "trades": result.trades,
        "depth": result.depth,
        "depth_summary": result.depth_summary,
    },
    "output/my_analysis",
)

# Load them back
data = load_data("output/my_analysis")
events = data["events"]
trades = data["trades"]
```

---

## 7. Flow Toxicity Analysis

Detect informed trading and measure price impact using three
microstructure metrics.  These work on any trades DataFrame.

### VPIN — Informed Trading Detection

```python
from ob_analytics import compute_vpin, plot_vpin, save_figure

# Partition trade volume into equal-sized buckets
vpin = compute_vpin(result.trades, bucket_volume=5.0)

# Trailing average > 0.7 signals toxic (informed) flow
print(vpin[["timestamp_end", "vpin", "vpin_avg"]].tail())

fig = plot_vpin(vpin, threshold=0.7)
save_figure(fig, "vpin.png")
```

### Kyle's Lambda — Price Impact

```python
from ob_analytics import compute_kyle_lambda, plot_kyle_lambda

kyle = compute_kyle_lambda(result.trades, window="5min")
print(f"λ = {kyle.lambda_:.6f}, t = {kyle.t_stat:.2f}, R² = {kyle.r_squared:.3f}")

fig = plot_kyle_lambda(kyle)
save_figure(fig, "kyle_lambda.png")
```

### Order Flow Imbalance

```python
from ob_analytics import order_flow_imbalance, plot_order_flow_imbalance

ofi = order_flow_imbalance(result.trades, window="1min")
print(ofi[["timestamp", "ofi"]].head())

fig = plot_order_flow_imbalance(ofi, trades=result.trades)
save_figure(fig, "ofi.png")
```

### Pipeline Integration (opt-in)

```python
from ob_analytics import Pipeline, PipelineConfig

config = PipelineConfig(vpin_bucket_volume=5.0)
result = Pipeline(config=config).run("orders.csv")

# Automatically computed
print(result.vpin.shape)
print(result.ofi.shape)
```

---

## Next Steps

- Browse the [API Reference](api/pipeline.md) for detailed documentation
- Check the [Changelog](changelog.md) for the latest changes
- Read the [README](https://github.com/mczielinski/ob-analytics) for
  architecture diagrams and financial concept glossary
