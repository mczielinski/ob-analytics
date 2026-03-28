---
title: Quickstart
---

# Quickstart

This guide covers the main ways to use ob-analytics, from the simplest
one-liner to writing your own exchange adapter.

---

## 1. Bitstamp — Bundled Sample Data

The package ships with ~5 hours of Bitstamp BTC/USD limit order events
(2015-05-01). This is the fastest way to see what the package does.

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
Events:        (50393, 14)
Trades:        (482, 10)
Depth:         (49376, 5)
Depth summary: (49216, 46)
```

### Visualize

```python
from ob_analytics import get_spread, plot_price_levels, plot_trades, save_figure

spread = get_spread(result.depth_summary)

fig = plot_price_levels(result.depth, spread, volume_scale=1e-8, col_bias=0.1)
save_figure(fig, "price_levels.png")

fig = plot_trades(result.trades)
save_figure(fig, "trades.png")
```

### Subplot composition

All plot functions accept an `ax` parameter for multi-panel figures:

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

Use individual functions when you need access to intermediate results:

```python
from ob_analytics import (
    load_event_data, event_match, match_trades, set_order_types,
    get_zombie_ids, price_level_volume, depth_metrics,
    order_aggressiveness, get_spread,
)

events = load_event_data("inst/extdata/orders.csv")
events = event_match(events)                        # Needleman-Wunsch fill pairing
trades = match_trades(events)
events = set_order_types(events, trades)            # market, flashed-limit, pacman, …
zombie_ids = get_zombie_ids(events, trades)
events = events[~events["id"].isin(zombie_ids)]
depth = price_level_volume(events)
depth_summary = depth_metrics(depth)                # best bid/ask, BPS bins
events = order_aggressiveness(events, depth_summary)
```

---

## 2. Custom Bitstamp-Format CSV

Point the pipeline at your own data and adjust `PipelineConfig` for
the instrument:

```python
from ob_analytics import Pipeline, PipelineConfig

config = PipelineConfig(
    price_decimals=2,
    match_cutoff_ms=5000,
    price_jump_threshold=10.0,
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
        price_decimals=4,
        match_cutoff_ms=500,
        price_jump_threshold=0.01,
        depth_bps=5,
        depth_bins=100,
    )
    ```

=== "High-Price Equity"

    ```python
    config = PipelineConfig(
        price_decimals=2,
        match_cutoff_ms=100,
        price_jump_threshold=50.0,
    )
    ```

---

## 3. Custom Exchange Adapter

For non-Bitstamp data, implement the `EventLoader` protocol. Your class
needs a `load` method returning a DataFrame with the columns documented
in `OrderEvent`.

### Example: generic CSV loader

```python
from pathlib import Path
import pandas as pd
from ob_analytics import Pipeline, PipelineConfig

class GenericCsvLoader:
    """Load events from a CSV with different column names."""

    COLUMN_MAP = {
        "order_id": "id",
        "event_time": "timestamp",
        "server_time": "exchange_timestamp",
        "side": "direction",
        "type": "action",
    }
    ACTION_MAP = {"new": "created", "partial_fill": "changed",
                  "fill": "changed", "cancel": "deleted"}
    DIRECTION_MAP = {"buy": "bid", "sell": "ask"}

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def load(self, source: str | Path) -> pd.DataFrame:
        df = pd.read_csv(source)
        df = df.rename(columns=self.COLUMN_MAP)
        df["action"] = df["action"].map(self.ACTION_MAP)
        df["direction"] = df["direction"].map(self.DIRECTION_MAP)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["exchange_timestamp"] = pd.to_datetime(df["exchange_timestamp"])
        # Compute fill deltas and event IDs (see BitstampLoader for reference)
        ...
        return df

pipeline = Pipeline(loader=GenericCsvLoader())
result = pipeline.run("my_exchange_data.csv")
```

### Example: cryptofeed adapter (conceptual)

```python
from pathlib import Path
import pandas as pd
from ob_analytics import PipelineConfig

class CryptofeedLoader:
    """Load events from a cryptofeed L3 order book log.

    Size of 0 means deletion. This adapter tracks state to infer
    actions and compute fills.
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def load(self, source: str | Path) -> pd.DataFrame:
        raw = pd.read_parquet(source)
        known_orders: dict[str, float] = {}
        rows = []

        for _, row in raw.iterrows():
            oid = row["order_id"]
            size = row["size"]
            prev_size = known_orders.get(oid)

            if prev_size is None:
                action, fill = "created", 0.0
            elif size == 0:
                action, fill = "deleted", prev_size
            else:
                action, fill = "changed", prev_size - size

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
    This is a conceptual example. A production adapter would handle
    out-of-order messages, reconnection gaps, and exchange-specific quirks.
    The key point: **any object with a `load` method returning the right
    DataFrame works** — no subclassing required.

### LOBSTER (built-in)

[LOBSTER](https://lobsterdata.com/) is supported out of the box with
`LobsterLoader`, `LobsterMatcher`, `LobsterTradeInferrer`, and optional
orderbook-backed depth via `LobsterFormat.compute_depth`.

```python
from ob_analytics import LobsterFormat, Pipeline

pipeline = Pipeline(format=LobsterFormat(trading_date="2012-06-21"))
result = pipeline.run("/path/to/extracted_lobster_folder")
```

Equivalent shorthand:

```python
result = Pipeline.from_format("lobster", trading_date="2012-06-21").run(
    "/path/to/extracted_lobster_folder"
)
```

Free sample files can be fetched with `download_sample()`. The included
`scripts/lobster_demo.py` downloads a sample, runs the pipeline, saves
Parquet, verifies round-trip I/O, and builds an HTML plot gallery:

```bash
uv run python scripts/lobster_demo.py --ticker AAPL
```

!!! note
    When message files contain cross trades (type 6) or trading halts
    (type 7), filtered rows may not align one-to-one with orderbook rows;
    the implementation logs a warning and uses the minimum consistent length.

---

## 4. Custom Matching Algorithm

Replace the default Needleman-Wunsch matcher:

```python
import pandas as pd
from ob_analytics import Pipeline, PipelineConfig

class SimpleTimeMatcher:
    """Match fills by closest timestamp (toy example)."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def match(self, events: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()
        events["matching_event"] = float("nan")
        # Your matching logic — must populate 'matching_event'
        ...
        return events

pipeline = Pipeline(matcher=SimpleTimeMatcher())
result = pipeline.run("inst/extdata/orders.csv")
```

---

## 5. Theming and Saving

```python
from ob_analytics import PlotTheme, set_plot_theme, plot_trades, save_figure

set_plot_theme(PlotTheme(
    style="whitegrid",
    context="talk",
    font_scale=1.2,
    rc={"axes.facecolor": "#f8f9fa", "figure.facecolor": "#ffffff"},
))

fig = plot_trades(result.trades)
save_figure(fig, "trades_hires.png", dpi=300)
```

---

## 6. Serialization

```python
from ob_analytics import save_data, load_data

save_data(
    {
        "events": result.events,
        "trades": result.trades,
        "depth": result.depth,
        "depth_summary": result.depth_summary,
    },
    "output/my_analysis",
)

data = load_data("output/my_analysis")
```

---

## 7. Flow Toxicity

Detect informed trading and measure price impact. These work on any trades
DataFrame.

### VPIN

```python
from ob_analytics import compute_vpin, plot_vpin, save_figure

vpin = compute_vpin(result.trades, bucket_volume=5.0)
fig = plot_vpin(vpin, threshold=0.7)
save_figure(fig, "vpin.png")
```

### Kyle's Lambda

```python
from ob_analytics import compute_kyle_lambda, plot_kyle_lambda

kyle = compute_kyle_lambda(result.trades, window="5min")
print(f"λ={kyle.lambda_:.6f}, t={kyle.t_stat:.2f}, R²={kyle.r_squared:.3f}")

fig = plot_kyle_lambda(kyle)
save_figure(fig, "kyle_lambda.png")
```

### Order Flow Imbalance

```python
from ob_analytics import order_flow_imbalance, plot_order_flow_imbalance

ofi = order_flow_imbalance(result.trades, window="1min")
fig = plot_order_flow_imbalance(ofi, trades=result.trades)
save_figure(fig, "ofi.png")
```

### Pipeline integration (opt-in)

```python
from ob_analytics import Pipeline, PipelineConfig

config = PipelineConfig(vpin_bucket_volume=5.0)
result = Pipeline(config=config).run("orders.csv")
# result.vpin and result.ofi are automatically populated
```

---

## 8. Interactive Visualizations (Plotly)

All plot functions accept `backend="plotly"` for interactive figures with
zoom, pan, and hover tooltips.

```bash
pip install ob-analytics[interactive]
```

```python
from ob_analytics import Pipeline, plot_price_levels, get_spread

result = Pipeline().run("inst/extdata/orders.csv")
spread = get_spread(result.depth_summary)

fig = plot_price_levels(
    result.depth, spread, volume_scale=1e-8, col_bias=0.1,
    price_from=232, backend="plotly",
)
fig.show()
fig.write_html("depth.html")
```

Custom backends can be registered:

```python
from ob_analytics import register_plot_backend
register_plot_backend("bokeh", "my_package._bokeh_backend")
```

---

## 9. Demo Scripts

Both demo scripts run the full pipeline, save Parquet output, verify round-trip
I/O, and generate an HTML plot gallery.

**Bitstamp:**

```bash
uv run python scripts/bitstamp_demo.py
uv run python scripts/bitstamp_demo.py --input path/to/orders.csv
uv run python scripts/bitstamp_demo.py --output ~/Desktop/bitstamp_gallery
```

**LOBSTER:**

```bash
uv run python scripts/lobster_demo.py
uv run python scripts/lobster_demo.py --ticker MSFT
uv run python scripts/lobster_demo.py --output ~/Desktop/lobster_gallery
```

---

## 10. Command-Line Interface

Installing the package registers the `ob-analytics` command. All subcommands
accept `-v` / `--verbose` for debug-level logging.

### Process a data source

```bash
ob-analytics process orders.csv -o results/
ob-analytics process data/ --format lobster --trading-date 2012-06-21
ob-analytics process orders.csv -o results/ --gallery
```

### Generate a gallery from saved Parquet

```bash
ob-analytics gallery results/parquet/ -o my_gallery/
ob-analytics gallery results/parquet/ --volume-scale 1e-8 --title "My Analysis"
```

### Run demos

```bash
ob-analytics bitstamp-demo --input orders.csv -o demo_out/
ob-analytics lobster-demo --ticker AAPL -o demo_out/
```

| Subcommand | Description |
|------------|-------------|
| `process` | Run the pipeline on a data source, save Parquet (optional `--gallery`) |
| `gallery` | Generate an HTML plot gallery from saved Parquet data |
| `bitstamp-demo` | Run the Bitstamp demo (pipeline + gallery) |
| `lobster-demo` | Download LOBSTER sample data and run the demo (pipeline + gallery) |

---

## Next steps

- [API Reference](api/pipeline.md) for detailed documentation
- [CLI Reference](api/cli.md) for CLI API docs
- [Changelog](changelog.md) for recent changes
- [GitHub README](https://github.com/mczielinski/ob-analytics) for
  architecture diagrams
