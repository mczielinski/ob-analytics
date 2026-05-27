---
title: Quickstart
---

# Quickstart

This page covers everyday usage. Skim the [Walkthrough](#walkthrough) to get
something running on the bundled sample data, then jump to the recipe you
need.

| If you want to… | Go to |
|-----------------|-------|
| Run the pipeline on bundled data and plot it | [Walkthrough](#walkthrough) |
| Use your own Bitstamp CSV or another instrument | [Working with your data](#working-with-your-data) |
| Process LOBSTER message + orderbook files | [LOBSTER](#lobster) |
| Plug in your own loader, trade source, or format | [Custom components](#custom-components) |
| Theme plots, save artefacts, or use Plotly | [Customising the output](#customising-the-output) |
| Compute VPIN / Kyle's λ / OFI | [Flow toxicity](#flow-toxicity) |
| Run from a shell | [Command-line interface](#cli) |
| Capture live order-book data from an exchange | [Live capture](#live-capture) |

---

## Walkthrough

The package ships with a bundled Bitstamp BTC/USD capture: `orders.csv` and
sibling `trades.csv` under `ob_analytics/_sample_data/`. Access the orders
path via `sample_csv_path()` (the pipeline loads `trades.csv` from the same
directory).

### One-line pipeline

```python
from ob_analytics import Pipeline, sample_csv_path

result = Pipeline().run(sample_csv_path())

print(f"Events:        {result.events.shape}")
print(f"Trades:        {result.trades.shape}")
print(f"Depth:         {result.depth.shape}")
print(f"Depth summary: {result.depth_summary.shape}")
```

```
Events:        (~314000, …)   # shape depends on bundled capture
Trades:        (~280, …)
Depth:         (…, 5)
Depth summary: (…, 46)
```

### Visualise

```python
from ob_analytics import get_spread, plot_price_levels, plot_trades, save_figure

spread = get_spread(result.depth_summary)

fig = plot_price_levels(result.depth, spread, volume_scale=1e-8, col_bias=0.1)
save_figure(fig, "price_levels.png")

fig = plot_trades(result.trades)
save_figure(fig, "trades.png")
```

All plot functions accept an `ax` parameter for multi-panel figures:

```python
import matplotlib.pyplot as plt
import pandas as pd
from ob_analytics import plot_trades, plot_events_histogram

# Pick a 10-minute window inside the sample (it spans 30 minutes).
t_start = result.trades["timestamp"].min()
t3 = t_start + pd.Timedelta(minutes=10)
t4 = t3 + pd.Timedelta(minutes=10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

plot_trades(result.trades, start_time=t3, end_time=t4, ax=ax1)
ax1.set_title(f"Trades {t3:%H:%M}–{t4:%H:%M}")

hist_data = result.events[["timestamp", "direction", "price", "volume"]].copy()
hist_data["volume"] *= 1e-8
plot_events_histogram(hist_data, val="price", bw=0.25, ax=ax2)
ax2.set_title("Price distribution")

fig.tight_layout()
fig.savefig("combined.png", dpi=150)
```

### Step-by-step (full control)

Use individual classes when you need access to intermediate results:

```python
from pathlib import Path
from ob_analytics import (
    BitstampLoader,
    BitstampTradeReader,
    set_order_types,
    price_level_volume,
    depth_metrics,
    order_aggressiveness,
    sample_csv_path,
)

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

### Point-in-time order book snapshot

Once you have processed events, reconstruct the full bid/ask ladder at any
timestamp:

```python
import pandas as pd
from ob_analytics import order_book

# Snapshot ten minutes into the run (events.timestamp is timezone-naive UTC).
tp = events["timestamp"].iloc[0] + pd.Timedelta(minutes=10)
snapshot = order_book(events, tp=tp, max_levels=5)
print(snapshot["timestamp"])
print(snapshot["bids"].head())
print(snapshot["asks"].head())
```

Use `bps_range` to filter to a basis-point band around the mid-price,
or `max_levels` to cap the number of price levels returned.

---

## Working with your data

### Custom Bitstamp-format CSV

Place `orders.csv` and `trades.csv` in the same directory (see
`scripts/collect_bitstamp_btcusd.py` for the expected `trades.csv` schema).
Point the pipeline at the **orders** path; it resolves sibling `trades.csv`
automatically.

```python
from ob_analytics import Pipeline, PipelineConfig

config = PipelineConfig(price_decimals=2, volume_decimals=8)
result = Pipeline(config=config).run("my_run/orders.csv")
```

#### Configuration presets

=== "BTC/USD (default)"

    ```python
    config = PipelineConfig(price_decimals=2, volume_decimals=8)
    ```

=== "ETH/USD"

    ```python
    config = PipelineConfig(price_decimals=2, volume_decimals=6)
    ```

=== "FX (EUR/USD)"

    ```python
    config = PipelineConfig(
        price_decimals=4,
        volume_decimals=2,
        depth_bps=5,
        depth_bins=100,
    )
    ```

=== "High-Price Equity"

    ```python
    config = PipelineConfig(price_decimals=2, volume_decimals=0)
    ```

### LOBSTER

[LOBSTER](https://lobsterdata.com/) message and orderbook files are
supported out of the box via `LobsterLoader`, `LobsterTradeReader`,
`LobsterWriter`, and `LobsterFormat`. Depth is read from the official
orderbook file (ground-truth) when present.

```python
from ob_analytics import LobsterFormat, Pipeline
from ob_analytics.protocols import RunContext

fmt = LobsterFormat()
ctx = RunContext(trading_date="2012-06-21")
result = Pipeline(format=fmt, ctx=ctx).run(
    "/path/to/extracted_lobster_folder"
)

# equivalent shorthand via the format registry:
result = Pipeline.from_format(
    "lobster", ctx=RunContext(trading_date="2012-06-21"),
).run("/path/to/extracted_lobster_folder")
```

#### Per-format extras

Some formats expose auxiliary event tables that don't fit the universal
events schema — LOBSTER trading halts, cross trades, and hidden
executions, for example. The pipeline attaches these to
`PipelineResult.extras`:

```python
result = Pipeline(format=LobsterFormat(), ctx=RunContext(trading_date="...")).run(path)
result.extras["trading_halts"]
result.extras["cross_trades"]
result.extras["hidden_executions"]
```

Bitstamp runs produce no extras (`result.extras == {}`). Plot helpers
`plot_trading_halts(result=result)` and
`plot_hidden_executions(result=result)` read from extras automatically.

!!! note
    When message files contain cross trades (event type 6) or trading halts
    (event type 7), filtered rows may not align one-to-one with orderbook
    rows; the implementation logs a warning and uses the minimum consistent
    length.

---

## Custom components

Every pipeline stage is a [Protocol](api/protocols.md). Implement the right
method signature on any object and pass it in — no inheritance required.

### Custom event loader

For non-Bitstamp, non-LOBSTER data, implement the `EventLoader` protocol.
Your `load` method must return a DataFrame with the columns subsequent
stages consume:

| Column | Type | Notes |
|--------|------|-------|
| `id` | int / str | Exchange-assigned order identifier |
| `event_id` | int | Sequential, 1-based, unique per event |
| `original_number` | int | Original input row number (stable event order) |
| `timestamp` | datetime64 | Local receive time |
| `exchange_timestamp` | datetime64 | Server time stamp |
| `price` | float | Order price |
| `volume` | float | Remaining size |
| `action` | category | `created` / `changed` / `deleted` |
| `direction` | category | `bid` / `ask` |
| `fill` | float | Volume executed by this event (0 for non-fills) |

[`OrderEvent`](api/models.md#ob_analytics.models.OrderEvent) is the
Pydantic-level contract; the columns above are what the pipeline reads.

#### Generic CSV loader

```python
from pathlib import Path
import pandas as pd
from ob_analytics import Pipeline, PipelineConfig
from ob_analytics.bitstamp import BitstampTradeReader

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

result = Pipeline(
    loader=GenericCsvLoader(),
    trade_source=BitstampTradeReader(),
).run("my_exchange_data/orders.csv")  # requires sibling trades.csv
```

#### Cryptofeed L3 adapter (conceptual)

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
    A production adapter would handle out-of-order messages, reconnection
    gaps, and exchange-specific quirks. The key point: any object with a
    `load` method returning the right DataFrame works — no subclassing
    required.

### Custom trade source

Implement `TradeSource` (a `load(events, source) -> DataFrame` method) when
trades come from an API, a database, or a non-CSV layout:

```python
import pandas as pd
from ob_analytics import Pipeline, PipelineConfig, sample_csv_path

class ApiTradeSource:
    def load(self, events: pd.DataFrame, source: object) -> pd.DataFrame:
        # Build the canonical trades DataFrame (timestamp, price, volume,
        # direction, maker_event_id, taker_event_id, maker, taker, …)
        raise NotImplementedError

result = Pipeline(
    config=PipelineConfig(),
    trade_source=ApiTradeSource(),
).run(sample_csv_path())
```

Bundle defaults in a `Format` subclass — see [Protocols](api/protocols.md).

---

## Customising the output

### Themes and saving

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

### Serialisation

Pipeline outputs are dict-of-DataFrames; `save_data` writes one Parquet
file per key, `load_data` reads them back.

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

For LOBSTER round-trip output (back to message + orderbook CSVs), pass
`fmt="lobster"` and a `RunContext` so the registered writer factory can
pick up `trading_date`:

```python
from ob_analytics import save_data
from ob_analytics.protocols import RunContext

save_data(
    data, "round_trip/", fmt="lobster",
    config=config, ctx=RunContext(trading_date="2012-06-21"),
)
```

### Plotly (interactive)

All plot functions accept `backend="plotly"` for interactive figures with
zoom, pan, and hover tooltips. Plotly is an optional dependency:

```bash
pip install -e ".[interactive]"
```

```python
from ob_analytics import Pipeline, plot_price_levels, get_spread, sample_csv_path

result = Pipeline().run(sample_csv_path())
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

## Flow toxicity

Detect informed trading and measure price impact. These work on any trades
DataFrame.

### VPIN

```python
from ob_analytics import compute_vpin, plot_vpin, save_figure

vpin = compute_vpin(result.trades, bucket_volume=5.0)
fig = plot_vpin(vpin, threshold=0.7)
save_figure(fig, "vpin.png")
```

### Kyle's lambda

```python
from ob_analytics import compute_kyle_lambda, plot_kyle_lambda

kyle = compute_kyle_lambda(result.trades, window="5min")
print(f"λ={kyle.lambda_:.6f}, t={kyle.t_stat:.2f}, R²={kyle.r_squared:.3f}")

fig = plot_kyle_lambda(kyle)
save_figure(fig, "kyle_lambda.png")
```

### Order flow imbalance

```python
from ob_analytics import order_flow_imbalance, plot_order_flow_imbalance

ofi = order_flow_imbalance(result.trades, window="1min")
fig = plot_order_flow_imbalance(ofi, trades=result.trades)
save_figure(fig, "ofi.png")
```

### Pipeline integration (opt-in)

Set `vpin_bucket_volume` on the config and the pipeline will compute
VPIN and OFI as part of the standard run:

```python
from ob_analytics import Pipeline, PipelineConfig

config = PipelineConfig(vpin_bucket_volume=5.0)
result = Pipeline(config=config).run("orders.csv")
# result.vpin and result.ofi are automatically populated
```

### Pluggable metrics

`Pipeline` accepts a `metrics=` keyword that takes any sequence of
`ToxicityMetric` implementations. The computed outputs land in
`result.metrics: dict[str, DataFrame]`:

```python
from ob_analytics import Pipeline, Vpin, Ofi, KyleLambda

result = Pipeline(
    metrics=[Vpin(bucket_volume=1_000), Ofi(window="1min"), KyleLambda()],
).run("orders.csv")

result.metrics["vpin"]         # DataFrame
result.metrics["kyle_lambda"]  # DataFrame
```

To plug in your own metric, implement the protocol:

```python
from dataclasses import dataclass
from ob_analytics import ToxicityMetric, register_metric

@dataclass(frozen=True)
class Amihud:
    """Amihud (2002) illiquidity = |return| / volume."""
    freq: str = "1min"
    name: str = "amihud"
    requires: tuple[str, ...] = ("trades",)
    primary_column: str = "amihud"

    def compute(self, result, config):
        trades = result.trades.set_index("timestamp").sort_index()
        ret = trades["price"].pct_change().abs()
        vol = trades["volume"]
        amihud = (ret / vol).resample(self.freq).mean()
        return amihud.reset_index().rename(columns={0: "amihud"})

register_metric("amihud", Amihud)

result = Pipeline(metrics=[Amihud()]).run("orders.csv")
result.metrics["amihud"]
```

The gallery automatically iterates over `result.metrics`. Built-in
metrics get their dedicated plotters (`plot_vpin`, etc.); user-defined
metrics are available on `result.metrics` for ad-hoc plotting.

---

## Live capture {#live-capture}

ob-analytics ships a small framework for capturing live order-book data
straight into the format the pipeline reads. Install the optional
``[live]`` extra (pulls in ``websockets``) and use the ``capture`` CLI verb:

```bash
pip install "ob-analytics[live]"

ob-analytics capture bitstamp --pair btcusd --minutes 10 --out /tmp/cap
ob-analytics process /tmp/cap/orders.csv --gallery --output /tmp/cap_out
```

Each capture run produces a self-contained directory:

| File | Contents |
|------|----------|
| `orders.csv` | BitstampLoader-compatible event log (`created` / `changed` / `deleted`) |
| `trades.csv` | Venue-reported trades (informational; pipeline infers fills itself) |
| `raw.jsonl` | Every raw WebSocket frame (omit with `--no-raw`) |
| `meta.json` | Run metadata: start/end, counts, per-capturer diagnostics |

The Bitstamp capturer also pulls a REST order-book snapshot at startup
(emitting synthetic `created` events for every resting order) and emits
synthetic `deleted` events at shutdown so every order id in `orders.csv`
has a complete `created -> ... -> deleted` lifecycle.

### Adding a new venue

Implement the `LiveCapturer` protocol -- three async-iterator methods --
and register your class:

```python
from ob_analytics.live import LiveCapturer, register_capturer

class CoinbaseCapturer(LiveCapturer):
    name = "coinbase"

    async def snapshot(self, config):
        # yield synthetic "created" events from a REST snapshot
        ...

    async def stream(self, config):
        # yield (kind, event, raw_frame) tuples for each live message
        ...

    async def shutdown_synthetic_events(self):
        # yield "deleted" events for everything still resting
        ...

register_capturer("coinbase", CoinbaseCapturer)
```

That's enough to make `ob-analytics capture coinbase` work. Persistence,
raw-frame archival, signal handling, and `meta.json` all live in the
generic runner -- you only write the per-venue parser.

---

## Command-line interface {#cli}

Installing the package registers the `ob-analytics` command. All
subcommands accept `-v` / `--verbose` for debug-level logging.

| Subcommand | Description |
|------------|-------------|
| `process` | Run the pipeline on a data source, save Parquet (optional `--gallery`) |
| `gallery` | Generate an HTML plot gallery from saved Parquet data |
| `bitstamp-demo` | Run the Bitstamp demo (pipeline + gallery) |
| `lobster-demo` | Run the LOBSTER demo on local data (pipeline + gallery) |
| `capture` | Live-capture market data from a registered venue (`[live]` extra) |

```bash
# Process a data source
ob-analytics process orders.csv -o results/
ob-analytics process data/ --format lobster --trading-date 2012-06-21
ob-analytics process orders.csv -o results/ --gallery

# Build a gallery from saved Parquet
ob-analytics gallery results/parquet/ -o my_gallery/
ob-analytics gallery results/parquet/ --volume-scale 1e-8 --title "My Analysis"

# End-to-end demos (pipeline + gallery)
ob-analytics bitstamp-demo --input orders.csv -o demo_out/
ob-analytics lobster-demo /path/to/lobster_data --trading-date 2012-06-21 -o demo_out/
```

The `bitstamp-demo` and `lobster-demo` subcommands are equivalent to
running `scripts/bitstamp_demo.py` / `scripts/lobster_demo.py` from a
clone — both run the pipeline, save Parquet, verify round-trip I/O, and
build an HTML plot gallery.

---

## Next steps

- [Architecture](architecture.md) — pipeline stages, design decisions, class diagram, module map
- [Glossary](glossary.md) — market-microstructure terms used throughout
- [API Reference](api/pipeline.md) — detailed module documentation
- [CLI Reference](api/cli.md) — argparse-level CLI docs
- [Changelog](changelog.md) — recent changes
