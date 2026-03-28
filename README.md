# ob-analytics

[![License](http://img.shields.io/badge/license-GPL%20%28%3E=%202%29-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-2.0.html)
[![Tests](https://img.shields.io/badge/tests-207%20passed-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)

**Limit order book analytics and visualization for Python.**

Reconstruct trades from raw exchange events, classify order types, compute
depth metrics, and visualize market microstructure — from Bitstamp-style CSVs
or [LOBSTER](https://lobsterdata.com/) message and orderbook files.

> Ported from the R
> [obAnalytics](https://cran.r-project.org/package=obAnalytics) CRAN package.
> Now a standalone Python package with a pipeline API, pluggable formats,
> flow-toxicity metrics, and Matplotlib/Plotly backends.

<p align="center">
  <img src="./assets/ob-analytics-price-levels.png" alt="Price levels depth heatmap (Bitstamp sample)" width="700">
</p>

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Architecture](#architecture)
- [Data Formats](#data-formats)
- [Configuration](#configuration)
- [Visualization](#visualization)
- [Extending the Package](#extending-the-package)
- [Testing](#testing)
- [Documentation](#documentation)
- [License](#license)

---

## Overview

Many exchanges publish only raw **order event streams** (order placed, changed,
cancelled) — not consolidated trade tapes. ob-analytics turns these streams
into structured analytics:

| Stage | What happens |
|-------|-------------|
| **Load & normalize** | Parse Bitstamp CSV or LOBSTER message file into a uniform event DataFrame |
| **Match fills** | Pair simultaneous bid/ask fills — Needleman–Wunsch alignment for Bitstamp; pass-through for LOBSTER (single-sided executions) |
| **Infer trades** | Build trade records with maker/taker attribution |
| **Classify orders** | Label each order as *market*, *resting-limit*, *flashed-limit*, *pacman*, *market-limit*, or *unknown* |
| **Depth & metrics** | Track price-level volume, best bid/ask, spread, and liquidity in configurable BPS bins. LOBSTER can use the official orderbook file for ground-truth depth |
| **Flow toxicity** *(optional)* | VPIN, Kyle's lambda, order-flow imbalance |
| **Visualize / export** | Depth heatmaps, event maps, trade charts, flow-toxicity plots, HTML galleries. Matplotlib (default) or Plotly backend. Parquet and LOBSTER round-trip I/O |

---

## Installation

```bash
pip install git+https://github.com/mczielinski/ob-analytics.git
```

With [uv](https://github.com/astral-sh/uv):

```bash
uv add git+https://github.com/mczielinski/ob-analytics.git
```

From a local clone:

```bash
git clone https://github.com/mczielinski/ob-analytics.git
cd ob-analytics
pip install -e .
```

Interactive Plotly figures (optional):

```bash
pip install "ob-analytics[interactive]"
```

**Requires** Python 3.11+. Core dependencies: NumPy, pandas, matplotlib,
seaborn, pydantic, pyarrow, loguru.

---

## Quickstart

### Pipeline (one line)

```python
from ob_analytics import Pipeline

result = Pipeline().run("inst/extdata/orders.csv")

result.events       # enriched events with order types and aggressiveness
result.trades       # inferred trades with maker/taker attribution
result.depth        # price-level volume time series
result.depth_summary  # best bid/ask, BPS bins, spread
result.metadata     # provenance (source, format, row counts, config snapshot)
```

### LOBSTER

```python
from ob_analytics import Pipeline, LobsterFormat

result = Pipeline(format=LobsterFormat(trading_date="2012-06-21")).run(
    "/path/to/lobster_data"
)
```

### Step-by-step (full control)

```python
from ob_analytics import (
    load_event_data, event_match, match_trades, set_order_types,
    get_zombie_ids, price_level_volume, depth_metrics, order_aggressiveness,
    get_spread, plot_price_levels, save_figure,
)

events = load_event_data("inst/extdata/orders.csv")
events = event_match(events)
trades = match_trades(events)
events = set_order_types(events, trades)
zombie_ids = get_zombie_ids(events, trades)
events = events[~events["id"].isin(zombie_ids)]
depth = price_level_volume(events)
depth_summary = depth_metrics(depth)
events = order_aggressiveness(events, depth_summary)

spread = get_spread(depth_summary)
fig = plot_price_levels(depth, spread, trades, volume_scale=1e-8)
save_figure(fig, "price_levels.png")
```

All plot functions accept an `ax` parameter for subplot composition.

---

## Architecture

The package combines **protocol-based** components with **format descriptors**
that bundle venue-specific defaults.

```mermaid
classDiagram
    class Pipeline {
        +config: PipelineConfig
        +loader: EventLoader
        +matcher: MatchingEngine
        +trade_inferrer: TradeInferrer
        +writer: DataWriter | None
        +run(source) PipelineResult
        +from_format(name, **kwargs)$ Pipeline
    }

    class PipelineConfig {
        +price_decimals: int
        +volume_decimals: int
        +price_divisor: int
        +timestamp_unit: str
        +match_cutoff_ms: int
        +price_jump_threshold: float
        +depth_bps: int
        +depth_bins: int
        +skip_zombie_detection: bool
        +vpin_bucket_volume: float | None
    }

    class Format {
        <<abstract>>
        +create_loader()
        +create_matcher()
        +create_trade_inferrer()
        +create_writer()
        +compute_depth()
        +config_defaults()
    }

    class BitstampFormat
    class LobsterFormat

    class EventLoader {
        <<Protocol>>
        +load(source) DataFrame
    }
    class MatchingEngine {
        <<Protocol>>
        +match(events) DataFrame
    }
    class TradeInferrer {
        <<Protocol>>
        +infer_trades(events) DataFrame
    }
    class DataWriter {
        <<Protocol>>
        +write(data, dest, **kwargs)
    }

    class PipelineResult {
        +events: DataFrame
        +trades: DataFrame
        +depth: DataFrame
        +depth_summary: DataFrame
        +vpin: DataFrame | None
        +ofi: DataFrame | None
        +metadata: dict
    }

    Pipeline --> PipelineConfig
    Pipeline --> EventLoader
    Pipeline --> MatchingEngine
    Pipeline --> TradeInferrer
    Pipeline --> DataWriter
    Pipeline --> PipelineResult
    Pipeline ..> Format : optional
    Format <|-- BitstampFormat
    Format <|-- LobsterFormat
```

**Design decisions:**

- **DataFrames internally; Pydantic at boundaries.** Pandas for speed;
  `OrderEvent`, `Trade`, etc. document column contracts.
- **Backward-compatible functions** — `load_event_data`, `event_match`, etc.
  remain available alongside `Pipeline`.
- **Pluggable everything** — any object with the right method signature works;
  no inheritance required (structural typing via `Protocol`).

### Module map

```
ob_analytics/
├── __init__.py           # Public API surface + format registration
├── pipeline.py           # Pipeline, PipelineResult, register_format
├── config.py             # PipelineConfig (frozen Pydantic model)
├── protocols.py          # EventLoader, MatchingEngine, TradeInferrer, DataWriter, Format
├── models.py             # OrderEvent, Trade, DepthLevel, OrderBookSnapshot, KyleLambdaResult
├── exceptions.py         # ObAnalyticsError hierarchy
│
├── event_processing.py   # BitstampLoader, BitstampWriter, BitstampFormat
├── lobster.py            # LobsterLoader/Matcher/TradeInferrer/Writer/Format, download_sample
├── matching_engine.py    # NeedlemanWunschMatcher, event_match
├── trades.py             # DefaultTradeInferrer, match_trades, trade_impacts
├── order_types.py        # set_order_types (market, flashed-limit, pacman, …)
├── order_book_reconstruction.py  # Point-in-time book snapshots
├── depth.py              # DepthMetricsEngine, price_level_volume, depth_metrics, get_spread
├── data.py               # save_data, load_data, process_data, get_zombie_ids
├── flow_toxicity.py      # compute_vpin, compute_kyle_lambda, order_flow_imbalance
│
├── visualisation.py      # plot_* dispatchers, PlotTheme, save_figure, backend registry
├── gallery.py            # HTML gallery generation
├── _matplotlib.py        # Matplotlib renderers
├── _plotly.py            # Plotly renderers
├── _chart_data.py        # Shared data prep for plot backends
├── _needleman_wunsch.py  # Sequence alignment internals
└── _utils.py             # Validation, numerics helpers
```

---

## Data Formats

| Format | Entry point | Matcher | Notes |
|--------|-------------|---------|-------|
| **Bitstamp CSV** | `Pipeline()` (default) | Needleman–Wunsch | Single CSV with order events |
| **LOBSTER** | `Pipeline(format=LobsterFormat(trading_date=...))` | Pass-through | Message file + optional orderbook; round-trip I/O via `LobsterWriter` |

LOBSTER demo (downloads sample data, runs pipeline, writes Parquet and a plot
gallery):

```bash
uv run python scripts/lobster_demo.py          # AAPL by default
uv run python scripts/lobster_demo.py --ticker MSFT
```

---

## Configuration

```python
from ob_analytics import PipelineConfig, Pipeline

config = PipelineConfig(
    price_decimals=4,        # FX: 4 decimal places
    match_cutoff_ms=500,     # tighter matching window
    depth_bps=5,             # 5-bps bins
    depth_bins=100,          # 100 bins = 500 bps total
    vpin_bucket_volume=5.0,  # enable VPIN + OFI
)
result = Pipeline(config=config).run("my_data.csv")
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `price_decimals` | `2` | Decimal places for price rounding |
| `volume_decimals` | `8` | Decimal places for volume rounding |
| `price_divisor` | `1` | Raw price divisor (`10000` for LOBSTER integer prices) |
| `timestamp_unit` | `"ms"` | Integer timestamp unit in source data |
| `match_cutoff_ms` | `5000` | Fill-pairing window for Needleman–Wunsch matcher |
| `price_jump_threshold` | `10.0` | Maker/taker swap heuristic threshold |
| `depth_bps` / `depth_bins` | `25` / `20` | BPS bin width and count |
| `zombie_offset_seconds` | `60` | Warm-up seconds before depth metrics |
| `skip_zombie_detection` | `False` | Skip zombie detection (set `True` for LOBSTER) |
| `vpin_bucket_volume` | `None` | Volume per VPIN bucket; `None` = skip flow toxicity |

---

## Visualization

Plot functions return `matplotlib.figure.Figure` by default. Pass
`backend="plotly"` for interactive figures (requires
`pip install ob-analytics[interactive]`). Functions never call `plt.show()`.

| Function | Description |
|----------|-------------|
| `plot_price_levels` | Depth heatmap with midprice and trade overlay |
| `plot_trades` | Trade price step plot |
| `plot_event_map` | Order placements/cancellations in price–time space |
| `plot_volume_map` | Flashed-order volume |
| `plot_current_depth` | Book snapshot (cumulative volume vs price) |
| `plot_volume_percentiles` | Stacked liquidity in BPS bins over time |
| `plot_events_histogram` | Price or volume distribution by side |
| `plot_time_series` | Generic step plot |
| `plot_vpin` | VPIN with toxicity threshold |
| `plot_order_flow_imbalance` | OFI bars with optional price overlay |
| `plot_kyle_lambda` | Signed volume vs price-change scatter with regression |
| `plot_hidden_executions` | Hidden execution volume (LOBSTER type 5) |
| `plot_trading_halts` | Trade price with shaded halt periods (LOBSTER type 7) |

Theming: `PlotTheme`, `set_plot_theme`, `get_plot_theme`.
Saving: `save_figure(fig, path, dpi=...)`.
Custom backends: `register_plot_backend(name, module_path)`.

---

## Extending the Package

**Custom loader** — implement `load(source) -> DataFrame` with columns matching
`OrderEvent`:

```python
pipeline = Pipeline(loader=MyLoader())
```

**Custom format** — subclass `Format` to bundle loader, matcher, inferrer,
writer, and config defaults, then register:

```python
from ob_analytics import register_format
register_format("myvenue", MyFormat)
result = Pipeline.from_format("myvenue", **kwargs).run(source)
```

**Custom matcher or trade inferrer** — pass directly to `Pipeline`; explicit
arguments override format defaults:

```python
Pipeline(format=LobsterFormat(...), matcher=MyMatcher())
```

---

## Testing

```bash
uv run pytest tests/ -v
uv run pytest tests/ --cov=ob_analytics
```

200+ pytest tests covering loaders, matching, trades, depth, visualization,
LOBSTER paths, and pipeline integration.

---

## Documentation

Zensical site with API reference generated from docstrings:

```bash
zensical serve      # local preview at http://localhost:8000
zensical build      # static site in site/
```

---

## License

GPL (>= 2)
