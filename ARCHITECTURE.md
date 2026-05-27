# Architecture

## Pipeline stages

ob-analytics turns **order event streams** and **authoritative trade records**
into structured analytics:

| Stage | What happens |
|-------|-------------|
| **Load & normalize** | Parse Bitstamp CSV or LOBSTER message file into a uniform event DataFrame |
| **Build trades** | Bitstamp: read companion `trades.csv` (live capture). LOBSTER: extract type 4/5 executions from the events frame via `LobsterTradeReader` |
| **Classify orders** | Label each order as *market*, *resting-limit*, *flashed-limit*, *market-limit*, or *unknown* |
| **Depth & metrics** | Track price-level volume, best bid/ask, spread, and liquidity in configurable BPS bins. LOBSTER can use the official orderbook file for ground-truth depth |
| **Flow toxicity** *(optional)* | VPIN, Kyle's lambda, order-flow imbalance |
| **Visualize / export** | Depth heatmaps, event maps, trade charts, flow-toxicity plots, HTML galleries. Matplotlib (default) or Plotly backend. Parquet and LOBSTER round-trip I/O |

---

## Design decisions

- **DataFrames internally; Pydantic at boundaries.** Pandas for speed;
  `OrderEvent`, `Trade`, etc. document column contracts.
- **Two API levels** — `Pipeline` for one-line runs; individual classes
  (`BitstampLoader`, `BitstampTradeReader`, etc.) for step-by-step control.
- **Pluggable everything** — any object with the right method signature works;
  no inheritance required (structural typing via `Protocol`).
- **Per-run parameters live on `RunContext`, not `Format`.** Construction-time
  parameters (schema choice, fixed venue config) belong on the `Format` ctor;
  per-session parameters (trading date, write-time level cap) live on
  `RunContext` and are passed to `Pipeline(format=..., ctx=...)`. This keeps
  formats reusable across multiple runs without re-instantiation and avoids
  baking session state into long-lived objects.

---

## Class diagram

The package combines **protocol-based** components with **format descriptors**
that bundle venue-specific defaults.

```mermaid
classDiagram
    class Pipeline {
        +config: PipelineConfig
        +loader: EventLoader
        +trade_source: TradeSource
        +writer: DataWriter | None
        +run(source) PipelineResult
        +from_format(name, **kwargs)$ Pipeline
    }

    class PipelineConfig {
        +price_decimals: int
        +volume_decimals: int
        +price_divisor: int
        +timestamp_unit: str
        +depth_bps: int
        +depth_bins: int
        +vpin_bucket_volume: float | None
    }

    class Format {
        <<abstract>>
        +create_loader(config, ctx) EventLoader
        +create_trade_source(config, ctx) TradeSource
        +create_writer(config, ctx) DataWriter | None
        +compute_depth(events, config, source, ctx) tuple | None
        +collect_extras(loader, events, source, ctx) dict
        +config_defaults() dict
    }

    class RunContext {
        +trading_date: object | None
        +extras: dict
    }

    class BitstampFormat
    class LobsterFormat

    class EventLoader {
        <<Protocol>>
        +load(source) DataFrame
    }
    class TradeSource {
        <<Protocol>>
        +load(events, source) DataFrame
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
        +extras: dict[str, DataFrame]
        +metrics: dict[str, DataFrame]
    }

    class ToxicityMetric {
        <<Protocol>>
        +name: str
        +requires: tuple[str, ...]
        +primary_column: str
        +compute(result, config) DataFrame
    }

    class Vpin
    class KyleLambda
    class Ofi

    Pipeline --> PipelineConfig
    Pipeline --> EventLoader
    Pipeline --> TradeSource
    Pipeline --> DataWriter
    Pipeline --> PipelineResult
    Pipeline ..> Format : optional
    Pipeline ..> RunContext : per-run params
    Pipeline ..> ToxicityMetric : metrics=[...]
    Format <|-- BitstampFormat
    Format <|-- LobsterFormat
    ToxicityMetric <|.. Vpin
    ToxicityMetric <|.. KyleLambda
    ToxicityMetric <|.. Ofi
```

---

## Data formats

| Format | Entry point | Trades |
|--------|-------------|--------|
| **Bitstamp CSV** | `Pipeline()` (default) | Companion `trades.csv` next to `orders.csv` (e.g. `scripts/collect_bitstamp_btcusd.py`) |
| **LOBSTER** | `Pipeline(format=LobsterFormat(), ctx=RunContext(trading_date=...))` | Embedded execution rows (types 4/5) in the message file |

The bundled sample under `ob_analytics/_sample_data/` is a modern BTC/USD
capture (`orders.csv` + `trades.csv`).

---

## Module map

```
ob_analytics/
├── __init__.py           # Public API surface + format registration + sample_csv_path()
├── _sample_data/         # Bundled Bitstamp sample (orders.csv + trades.csv)
├── pipeline.py           # Pipeline, PipelineResult, register_format
├── config.py             # PipelineConfig (frozen Pydantic model)
├── protocols.py          # EventLoader, TradeSource, DataWriter, Format
├── models.py             # OrderEvent, Trade, DepthLevel, OrderBookSnapshot, KyleLambdaResult
├── exceptions.py         # ObAnalyticsError hierarchy
├── cli.py                # CLI entry point (process, gallery, bitstamp-demo, lobster-demo, capture)
│
├── bitstamp.py           # BitstampLoader, BitstampTradeReader, BitstampWriter, BitstampFormat
├── lobster.py            # LobsterLoader, LobsterTradeReader, LobsterWriter, LobsterFormat
├── analytics.py          # order_aggressiveness, trade_impacts, set_order_types, order_book
├── depth.py              # DepthMetricsEngine, price_level_volume, depth_metrics, get_spread
├── data.py               # save_data, load_data, writer registry
├── flow_toxicity.py      # compute_vpin, compute_kyle_lambda, order_flow_imbalance
├── _utils.py             # Validation, numerics, timestamp conversion helpers
│
├── metrics/              # Pluggable ToxicityMetric protocol + built-ins
│   ├── __init__.py       # registry: register_metric, list_metrics
│   ├── _base.py          # ToxicityMetric protocol
│   ├── vpin.py           # Vpin (wraps compute_vpin)
│   ├── kyle_lambda.py    # KyleLambda (wraps compute_kyle_lambda)
│   └── ofi.py            # Ofi (wraps order_flow_imbalance)
│
├── live/                 # Optional live order-book capture ([live] extra)
│   ├── __init__.py       # registry: register_capturer, list_capturers, get_capturer
│   ├── _base.py          # LiveCapturer protocol, CaptureConfig, CaptureResult, CaptureSink
│   ├── _runner.py        # Generic asyncio driver + FileCaptureSink
│   └── bitstamp.py       # BitstampCapturer (built-in)
│
└── visualization/        # Plotting subsystem
    ├── __init__.py       # plot_* dispatchers, PlotTheme, save_figure, backend registry
    ├── gallery.py        # HTML gallery generation
    ├── _data.py          # Shared data prep for plot backends
    ├── _matplotlib.py    # Matplotlib renderers
    └── _plotly.py        # Plotly renderers
```

**`ob_analytics.live`** is an optional async sub-package (install with
`pip install "ob-analytics[live]"`) that captures live exchange data into
the same CSV schema the pipeline reads. Implement the `LiveCapturer`
protocol -- three async-iterator methods (`snapshot`, `stream`,
`shutdown_synthetic_events`) -- and call `register_capturer` to add a
new venue; the registry pattern mirrors `Format` and `register_writer`.
The runner (`run_capturer`) handles persistence, raw-frame archival,
signal handling, and `meta.json` finalisation so capturer authors only
write the per-venue parser.
