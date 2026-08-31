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
| **Flow toxicity** *(post-run)* | VPIN, Kyle's lambda, order-flow imbalance computed from `result.trades` |
| **Visualize / export** | Depth heatmaps, event maps, trade charts, flow-toxicity plots, HTML galleries. Matplotlib (default) or Plotly backend. Parquet and LOBSTER round-trip I/O |

---

## Design decisions

- **DataFrames end to end.** Pandas for speed; the column-list constants and
  `validate_*` functions in `schemas.py` document the column contracts.
- **Two API levels** — `Pipeline` for one-line runs; individual classes
  (`BitstampLoader`, `BitstampTradeReader`, etc.) for step-by-step control.
- **Pluggable everything** — any object with the right method signature works;
  no inheritance required (structural typing via `Protocol`).
- **One `Source` shape, file or live.** Every data source states its `level`
  (L2/L3) and `feed_type`, carries typed `settings`, and implements one or both
  capabilities — `OfflineSource` (replay stored files) and `LiveSource`
  (capture a venue). Both register in the one `SOURCES` registry
  (`register_source`), and third-party sources load through the
  `ob_analytics.sources` entry-point group.
- **Per-run parameters live on `RunContext`, not `Source`.** Construction-time
  parameters (typed `settings`, fixed venue config) belong on the source;
  per-session parameters (trading date, write-time level cap) live on
  `RunContext` and are passed to `Pipeline(source=..., ctx=...)`. This keeps
  sources reusable across multiple runs without re-instantiation and avoids
  baking session state into long-lived objects.

---

## Scale envelope

ob-analytics keeps the full event, depth, and trade tables in memory (pandas), so
the working set scales with the event count. Peak RSS grows roughly **linearly at
~1 GiB per 1M events**, with the depth stages (`price_level_volume` →
`depth_metrics`) dominating both memory and time:

| events  | peak RSS  | depth stages |
|---------|----------:|-------------:|
| 314 k   | ~0.43 GiB | ~14 s |
| 628 k   | ~0.73 GiB | ~25 s |
| 942 k   | ~1.02 GiB | ~38 s |
| 1.26 M  | ~1.32 GiB | ~51 s |

*(Measured with [`scripts/bench_scale.py`](https://github.com/mczielinski/ob-analytics/blob/main/scripts/bench_scale.py): the bundled
~314k-event sample tiled to each size, each size run in its own process, peak RSS
via `getrusage`. Slightly conservative — tiling adds some transient overhead.)*

**Comfortable ceiling ≈ 5M events (~5 GiB)** on a typical 16 GB machine — i.e.
session-scale data, a few hours of a single liquid instrument. For larger inputs
(a full NASDAQ MBO day is **10–100M events**, well past this), the recommended
pattern is to **pre-slice by time window** and process each slice independently,
concatenating the per-slice outputs. ob-analytics deliberately ships no
streaming / out-of-core machinery; pre-slicing keeps the in-memory model simple
and predictable (a chunked-run helper could automate it later if real workloads
demand it).

---

## Class diagram

The package combines **protocol-based** components with **source descriptors**
that bundle venue-specific defaults and capabilities.

```mermaid
classDiagram
    class Pipeline {
        +config: PipelineConfig
        +loader: EventLoader
        +trade_source: TradeSource
        +writer: DataWriter | None
        +run(source) PipelineResult
        +from_source(name, **kwargs)$ Pipeline
    }

    class PipelineConfig {
        +price_decimals: int
        +volume_decimals: int
        +price_divisor: int
        +timestamp_unit: str
        +depth_bps: int
        +depth_bins: int
    }

    class Source {
        <<Protocol>>
        +name: str
        +level: Level
        +feed_type: FeedType
        +settings: SourceSettings
    }
    class OfflineSource {
        <<Protocol>>
        +create_loader(config, ctx) EventLoader
        +create_trade_source(config, ctx) TradeSource
        +create_writer(config, ctx) DataWriter | None
        +compute_depth(events, config, source, ctx) tuple | None
        +config_defaults() dict
    }
    class LiveSource {
        <<Protocol>>
        +snapshot(config) AsyncIterator
        +stream(config) AsyncIterator
        +shutdown_synthetic_events() AsyncIterator
    }

    class RunContext {
        +trading_date: object | None
    }

    class BitstampSource
    class LobsterSource

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
        +config: PipelineConfig
        +level: Level
    }

    Pipeline --> PipelineConfig
    Pipeline --> EventLoader
    Pipeline --> TradeSource
    Pipeline --> DataWriter
    Pipeline --> PipelineResult
    Pipeline ..> OfflineSource : source=
    Pipeline ..> RunContext : per-run params
    Source <|-- OfflineSource
    Source <|-- LiveSource
    OfflineSource <|.. BitstampSource
    LiveSource <|.. BitstampSource
    OfflineSource <|.. LobsterSource
```

---

## Data formats

| Source | Level | Entry point | Trades |
|--------|-------|-------------|--------|
| **Bitstamp** (CSV replay + live capture) | L3 | `Pipeline()` (default) · `capture bitstamp` | Companion `trades.csv` next to `orders.csv` (e.g. `scripts/collect_bitstamp_btcusd.py`) |
| **LOBSTER** | L3 | `Pipeline(source=LobsterSource(), ctx=RunContext(trading_date=...))` | Embedded execution rows (types 4/5) in the message file |
| **L2 depth CSV** | L2 | `Pipeline.from_source("depth_csv").run(...)` | Optional companion `trades.csv` (signed via trade-sign classification) |
| **CCXT** (live L2 capture) | L2 | `capture ccxt --exchange <venue>` | Public trade tape (taker side) |
| **cryptofeed** (live capture) | L2 or L3, discovered from the venue | `capture cryptofeed --exchange <venue>` | Public trade tape (taker side) |

The bundled sample under `ob_analytics/_sample_data/` is a modern BTC/USD
capture (`orders.csv` + `trades.csv`).

### L2 vs L3: the resolution axis

A source declares a **`level`** (`Level.L2` / `Level.L3`), orthogonal to
its `FeedType` crossing invariant:

- **L3 (market-by-order)** — the per-order model above. Events carry order IDs,
  so `set_order_types`, `order_aggressiveness`, and queue reconstruction run.
- **L2 (market-by-price)** — price-level feeds (Binance, Kalshi, Polymarket,
  most CCXT sources) publish `[price, quantity]` levels and diffs with **no
  order IDs**. The loader (a `DepthSource`) yields the depth frame directly and
  the pipeline **skips the per-order stages**: `PipelineResult.events` comes
  back empty (schema-valid) and `PipelineResult.level is Level.L2`.
  `DepthMetricsEngine` already consumes an absolute price-level book, so depth /
  spread / trade analytics run unchanged. See the
  [Process L2 feeds](https://github.com/mczielinski/ob-analytics/blob/main/docs/howto/l2-depth.md)
  how-to.

---

## Module map

```
ob_analytics/
├── __init__.py           # Public API surface + source registration + sample_csv_path()
├── _sample_data/         # Bundled Bitstamp sample (orders.csv + trades.csv)
├── pipeline.py           # Pipeline, PipelineResult
├── sources.py            # SOURCES registry: register_source, list_sources, get_source, entry-point discovery
├── config.py             # PipelineConfig, SourceSettings (frozen Pydantic models)
├── protocols.py          # EventLoader, TradeSource, DataWriter, Source, OfflineSource
├── schemas.py            # column constants + validators (validate_events_df, …)
├── exceptions.py         # ObAnalyticsError hierarchy
├── cli.py                # CLI entry point (process, gallery, bitstamp-demo, lobster-demo, capture, sources)
│
├── bitstamp.py           # BitstampLoader, BitstampTradeReader, BitstampWriter, BitstampSource (offline + live)
├── lobster.py            # LobsterLoader, LobsterTradeReader, LobsterWriter, LobsterSource
├── depth_l2.py           # L2DepthLoader, L2TradeReader, DepthCsvWriter, DepthCsvSource (price-level)
├── analytics.py          # order_aggressiveness, trade_impacts, set_order_types, order_book
├── depth.py              # DepthMetricsEngine, price_level_volume, depth_metrics, get_spread
├── data.py               # save_data, load_data, writer registry
├── flow_toxicity.py      # compute_vpin, compute_kyle_lambda, order_flow_imbalance, KyleLambdaResult
├── _utils.py             # Validation, numerics, timestamp conversion helpers
│
├── live/                 # Optional live-capture machinery ([live] / [ccxt] / [cryptofeed] extras)
│   ├── __init__.py       # live-source registration (ccxt, cryptofeed) + re-exports
│   ├── _base.py          # LiveSource protocol, CaptureConfig, CaptureResult, CaptureSink
│   ├── _runner.py        # Generic asyncio driver + FileCaptureSink
│   ├── bitstamp.py       # Bitstamp WebSocket engine (driven by BitstampSource)
│   ├── ccxt_source.py    # CcxtSource, CcxtSettings (any CCXT venue, L2)
│   └── cryptofeed_source.py  # CryptofeedSource, CryptofeedSettings (L2 or native L3)
│
└── visualization/        # Plotting subsystem
    ├── __init__.py       # plot() dispatcher + RENDERERS registry, PlotTheme, save_figure
    ├── gallery.py        # HTML gallery generation
    ├── _data.py          # Shared data prep for plot backends
    ├── _matplotlib.py    # Matplotlib renderers
    └── _plotly.py        # Plotly renderers
```

**Live capture** is optional (install with `pip install "ob-analytics[live]"`,
`[ccxt]`, or `[cryptofeed]`) and writes into the same CSV schema the pipeline reads. Give your
`Source` the live capability -- the three `LiveSource` async-iterator methods
(`snapshot`, `stream`, `shutdown_synthetic_events`) -- and `register_source`
it; a source can add the offline factories too and do both. The runner
(`run_capturer`) handles persistence, raw-frame archival, signal handling, and
`meta.json` finalisation so source authors only write the per-venue parser.
