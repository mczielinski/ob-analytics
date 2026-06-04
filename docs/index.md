---
title: Home
---

# ob-analytics

**Limit order book analytics and visualization for Python.**

Load order events, attach authoritative trades (Bitstamp `trades.csv` or
LOBSTER executions), classify order types, compute depth metrics, and
visualize market microstructure — from Bitstamp-style CSVs or
[LOBSTER](https://lobsterdata.com/) message and orderbook files.

## What it does

| Stage | Description |
|-------|-------------|
| **Load & normalize** | Parse Bitstamp CSV or LOBSTER message file into a uniform event DataFrame |
| **Build trades** | Bitstamp: companion `trades.csv`. LOBSTER: execution rows (types 4/5) in the message file |
| **Classify orders** | Label as market, resting-limit, flashed-limit, market-limit, or unknown |
| **Depth & metrics** | Price-level volume, best bid/ask, spread, liquidity in BPS bins |
| **Flow toxicity** *(post-run)* | VPIN, Kyle's lambda, order-flow imbalance from `result.trades` |
| **Visualize / export** | Depth heatmaps, event maps, trade charts, galleries; Matplotlib or Plotly; Parquet and LOBSTER round-trip I/O |

## Pipeline

```mermaid
flowchart LR
    subgraph in["Inputs"]
        CSV[Bitstamp orders + trades]
        LOB[LOBSTER msg + orderbook]
    end

    subgraph pipeline["Pipeline"]
        direction TB
        L[Load & normalize]
        T[Build trades]
        C[Classify orders]
        D[Depth metrics]
        L --> T --> C --> D
    end

    subgraph out["Outputs"]
        EV[Events · Trades]
        DP[Depth · Summary]
        VZ[Plots · Parquet · LOBSTER files]
    end

    CSV --> L
    LOB --> L
    pipeline --> EV
    pipeline --> DP
    EV & DP --> VZ
```

All processing stages are pluggable via [Protocol interfaces](api/protocols.md).
See the [Architecture](architecture.md) page for the full class diagram, design
decisions, and module map.

## Quick example

### Bitstamp (bundled sample)

```python
from ob_analytics import Pipeline, sample_csv_path

result = Pipeline().run(sample_csv_path())
print(result.events.shape, result.trades.shape)
```

### LOBSTER

```python
from ob_analytics import LobsterFormat, Pipeline, RunContext

result = Pipeline(
    format=LobsterFormat(),
    ctx=RunContext(trading_date="2012-06-21"),
).run("/path/to/lobster_data")
```

## CLI

A command-line interface is included for common workflows:

```bash
ob-analytics process orders.csv -o results/
ob-analytics gallery results/parquet/ -o my_gallery/
ob-analytics bitstamp-demo --input orders.csv
ob-analytics lobster-demo /path/to/lobster_data --trading-date 2012-06-21
```

See the [Quickstart CLI section](quickstart.md#cli) for full details.

## Next steps

- **[Quickstart](quickstart.md)** — full walkthrough with visualization, configuration, custom loaders, and CLI
- **[Architecture](architecture.md)** — pipeline stages, design decisions, class diagram, module map
- **[Glossary](glossary.md)** — market-microstructure terms used throughout
- **[API Reference](api/pipeline.md)** — detailed documentation for every module
- **[CLI Reference](api/cli.md)** — command-line interface API docs
- **[Changelog](changelog.md)** — recent changes
