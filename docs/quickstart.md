---
title: Getting started
---

# Getting started

Install the package, run the pipeline on the bundled sample, and render your
first plot — about ten minutes. When something here sparks a question, the
[Tutorial](tutorial/index.md) teaches the concepts; the how-to guides
(linked [below](#where-next)) cover every recipe.

!!! note "These docs follow `main`"
    This site is built from the `main` branch on GitHub. It can describe
    features that are not yet in the newest release on PyPI. To get
    everything these pages describe, install from source:
    `pip install git+https://github.com/mczielinski/ob-analytics.git`.
    The [changelog](changelog.md) lists what changed since the last release.

## Install

```bash
pip install ob-analytics
```

Optional extras:

```bash
pip install "ob-analytics[interactive]"   # Plotly figures
pip install "ob-analytics[bokeh]"         # Bokeh figures (four of the plots)
pip install "ob-analytics[databento]"     # read Databento DBN files
pip install "ob-analytics[live]"          # live Bitstamp capture (websockets)
pip install "ob-analytics[ccxt]"          # live L2 capture from CCXT venues
pip install "ob-analytics[cryptofeed]"    # live L2 or L3 capture through cryptofeed
```

Requires Python 3.11+. Live capture (the `live`, `ccxt` and `cryptofeed`
extras) is supported on Linux and macOS only. CI does not test it on
Windows, where it takes no lock on the capture folder and Ctrl-C does not
close the files cleanly. On Windows, CI tests reading files, the analytics
and the plots.

## Sources

The pipeline reads these sources. `ob-analytics sources` lists them, with
what each can do.

| Source | Level | What it reads or captures | Extra |
|--------|-------|---------------------------|-------|
| `bitstamp` | L3 | Bitstamp `orders.csv` + `trades.csv`, and live capture | `live` to capture |
| `lobster` | L3 | LOBSTER message and orderbook files | none |
| `databento` | L3 | Databento market-by-order (DBN) files | `databento` |
| `depth_csv` | L2 | price-level `depth.csv` + `trades.csv` | none |
| `ccxt` | L2 | live capture from a venue that CCXT supports | `ccxt` |
| `cryptofeed` | L2 or L3 | live capture from a cryptofeed venue | `cryptofeed` |

A capture from `ccxt` or `cryptofeed` writes the same files that `depth_csv`
or `bitstamp` reads, so `ob-analytics process` runs on it directly.

## Run the pipeline

The package ships with a bundled Bitstamp BTC/USD capture. `sample_csv_path()`
returns the orders path; the pipeline finds the sibling `trades.csv` itself.

```python
from ob_analytics import Pipeline, sample_csv_path

result = Pipeline().run(sample_csv_path())

print(f"Events:        {result.events.shape}")
print(f"Trades:        {result.trades.shape}")
print(f"Depth:         {result.depth.shape}")
print(f"Depth summary: {result.depth_summary.shape}")
```

<!-- docs-test: output -->

```text
Events:        (314057, 13)
Trades:        (284, 10)
Depth:         (313561, 5)
Depth summary: (313561, 46)
```

`result` holds four DataFrames: the normalized event stream, the trades,
price-level depth, and a per-timestamp depth summary (best bid/ask, spread,
liquidity bins).

## Make your first plot

Name a concept and `plot_result` (or `result.plot`) wires up the right data
preparation and rendering for you:

```python
from ob_analytics.visualization import plot_result, available_concepts, save_figure

# What can this result plot? (varies by data format)
available_concepts(result)
# {'trade_tape': ['L2', 'L3'], 'depth_heatmap': ['L2'], 'order_outcome': ['L3'], ...}

fig = plot_result(result, "depth_heatmap")       # drawn at L2 only
save_figure(fig, "price_levels.png")

fig = result.plot("trade_tape", "L3")            # equivalent method form
save_figure(fig, "trades.png")
```

A concept that this result has at both L2 and L3, such as `trade_tape` on
an L3 run, needs `level=`; leaving it out raises `ConfigError`. Both forms
also take `ax=` to draw on a Matplotlib axes of your own.

Overrides flow through to the underlying prepare function, and
`backend="plotly"` gives an interactive figure (with the `[interactive]`
extra); `backend="bokeh"` (with the `[bokeh]` extra) gives another, suited
to Bokeh / Panel server dashboards and streaming views:

```python
fig = result.plot("depth_heatmap", col_bias=0.1, backend="plotly")
fig = result.plot("depth_heatmap", col_bias=0.1, backend="bokeh")
```

## Try the CLI

The same pipeline, from a shell:

```bash
ob-analytics process orders.csv -o results/ --gallery
ob-analytics bitstamp-demo orders.csv
```

See [Run from the command line](howto/cli.md) for all subcommands.

## Where next {#where-next}

| If you want to… | Go to |
|-----------------|-------|
| Understand order books, L1/L2/L3, and what the plots mean | [Tutorial](tutorial/index.md) |
| Step through pipeline stages, snapshot the book, compose figures | [Step-by-step control](howto/full-control.md) |
| Use your own Bitstamp CSV or another instrument | [Use your own data](howto/your-own-data.md) |
| Process LOBSTER message + orderbook files | [Process LOBSTER files](howto/lobster.md) |
| Plug in your own loader, trade source, or format | [Plug in custom components](howto/custom-components.md) |
| Theme plots, save artefacts, or use Plotly | [Save, load, and export](howto/output.md) |
| Compute VPIN / Kyle's λ / OFI | [Compute flow toxicity](howto/flow-toxicity.md) |
| Process Databento market-by-order files | [Process Databento files](howto/databento.md) |
| Process price-level (L2) feeds | [Process L2 feeds](howto/l2-depth.md) |
| Capture live order-book data from an exchange | [Capture live data](howto/live-capture.md) |
| Run everything from a shell | [Run from the command line](howto/cli.md) |
| See how the pipeline is designed, and its known limitations | [Architecture](architecture.md) |
| Look up a term | [Glossary](glossary.md) |

!!! info "Looking for the old quickstart?"
    This page used to hold every recipe. The content now lives in the
    how-to guides linked above — nothing was removed, only reorganized.
