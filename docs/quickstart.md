---
title: Getting started
---

# Getting started

Install the package, run the pipeline on the bundled sample, and render your
first plot — about ten minutes. When something here sparks a question, the
[Tutorial](tutorial/index.md) teaches the concepts; the how-to guides
(linked [below](#where-next)) cover every recipe.

## Install

```bash
pip install ob-analytics
```

Optional extras:

```bash
pip install "ob-analytics[interactive]"   # Plotly figures
pip install "ob-analytics[live]"          # live exchange capture (websockets)
```

Requires Python 3.11+.

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

```
Events:        (~314000, …)   # shape depends on bundled capture
Trades:        (~280, …)
Depth:         (…, 5)
Depth summary: (…, 46)
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

fig = plot_result(result, "depth_heatmap")       # level defaults to L2
save_figure(fig, "price_levels.png")

fig = result.plot("trade_tape", "L3")            # equivalent method form
save_figure(fig, "trades.png")
```

Overrides flow through to the underlying prepare function, and
`backend="plotly"` gives an interactive figure (with the `[interactive]`
extra):

```python
fig = result.plot("depth_heatmap", col_bias=0.1, backend="plotly")
```

## Try the CLI

The same pipeline, from a shell:

```bash
ob-analytics process orders.csv -o results/ --gallery
ob-analytics bitstamp-demo --input orders.csv
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
| Capture live order-book data from an exchange | [Capture live data](howto/live-capture.md) |
| Run everything from a shell | [Run from the command line](howto/cli.md) |
| See how the pipeline is designed | [Architecture](architecture.md) |
| Look up a term | [Glossary](glossary.md) |

!!! info "Looking for the old quickstart?"
    This page used to hold every recipe. The content now lives in the
    how-to guides linked above — nothing was removed, only reorganized.
