---
title: Save, load, and export
---

# Theme plots, save artefacts, and export data

How to style figures, save them, serialise pipeline outputs, and switch to
the interactive Plotly backend.

## Themes and saving

There is no global theme to set. Pass a `PlotTheme` to `plot()` and it
applies only to that call (matplotlib backend only):

```python
from ob_analytics.visualization import plot, save_figure, prepare, PlotTheme

theme = PlotTheme(
    style="whitegrid",
    context="talk",
    font_scale=1.2,
    rc={"axes.facecolor": "#f8f9fa", "figure.facecolor": "#ffffff"},
)

fig = plot("trade_tape", level="L2", theme=theme, **prepare.trades(result.trades))
save_figure(fig, "trades_hires.png", dpi=300)
```

## Serialisation

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

## Plotly (interactive)

`plot()` accepts `backend="plotly"` for interactive figures with
zoom, pan, and hover tooltips. Plotly is an optional dependency:

```bash
pip install "ob-analytics[interactive]"
```

```python
from ob_analytics import Pipeline, sample_csv_path

result = Pipeline().run(sample_csv_path())

# col_bias is a power-law gamma: 1.0 (default) is linear so high-volume walls
# stand out; 0.1 brightens thin levels to expose near-touch structure in
# heavy-tailed books; <= 0 selects a log scale.
fig = result.plot("depth_heatmap", backend="plotly", col_bias=0.1)
fig.show()
fig.write_html("depth.html")
```

Whole new backends can be registered by module path:

```python
from ob_analytics.visualization import register_plot_backend
register_plot_backend("bokeh", "my_package._bokeh_backend")
```

## Related

- [Visualization API](../api/visualization.md) — `plot`, `prepare`, concepts and levels
- [Data I/O API](../api/data.md) — `save_data` / `load_data`
