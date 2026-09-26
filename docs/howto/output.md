---
title: Save, load, and export
---

# Theme plots, save artefacts, and export data

How to style figures, save them, serialise pipeline outputs, and switch to
the interactive Plotly backend.

## Themes and saving

There is no global theme to set. Pass a `PlotTheme` to `plot()` and it
applies only to that call, on any backend:

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

The same theme styles the Plotly and Bokeh backends. Each backend reads
what it can from the shared fields and ignores the other backends'
overrides:

| Field | Matplotlib | Plotly | Bokeh |
|---|---|---|---|
| `palette` | every mark's colour | every mark's colour | every mark's colour |
| `context`, `font_scale` | Seaborn text sizes | template font size | title, axis, and tick text |
| `style` | Seaborn style | `plotly_white`, or `seaborn` for `dark`/`darkgrid` | white, or Seaborn's gray for `dark`/`darkgrid` |
| `rc` | applied last | — | — |
| `plotly_layout` | — | applied last to the template | — |
| `bokeh_figure` | — | — | passed to `bokeh.plotting.figure` |

To change colours, pass a `Palette`. Its fields name what each colour
means, such as `bid`/`ask` (side), `buy`/`sell` (aggressor), and
`price_line`/`reference_line` (neutral marks):

```python
from ob_analytics.visualization import Palette, PlotTheme

theme = PlotTheme(
    palette=Palette(buy="#1f77b4", sell="#d62728"),
    plotly_layout={"font": {"family": "Georgia"}},
)
fig = plot(
    "trade_tape",
    level="L2",
    backend="plotly",
    theme=theme,
    **prepare.trades(result.trades),
)
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

To hand the tables to another tool without writing files first, convert the
result in memory:

```python
tables = result.to_arrow()    # dict[str, pyarrow.Table]
frames = result.to_polars()   # dict[str, polars.DataFrame], needs polars
```

Both give the same four keys as the dict above. The Arrow tables carry the
schema version and tick size in their metadata, the same as the Parquet files.
See [Frame types: pandas in, pandas out](../schema.md#frame-types-pandas-in-pandas-out).

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

## Plotly and Bokeh (interactive)

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

`backend="bokeh"` (with the `[bokeh]` extra: `pip install "ob-analytics[bokeh]"`)
renders the core concepts — `trade_tape`, `depth_heatmap`, `book_snapshot`,
`depth_chart` — the same way, for Bokeh / Panel server dashboards and
streaming views:

```python
fig = result.plot("depth_heatmap", backend="bokeh", col_bias=0.1)
```

Whole new backends can be registered by module path:

```python
from ob_analytics.visualization import register_plot_backend
register_plot_backend("altair", "my_package._altair_backend")
```

## Related

- [Visualization API](../api/visualization.md) — `plot`, `prepare`, concepts and levels
- [Data I/O API](../api/data.md) — `save_data` / `load_data`
