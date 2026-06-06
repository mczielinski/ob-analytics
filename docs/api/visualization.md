---
title: Visualization
---

# Visualization

The unified `plot()` dispatcher renders a plot *concept* at a resolution
*level* on a chosen backend from already-prepared data. Prepare the data with
the matching `prepare_<concept>_data` helper in
`ob_analytics.visualization._data` (or a gallery panel builder) and spread it
as keyword arguments:

```python
from ob_analytics.visualization import plot
from ob_analytics.visualization import _data

fig = plot("trade_tape", backend="matplotlib", **_data.prepare_trades_data(trades))
```

`backend="matplotlib"` (default) returns a Matplotlib figure;
`backend="plotly"` returns an interactive Plotly figure (requires
`pip install ob-analytics[interactive]`). Renderers never call `plt.show()`.

Every concept declares a resolution **level** — `Level.L2` (Market-By-Price
aggregate) or `Level.L3` (Market-By-Order, per order). A concept registered at
a single level resolves it automatically, so you pass only the concept name;
*comparable* concepts (both `L2` and `L3` registered) take an explicit
`level=`.

Available concepts, all currently **L2**: `time_series`, `trade_tape`,
`depth_heatmap`, `order_activity`, `cancellations`, `book_snapshot`,
`volume_percentiles`, `events_histogram`, `hidden_executions`. Level-less
analytics: `vpin`, `order_flow_imbalance`, `kyle_lambda`, `trading_halts`.

## Dispatcher

::: ob_analytics.visualization.plot

## Theme and Saving

Pass `theme=PlotTheme(...)` to `plot()` to override `DEFAULT_THEME` for a
single call (matplotlib backend only); there is no global theme to set.

::: ob_analytics.visualization.PlotTheme

::: ob_analytics.visualization.DEFAULT_THEME

::: ob_analytics.visualization.save_figure

::: ob_analytics.visualization.infer_volume_scale

## Renderer registry

Backends self-register their renderers into `RENDERERS`, keyed by the
coordinate `(concept, level, backend)` (where *level* is a `Level` or `None`
for level-less analytics). Register a whole new backend module with
`register_plot_backend`, or a single renderer directly with
`RENDERERS.register((concept, level, backend), fn)`.

::: ob_analytics.visualization.register_plot_backend

::: ob_analytics.visualization.RENDERERS
