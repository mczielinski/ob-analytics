---
title: Visualization
---

# Visualization

The unified `plot()` dispatcher renders a named plot on a chosen backend from
already-prepared data. Prepare the data with the matching `prepare_<name>_data`
helper in `ob_analytics.visualization._data` (or a gallery panel builder) and
spread it as keyword arguments:

```python
from ob_analytics.visualization import plot
from ob_analytics.visualization import _data

fig = plot("trades", backend="matplotlib", **_data.prepare_trades_data(trades))
```

`backend="matplotlib"` (default) returns a Matplotlib figure;
`backend="plotly"` returns an interactive Plotly figure (requires
`pip install ob-analytics[interactive]`). Renderers never call `plt.show()`.

Available plot names: `time_series`, `trades`, `price_levels`, `event_map`,
`volume_map`, `current_depth`, `volume_percentiles`, `events_histogram`,
`vpin`, `order_flow_imbalance`, `kyle_lambda`, `hidden_executions`,
`trading_halts`.

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

Backends self-register their renderers into `RENDERERS`, keyed by
`(plot_name, backend)`. Register a whole new backend module with
`register_plot_backend`, or a single renderer directly with
`RENDERERS.register((name, backend), fn)`.

::: ob_analytics.visualization.register_plot_backend

::: ob_analytics.visualization.RENDERERS
