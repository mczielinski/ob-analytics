---
title: Visualization
---

# Visualization

To plot straight from a pipeline result, the one-liner `plot_result` (or
`result.plot`) names a concept and wires up the prepare function and context
for you; `available_concepts(result)` lists what a given result can render:

```python
from ob_analytics.visualization import plot_result

fig = plot_result(result, "depth_heatmap")          # level defaults to L2
fig = result.plot("trade_tape", "L3", backend="plotly")
```

For full control, the unified `plot()` dispatcher renders a *concept* at a
resolution *level* on a chosen backend from already-prepared data. Prepare the
payload with the matching helper in the public `prepare` namespace (friendly
wrappers over the internal prepare-data functions) and spread it as keyword
arguments:

```python
from ob_analytics.visualization import plot, prepare

fig = plot("trade_tape", level="L2", **prepare.trades(trades))
```

`backend="matplotlib"` (default) returns a Matplotlib figure;
`backend="plotly"` returns an interactive Plotly figure (requires
`pip install ob-analytics[interactive]`). Renderers never call `plt.show()`.

Every concept declares a resolution **level** — `Level.L2` (Market-By-Price
aggregate) or `Level.L3` (Market-By-Order, per order). A concept registered at
a single level resolves it automatically, so you pass only the concept name;
*comparable* concepts (both `L2` and `L3` registered) take an explicit
`level=`.

Concepts with both L2 and L3 faces: `trade_tape`, `order_activity`,
`cancellations`, `book_snapshot`, `depth_chart`, `liquidity_at_touch`. L2-only:
`time_series`, `depth_heatmap`, `volume_percentiles`, `events_histogram`,
`hidden_executions`, `price_view`, `trade_size`. L3-only: `order_outcome`,
`queue_position`. Level-less analytics: `vpin`, `order_flow_imbalance`,
`kyle_lambda`, `ofi_horizon`, `trading_halts`.

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
