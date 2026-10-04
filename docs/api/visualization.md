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
`pip install ob-analytics[interactive]`); `backend="bokeh"` returns a Bokeh
figure (requires `pip install ob-analytics[bokeh]`), covering the core
concepts — `trade_tape`, `depth_heatmap`, `book_snapshot`, `depth_chart` —
for Bokeh / Panel server dashboards and streaming views. Renderers never
call `plt.show()`.

Every concept declares a resolution **level** — `Level.L2` (Market-By-Price
aggregate) or `Level.L3` (Market-By-Order, per order). A concept registered at
a single level resolves it automatically, so you pass only the concept name;
*comparable* concepts (both `L2` and `L3` registered) take an explicit
`level=`.

Concepts with both L2 and L3 faces: `trade_tape`, `order_activity`,
`cancellations`, `book_snapshot`, `depth_chart`, `book_replay`,
`liquidity_at_touch`. L2-only:
`time_series`, `depth_heatmap`, `volume_percentiles`, `events_histogram`,
`hidden_executions`, `price_view`, `trade_size`. L3-only: `order_outcome`,
`queue_position`. Level-less analytics: `vpin`, `order_flow_imbalance`,
`kyle_lambda`, `ofi_horizon`, `l1_ticker`, `book_signals`, `bars`,
`transaction_costs`, `trading_halts`. The first five are also registered
[metrics](metrics.md), so `result.plot("vpin")` computes and draws them.

## Book replay

`book_replay` steps through the book over part of a run. The ladder shows the
book at one instant per `interval`. Beside it, the trades panel shows every
trade of the replay and the mid, with a line at the time on screen. Drag the
slider, or press a play button: `Play 1×` is real time, and each speed in
`speeds` gets a button. A speed is smooth only when a frame lasts about 50 ms
or less, so at 1 s per frame, `Play 20×` glides and the slower speeds step.
For smoother slow playback, pass a shorter `interval`. The file grows with the
number of frames, which `max_frames` caps at 600. When `interval` would need
more frames, the replay uses a longer interval and logs a warning.

`result.plot()` and the gallery replay the gallery's zoom window: the middle
half of a run shorter than two hours, or the second quarter of a longer run.
They set `interval` to 1 s, or longer so that the window fits in about 300
frames. On the sample data the window is about 15 minutes, so each frame is
about 3 s and every speed steps. To choose the window and the interval
yourself, pass `start_time`, `end_time` and `interval`:

```python
import pandas as pd

start = result.trades["timestamp"].iloc[100]
fig = result.plot(
    "book_replay", "L2", backend="plotly",
    start_time=start, end_time=start + pd.Timedelta("5min"), interval="1s",
)
```

Click a trade to move the replay to the last frame at or before it. The price
level the trade took is marked in the trade's colour: the ask for a buy, the
bid for a sell.

The L2 face replays the depth table, so its touch equals the depth summary's
at every frame. The L3 face rebuilds the per-order book from the events at
each frame, with stale crossed orders removed (`uncross=True`, the default
here) so that both faces show the same market. Pass `uncross=False` for the
faithful L3 book. Every frame keeps one price range: the mid's path, wide
enough for about `price_levels` price levels per side.

```python
from ob_analytics.visualization import plot_result

fig = plot_result(result, "book_replay", "L3", backend="plotly")
fig.write_html("replay.html")
```

The replay is drawn by Plotly only. The clicking needs a script, which Plotly
cannot store inside a figure, so the replay is a `BookReplayFigure`: a Plotly
figure that adds the script whenever it becomes HTML (`write_html`,
`to_html`, `show`, and display in a notebook). In a notebook it displays as
HTML, so the notebook must be trusted for the clicking to work.

::: ob_analytics.visualization.prepare.book_replay

## Dispatcher

::: ob_analytics.visualization.plot

## Theme and Saving

Pass `theme=PlotTheme(...)` to `plot()` to override `DEFAULT_THEME` for a
single call, on any backend; there is no global theme to set.

::: ob_analytics.visualization.PlotTheme

::: ob_analytics.visualization.DEFAULT_THEME

::: ob_analytics.visualization.Palette

::: ob_analytics.visualization.DEFAULT_PALETTE

::: ob_analytics.visualization.save_figure

::: ob_analytics.visualization.infer_volume_scale

## Renderer registry

Backends self-register their renderers into `RENDERERS`, keyed by the
coordinate `(concept, level, backend)` (where *level* is a `Level` or `None`
for level-less analytics). A concept is level-less or drawn at a level, and the
same kind on every backend: registering it the other way raises `ValueError`.
Register a whole new backend module with
`register_plot_backend`, or a single renderer directly with
`RENDERERS.register((concept, level, backend), fn)`. To see how a concept is
registered before adding to it, list its `(level, backend)` pairs with
`RENDERERS.placements(concept)`.

::: ob_analytics.visualization.register_plot_backend

::: ob_analytics.visualization.RENDERERS

::: ob_analytics.visualization.RendererRegistry
