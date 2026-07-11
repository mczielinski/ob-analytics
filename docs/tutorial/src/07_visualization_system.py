# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # The visualization system
#
# Every figure in this tutorial — the ladders, the keyframe strips, the
# heatmaps, the queue trajectories, the toxicity faces — came from one
# small system with three moving parts. This closing chapter names them,
# so that when you want a figure this tutorial did not draw, you know
# which of the three to change.
#
# The three parts are:
#
# - **concept** — *what* to show (`depth_heatmap`, `trade_tape`,
#   `queue_position`, …);
# - **level** — at what *resolution*, `L2` (aggregated) or `L3`
#   (per-order), the distinction from [chapter 2](02_three_resolutions.md);
# - **backend** — with which *renderer*, Matplotlib (static, default) or
#   Plotly (interactive).
#
# Name a concept and a level; the system finds the data preparation and
# the renderer for you.
#
# ## The one-liner
#
# You have used it since chapter 1. A `PipelineResult` knows how to plot
# itself:

# %%
# %matplotlib inline
from ob_analytics import Pipeline, sample_csv_path

result = Pipeline().run(sample_csv_path())
fig = result.plot("depth_heatmap")  # level defaults sensibly per concept

# %% [markdown]
# `result.plot(concept, level)` (and the equivalent free function
# `plot_result(result, concept, level)`) is the whole convenience layer.
# It does not guess what a result can draw — it *knows*, because that
# depends on the data's resolution, exactly as chapter 2 promised. Ask
# it:

# %%
from ob_analytics.visualization import available_concepts

available_concepts(result)

# %% [markdown]
# Each concept maps to the levels this particular result can render.
# A Bitstamp capture carries per-order identity, so most concepts offer
# both `L2` and `L3`; some are inherently one resolution —
# `depth_heatmap` aggregates by construction (`L2` only), while
# `queue_position` and `order_outcome` need per-order identity (`L3`
# only). Run this on a feed that lost identity and the L3 rows would
# simply not appear: **the menu is a function of the data**, not a fixed
# list.
#
# ## The concept catalogue
#
# Here is the whole vocabulary at a glance — every concept rendered from
# this one capture, grouped by the question it answers and tagged with
# the chapter that introduced it:

# %%
import matplotlib.pyplot as plt
import pandas as pd

from _docs_theme import DOCS_THEME
from ob_analytics.depth import get_spread
from ob_analytics.visualization import plot, prepare

t0 = result.trades["timestamp"].min()
t1 = t0 + pd.Timedelta(minutes=10)
spread = get_spread(result.depth_summary)

catalogue = [
    (
        "depth_heatmap",
        "L2",
        lambda: prepare.price_levels(
            result.depth,
            spread=spread,
            trades=result.trades,
            col_bias=0.4,
            start_time=t0,
            end_time=t1,
        ),
    ),
    (
        "trade_tape",
        "L2",
        lambda: prepare.trades(
            result.trades, spread=spread, start_time=t0, end_time=t1
        ),
    ),
    ("order_outcome", "L3", lambda: prepare.order_outcome_l3(result.events)),
    (
        "queue_position",
        "L3",
        lambda: prepare.queue_position_l3(result.events, start_time=t0, end_time=t1),
    ),
    (
        "volume_percentiles",
        "L2",
        lambda: prepare.volume_percentiles(
            result.depth_summary, start_time=t0, end_time=t1
        ),
    ),
    ("cancellations", "L3", lambda: prepare.cancellations_l3(result.events)),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, (concept, level, make_payload) in zip(axes.ravel(), catalogue):
    plot(concept, level=level, ax=ax, theme=DOCS_THEME, **make_payload())
    ax.set_title(f"{concept}  ·  {level}", fontsize=10)
fig.tight_layout()

# %% [markdown]
# Six of the fourteen concepts, one capture, one loop. The full set,
# organised:
#
# | Question | Concepts | Introduced |
# |---|---|---|
# | What does the book look like? | `book_snapshot`, `depth_chart`, `depth_heatmap` | [1](01_from_price_to_book.md), [5](05_depth.md) |
# | What traded? | `trade_tape`, `trade_size`, `events_histogram` | [1](01_from_price_to_book.md) |
# | How much liquidity, and where? | `volume_percentiles`, `liquidity_at_touch`, `price_view` | [5](05_depth.md) |
# | Who placed, waited, cancelled, filled? | `order_activity`, `order_outcome`, `queue_position`, `cancellations` | [2](02_three_resolutions.md), [4](04_lifecycles.md) |
#
# A fifth family — the **flow-toxicity** faces `vpin`, `kyle_lambda`,
# `order_flow_imbalance`, `ofi_horizon` from [chapter 6](06_flow_toxicity.md)
# — sits slightly apart: they plot a *computed metric*, not the result
# directly, so they do not appear in `available_concepts` and are drawn
# the long way (next section).
#
# ## Under the one-liner: `plot` and `prepare`
#
# The one-liner is a shortcut for two explicit steps you have watched
# every composite figure in this tutorial take. **Prepare** the data for
# a concept, then **plot** it:

# %%
payload = prepare.trades(result.trades, spread=spread, start_time=t0, end_time=t1)
fig = plot("trade_tape", level="L2", **payload)

# %% [markdown]
# `prepare.<concept>(...)` turns analysis frames into the exact payload a
# face needs and hands it back as a dict; `plot(concept, level, **payload)`
# renders it. `result.plot("trade_tape", "L2")` is precisely these two
# lines with the arguments filled in for you.
#
# Reach past the one-liner for three things, all of which appeared
# earlier in this tutorial:
#
# 1. **Custom arguments** to a face — the `col_bias`, `start_time`,
#    `price_from` windowing knobs from chapter 5 are `prepare` arguments.
# 2. **`ax=`** to place a face in a multi-panel figure — every keyframe
#    strip and story composite in this tutorial is `plot(..., ax=...)`
#    into a grid.
# 3. **Concepts that need a computed input** — the toxicity faces:

# %%
from ob_analytics.flow_toxicity import compute_vpin

vpin = compute_vpin(result.trades, bucket_volume=result.trades["volume"].sum() / 20)
fig = plot("vpin", **prepare.vpin(vpin, threshold=0.7))

# %% [markdown]
# The progression across this whole tutorial has been exactly this
# ladder: `result.plot(...)` when you want a figure now, `plot` +
# `prepare` when you want control, and — for data past the in-memory
# envelope — the pre-slicing you met in chapter 3. Convenience, control,
# scale — the same three levels seaborn and similar libraries offer.
#
# ## Two backends
#
# Everything so far rendered with Matplotlib. Pass `backend="plotly"`
# for an interactive figure — zoom, pan, hover — from the *same* concept
# and data (Plotly ships in the `[interactive]` extra):

# %%
fig = result.plot("depth_heatmap", backend="plotly", col_bias=0.4)
type(fig).__module__.split(".")[0], type(fig).__name__

# %% [markdown]
# The call returns a Plotly figure instead of a Matplotlib one; in a
# notebook `fig.show()` renders it live, and `fig.write_html("depth.html")`
# saves a standalone interactive file. The rule of thumb:
#
# | You want… | Backend |
# |---|---|
# | A figure for a paper, README, or the docs | `matplotlib` (default) |
# | To explore — zoom into a burst, read exact values on hover | `plotly` |
# | A third renderer (Bokeh, …) | register your own (below) |
#
# Backends live in a registry, so a new one is a module path away —
# `register_plot_backend("bokeh", "my_package._bokeh_backend")` — the
# same structural-typing story as loaders and formats from
# [chapter 3](03_loading_data.md). See
# [Extending ob-analytics](../extending.md).
#
# ## Theming
#
# There is no global style to set and accidentally leak between figures.
# A `PlotTheme` is passed per call and applies only to that call (it is
# how every figure in this tutorial shares one look — the `DOCS_THEME`
# used above):

# %%
from ob_analytics.visualization import PlotTheme, save_figure

theme = PlotTheme(style="whitegrid", context="talk", font_scale=1.1)
fig = plot(
    "trade_tape",
    level="L2",
    theme=theme,
    **prepare.trades(result.trades, spread=spread, start_time=t0, end_time=t1),
)
save_figure(fig, "trades_talk.png", dpi=200)

# %% [markdown]
# `PlotTheme` bundles a Seaborn style, context, font scale, and any
# Matplotlib `rc` overrides; `save_figure` writes at the DPI you ask for.
# Theming is Matplotlib-only — Plotly figures carry their own styling.
#
# ## Wrapping up
#
# You started with a single number on a screen and the claim that it was
# the top of an order book. Eight chapters later you can load one from
# any of three feeds, reconstruct it to the individual order, read its
# depth and its queue, score the toxicity of its flow, and render any of
# it two ways — all from frames small enough, at the toy scale, to check
# by hand.
#
# Where to go next:
#
# - **[How-to guides](../howto/full-control.md)** — task-focused recipes:
#   your own data, LOBSTER, custom loaders, live capture, the CLI.
# - **[Extending ob-analytics](../extending.md)** — add a data source, an
#   export format, a plot backend, a live capturer.
# - **API reference** — the [visualization](../api/visualization.md),
#   [analytics](../api/analytics.md), and [flow-toxicity](../api/flow_toxicity.md)
#   modules, function by function.
# - **[Glossary](../glossary.md)** — every term this tutorial introduced,
#   in one place.
#
# ---
#
# *Vocabulary introduced here — concept, level, backend, the
# [`plot` / `prepare`](../api/visualization.md) split — lives in the
# [Glossary](../glossary.md).*
