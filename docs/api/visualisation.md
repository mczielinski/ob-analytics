---
title: Visualization
---

# Visualization

All plot functions accept an optional `backend` parameter
(default `"matplotlib"`).  Set `backend="plotly"` for interactive figures
with zoom, pan, and hover tooltips.

Plotly is an optional dependency — install via `pip install ob-analytics[interactive]`.

## Theme

::: ob_analytics.visualisation.PlotTheme

::: ob_analytics.visualisation.set_plot_theme

::: ob_analytics.visualisation.get_plot_theme

::: ob_analytics.visualisation.save_figure

## Backend Registration

::: ob_analytics.visualisation.register_plot_backend

## Plot Functions

::: ob_analytics.visualisation.plot_time_series

::: ob_analytics.visualisation.plot_trades

::: ob_analytics.visualisation.plot_price_levels

::: ob_analytics.visualisation.plot_event_map

::: ob_analytics.visualisation.plot_volume_map

::: ob_analytics.visualisation.plot_current_depth

::: ob_analytics.visualisation.plot_volume_percentiles

::: ob_analytics.visualisation.plot_events_histogram

## Flow Toxicity Plots

::: ob_analytics.visualisation.plot_vpin

::: ob_analytics.visualisation.plot_order_flow_imbalance

::: ob_analytics.visualisation.plot_kyle_lambda
