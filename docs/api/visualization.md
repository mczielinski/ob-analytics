---
title: Visualization
---

# Visualization

All plot functions return a figure and accept `backend="matplotlib"` (default)
or `backend="plotly"` for interactive figures. Plotly requires
`pip install ob-analytics[interactive]`. Functions never call `plt.show()`.

## Theme and Saving

::: ob_analytics.visualization.PlotTheme

::: ob_analytics.visualization.set_plot_theme

::: ob_analytics.visualization.get_plot_theme

::: ob_analytics.visualization.save_figure

## Backend Registration

::: ob_analytics.visualization.register_plot_backend

## Core Plots

::: ob_analytics.visualization.plot_price_levels

::: ob_analytics.visualization.plot_trades

::: ob_analytics.visualization.plot_event_map

::: ob_analytics.visualization.plot_volume_map

::: ob_analytics.visualization.plot_current_depth

::: ob_analytics.visualization.plot_volume_percentiles

::: ob_analytics.visualization.plot_events_histogram

::: ob_analytics.visualization.plot_time_series

## Flow Toxicity Plots

::: ob_analytics.visualization.plot_vpin

::: ob_analytics.visualization.plot_order_flow_imbalance

::: ob_analytics.visualization.plot_kyle_lambda

## LOBSTER-Specific Plots

::: ob_analytics.visualization.plot_hidden_executions

::: ob_analytics.visualization.plot_trading_halts
