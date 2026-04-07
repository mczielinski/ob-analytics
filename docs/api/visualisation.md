---
title: Visualization
---

# Visualization

All plot functions return a figure and accept `backend="matplotlib"` (default)
or `backend="plotly"` for interactive figures. Plotly requires
`pip install ob-analytics[interactive]`. Functions never call `plt.show()`.

## Theme and Saving

::: ob_analytics.visualisation.PlotTheme

::: ob_analytics.visualisation.set_plot_theme

::: ob_analytics.visualisation.get_plot_theme

::: ob_analytics.visualisation.save_figure

## Backend Registration

::: ob_analytics.visualisation.register_plot_backend

## Core Plots

::: ob_analytics.visualisation.plot_price_levels

::: ob_analytics.visualisation.plot_trades

::: ob_analytics.visualisation.plot_event_map

::: ob_analytics.visualisation.plot_volume_map

::: ob_analytics.visualisation.plot_current_depth

::: ob_analytics.visualisation.plot_volume_percentiles

::: ob_analytics.visualisation.plot_events_histogram

::: ob_analytics.visualisation.plot_time_series

## Flow Toxicity Plots

::: ob_analytics.visualisation.plot_vpin

::: ob_analytics.visualisation.plot_order_flow_imbalance

::: ob_analytics.visualisation.plot_kyle_lambda

## LOBSTER-Specific Plots

::: ob_analytics.visualisation.plot_hidden_executions

::: ob_analytics.visualisation.plot_trading_halts
