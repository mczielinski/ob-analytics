"""Public data-preparation API for ob-analytics visualizations.

Each function turns analysis frames (a :class:`~ob_analytics.pipeline.PipelineResult`'s
``events`` / ``trades`` / ``depth`` / ``depth_summary``, or an analytic result)
into the payload dict that :func:`ob_analytics.visualization.plot` renders::

    from ob_analytics.visualization import plot, prepare
    fig = plot("trade_tape", level="L2", **prepare.trades(result.trades))

These are thin, stable re-exports of the implementations in
:mod:`ob_analytics.visualization._data` (kept private as the impl module).  For
the common case of plotting straight from a pipeline result, prefer the
one-liner :func:`ob_analytics.visualization.plot_result`, which wires these up
for you; reach for this namespace when you need to pass custom keyword
arguments to a single face.
"""

from __future__ import annotations

from ob_analytics.visualization._data import (
    prepare_book_snapshot_data as book_snapshot,
    prepare_cancellations_l3_data as cancellations_l3,
    prepare_event_map_data as event_map,
    prepare_events_histogram_data as events_histogram,
    prepare_hidden_executions_data as hidden_executions,
    prepare_kyle_lambda_data as kyle_lambda,
    prepare_liquidity_at_touch_data as liquidity_at_touch,
    prepare_ofi_data as ofi,
    prepare_order_activity_l3_data as order_activity_l3,
    prepare_order_outcome_l3_data as order_outcome_l3,
    prepare_price_levels_data as price_levels,
    prepare_time_series_data as time_series,
    prepare_trade_tape_l3_data as trade_tape_l3,
    prepare_trades_data as trades,
    prepare_trading_halts_data as trading_halts,
    prepare_volume_map_data as volume_map,
    prepare_volume_percentiles_data as volume_percentiles,
    prepare_vpin_data as vpin,
)

__all__ = [
    "book_snapshot",
    "cancellations_l3",
    "event_map",
    "events_histogram",
    "hidden_executions",
    "kyle_lambda",
    "liquidity_at_touch",
    "ofi",
    "order_activity_l3",
    "order_outcome_l3",
    "price_levels",
    "time_series",
    "trade_tape_l3",
    "trades",
    "trading_halts",
    "volume_map",
    "volume_percentiles",
    "vpin",
]
