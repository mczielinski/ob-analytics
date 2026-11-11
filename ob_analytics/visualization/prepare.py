"""Public data-preparation API for ob-analytics visualizations.

Each function turns analysis frames (a :class:`~ob_analytics.pipeline.PipelineResult`'s
``events`` / ``trades`` / ``depth`` / ``depth_summary``, or an analytic result)
into the payload dict that :func:`ob_analytics.visualization.plot` renders.
Each is named after the plot concept it prepares, and takes ``level=`` by the
same rule as :func:`~ob_analytics.visualization.plot`: leave it out when the
concept is drawn at one level (or is level-less), and pass ``"L2"`` or
``"L3"`` when the concept is drawn at both::

    from ob_analytics.visualization import plot, prepare
    fig = plot("trade_tape", level="L2", **prepare.trade_tape(result.trades, level="L2"))

A concept drawn at both levels reads different tables at each: its function
takes the arguments of the face at the *level* given.  For the common case of
plotting straight from a pipeline result, prefer the one-liner
:func:`ob_analytics.visualization.plot_result`, which wires these up for you;
reach for this namespace when you need to pass custom keyword arguments to a
single face.

:func:`hidden_liquidity_overlay` is the one function here that prepares no
plot of its own: it returns the iceberg and hidden-trade frames that the
depth heatmap and the L3 order activity draw on top.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ob_analytics.pipeline import _UNSET
from ob_analytics.protocols import Level
from ob_analytics.visualization import _data
from ob_analytics.visualization._data import (
    prepare_amihud_data as amihud,
    prepare_bars_data as bars,
    prepare_book_replay_data as book_replay,
    prepare_book_signals_data as book_signals,
    prepare_book_snapshot_data as book_snapshot,
    prepare_depth_chart_data as depth_chart,
    prepare_depth_heatmap_data as depth_heatmap,
    prepare_events_histogram_data as events_histogram,
    prepare_hidden_executions_data as hidden_executions,
    prepare_hidden_liquidity_overlay as hidden_liquidity_overlay,
    prepare_kyle_lambda_data as kyle_lambda,
    prepare_l1_ticker_data as l1_ticker,
    prepare_ofi_by_window_data as ofi_by_window,
    prepare_order_flow_imbalance_data as order_flow_imbalance,
    prepare_order_outcome_data as order_outcome,
    prepare_price_view_data as price_view,
    prepare_queue_position_data as queue_position,
    prepare_roll_spread_data as roll_spread,
    prepare_time_series_data as time_series,
    prepare_trade_size_data as trade_size,
    prepare_trading_halts_data as trading_halts,
    prepare_transaction_costs_data as transaction_costs,
    prepare_volume_percentiles_data as volume_percentiles,
    prepare_vpin_data as vpin,
)


def _face(
    concept: str,
    level: Level | str | None,
    faces: dict[Level | None, Callable[..., dict[str, Any]]],
) -> Callable[..., dict[str, Any]]:
    """The prepare function of *concept*'s face at *level*."""
    return faces[_data.resolve_level(concept, level)]


def trade_tape(
    *args: Any, level: Level | str | None = _UNSET, **kwargs: Any
) -> dict[str, Any]:
    """Prepare the trade tape.  ``level`` is required.

    L2: ``trades``, and optionally ``spread``, ``start_time`` and
    ``end_time``.  Draws the trade price as a step line.

    L3: ``events`` and ``trades``, and optionally ``spread``,
    ``volume_scale``, ``start_time``, ``end_time``, ``price_from``,
    ``price_to`` and ``density_threshold``.  Draws each trade at the end of a
    bar that shows how long the maker order had rested.
    """
    faces = {
        Level.L2: _data.prepare_trade_tape_l2_data,
        Level.L3: _data.prepare_trade_tape_l3_data,
    }
    return _face("trade_tape", level, faces)(*args, **kwargs)


def order_activity(
    *args: Any, level: Level | str | None = _UNSET, **kwargs: Any
) -> dict[str, Any]:
    """Prepare the order activity map.  ``level`` is required.

    L2: ``events``, and optionally ``start_time``, ``end_time``,
    ``price_from``, ``price_to``, ``volume_from``, ``volume_to`` and
    ``volume_scale``.  Draws created and deleted events at their price.

    L3: ``events``, and optionally ``volume_scale``, ``start_time``,
    ``end_time``, ``price_from``, ``price_to``, ``max_spans``,
    ``marker_threshold`` and the three frames of
    :func:`hidden_liquidity_overlay`.  Draws each order as one span from
    placement to its end.
    """
    faces = {
        Level.L2: _data.prepare_order_activity_l2_data,
        Level.L3: _data.prepare_order_activity_l3_data,
    }
    return _face("order_activity", level, faces)(*args, **kwargs)


def cancellations(
    *args: Any, level: Level | str | None = _UNSET, **kwargs: Any
) -> dict[str, Any]:
    """Prepare the cancellations plot.  ``level`` is required.

    L2: ``events``, and optionally ``action``, ``event_type``,
    ``start_time``, ``end_time``, ``price_from``, ``price_to``,
    ``volume_from``, ``volume_to``, ``volume_scale`` and ``log_scale``.
    Draws the cancelled volume at each price through time.

    L3: ``events``, and optionally ``volume_scale``, ``start_time``,
    ``end_time``, ``age_floor_s``, ``distance_floor_bps`` and
    ``distance_cap_bps``.  Draws each cancellation by how long the order
    rested and how far from the touch it sat.
    """
    faces = {
        Level.L2: _data.prepare_cancellations_l2_data,
        Level.L3: _data.prepare_cancellations_l3_data,
    }
    return _face("cancellations", level, faces)(*args, **kwargs)


def liquidity_at_touch(
    *args: Any, level: Level | str | None = _UNSET, **kwargs: Any
) -> dict[str, Any]:
    """Prepare the liquidity at the touch.  ``level`` is required.

    L2: ``depth_summary``, and optionally ``volume_scale``, ``start_time``,
    ``end_time`` and ``events``.  Draws the resting size at the best bid and
    ask, with a strip of the events beneath.

    L3: ``events``, and optionally ``side``, ``n_time``, ``start_time`` and
    ``end_time``.  Draws the touch queue, each cell coloured by the age of
    the order at that place in the queue.
    """
    faces = {
        Level.L2: _data.prepare_liquidity_at_touch_l2_data,
        Level.L3: _data.prepare_liquidity_at_touch_l3_data,
    }
    return _face("liquidity_at_touch", level, faces)(*args, **kwargs)


__all__ = [
    "amihud",
    "bars",
    "book_replay",
    "book_signals",
    "book_snapshot",
    "cancellations",
    "depth_chart",
    "depth_heatmap",
    "events_histogram",
    "hidden_executions",
    "hidden_liquidity_overlay",
    "kyle_lambda",
    "l1_ticker",
    "liquidity_at_touch",
    "ofi_by_window",
    "order_activity",
    "order_flow_imbalance",
    "order_outcome",
    "price_view",
    "queue_position",
    "roll_spread",
    "time_series",
    "trade_size",
    "trade_tape",
    "trading_halts",
    "transaction_costs",
    "volume_percentiles",
    "vpin",
]
