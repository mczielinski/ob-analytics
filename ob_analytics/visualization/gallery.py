"""Reusable plot gallery generator for ob-analytics.

A :class:`GalleryModel` is the single source of truth: leveled order-book
:class:`PlotConcept` objects (each with up to one :class:`PlotSpec` variant per
:class:`~ob_analytics.visualization.Level`) plus a separate list of level-less
cards.  A *view* (``l2`` / ``l3`` / ``both`` / ``comparison``)
projects that model into gallery cards; :func:`generate_gallery` renders every
selected face across the backend (or L2|L3) column axis, saves the figures, and
writes a standalone HTML page.

Usage::

    from ob_analytics.visualization.gallery import generate_gallery

    gallery_path = generate_gallery(result, "output/gallery/", view="both")
"""

from __future__ import annotations

import html as html_mod
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from matplotlib.axes import Axes

from ob_analytics._utils import (
    lots_to_size,
    ticks_to_price,
    ticks_to_price_if_integer,
)
from ob_analytics.depth import get_spread
from ob_analytics.exceptions import ConfigError, UnknownNameError
from ob_analytics.flow_toxicity import OFI_WINDOWS
from ob_analytics.pipeline import _UNSET, PipelineResult
from ob_analytics.protocols import Level
from ob_analytics.visualization import (
    PlotTheme,
    _check_ax,
    _concepts,
    _data as _viz_data,
    _resolve_level,
    infer_volume_scale,
    plot,
    save_figure,
)
from ob_analytics.visualization._renderers import (
    _BACKEND_MODULES,
    _kind_text,
    _load_backend,
    _same_kind,
)

__all__ = [
    "GalleryModel",
    "PlotConcept",
    "PlotSpec",
    "available_concepts",
    "bars_card",
    "build_gallery_model",
    "display_result",
    "generate_gallery",
    "kyle_lambda_card",
    "ofi_by_window_card",
    "plot_result",
    "trading_halts_card",
]

#: Views recognised by :func:`generate_gallery` / :func:`_project`.
VIEWS = ("l2", "l3", "both", "comparison")


@dataclass
class PlotSpec:
    """How to prepare and label a single plot face.

    A spec carries no level of its own -- the level is a coordinate on the
    owning :class:`PlotConcept` (for leveled concepts) or ``None`` (for the
    level-less analytics in :attr:`GalleryModel.analytics`).

    Parameters
    ----------
    name : str
        Identifier / file stem for level-less cards (e.g. ``"vpin"``).  For a
        concept variant the file stem is derived from the concept + level, so
        ``name`` is only metadata there.
    title : str
        Human-readable title (used directly for analytic cards; concept cards
        take their title from the concept).
    plot_name : str
        Dispatcher concept key for :func:`ob_analytics.visualization.plot`.
    prepare : Callable
        ``prepare_<name>_data`` helper returning the renderer payload.
    prep_kwargs : dict
        Keyword arguments passed to *prepare*.
    """

    name: str
    title: str
    plot_name: str
    prepare: Callable[..., dict]
    prep_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlotConcept:
    """A level-neutral plot identity with up to one variant per :class:`Level`.

    The ``key`` never carries a level; the level lives in the registry
    coordinate ``(concept, level, backend)`` and in the keys of ``variants``.
    """

    key: str
    title: str
    variants: dict[Level, PlotSpec]
    note: str = ""

    def at(self, level: Level) -> PlotSpec | None:
        """Return the variant registered at *level*, or ``None``."""
        return self.variants.get(level)

    @property
    def comparable(self) -> bool:
        """True when both an L2 and an L3 face are registered.

        Derived, never curated: the existence of both variants *is* the
        L2<->L3 pairing, so it can never drift out of sync with what is built.
        """
        return Level.L2 in self.variants and Level.L3 in self.variants


@dataclass
class GalleryModel:
    """The gallery's inventory: leveled concepts + level-less analytics."""

    concepts: list[PlotConcept]
    analytics: list[PlotSpec] = field(default_factory=list)


def _zoom_window(
    trades: pd.DataFrame, clock: pd.DataFrame
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The gallery's zoom window: the busiest stretch of trading.

    The window is half of the run when the run is shorter than two hours,
    and a quarter of it otherwise (see :func:`~ob_analytics.visualization.busiest_window`).
    *clock* (the events, or the depth on a price-level run) sets the run's
    length, and picks the window when there are no trades.
    """
    t_min, t_max = clock["timestamp"].min(), clock["timestamp"].max()
    duration = t_max - t_min
    if pd.isna(duration) or duration <= pd.Timedelta(0):
        return t_min, t_max
    width = duration / 2 if duration < pd.Timedelta(hours=2) else duration / 4
    return _viz_data.busiest_window(trades if not trades.empty else clock, width)


def _window_over(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (*start*, *end*) if it holds a row of *frame*, else *frame*'s span.

    The shared zoom window comes from one clock (events or depth), but some
    faces are given the depth summary with its first minute dropped.  On a
    short or sparse capture the two need not overlap, and a window with no
    rows leaves the face nothing to draw.  A zoom window derived from *frame*
    alone would not fix that: it can fall between two sparse rows.  So the
    face keeps the shared window when it can, and otherwise shows all of
    *frame*.  An empty *frame* returns the window unchanged.
    """
    ts = frame["timestamp"]
    if ts.empty or ts.between(start, end).any():
        return start, end
    return ts.min(), ts.max()


def _concept_cards(scene: _concepts.Scene) -> list[PlotConcept]:
    """One :class:`PlotConcept` per table concept the run has a face for.

    Every face comes from the concept table, bound to *scene*; a face whose
    data the run lacks binds to nothing and is left out.
    """
    cards: list[PlotConcept] = []
    for entry in _concepts.CONCEPTS:
        variants: dict[Level, PlotSpec] = {}
        title = entry.title.format(snapshot=scene.snapshot_label)
        for level, face in entry.faces.items():
            if level is None or face.bind is None:
                continue
            binding = face.bind(scene)
            if binding is not None:
                variants[level] = PlotSpec(
                    entry.key, title, entry.key, binding.prepare, binding.kwargs
                )
        if variants:
            cards.append(PlotConcept(entry.key, title, variants, note=entry.note))
    return cards


def _compute_settings(metric: Any) -> frozenset[str]:
    """The keyword names *metric*'s ``compute`` takes after the result."""
    params = list(inspect.signature(metric.compute).parameters.values())[1:]
    return frozenset(
        p.name for p in params if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
    )


def _metric_payload(metric: Any, result: PipelineResult, **settings: Any) -> dict:
    """Compute *metric* over *result* and prepare its payload.

    Each keyword in *settings* goes to the method that names it: to
    ``compute`` when its signature has it, otherwise to ``prepare``, so
    ``plot_result(result, "vpin", bucket_volume=5.0, threshold=0.8)`` sets
    the bucket size of the measurement and the alert line of the picture.
    """
    names = _compute_settings(metric)
    compute = {k: v for k, v in settings.items() if k in names}
    prepare = {k: v for k, v in settings.items() if k not in names}
    return metric.prepare(metric.compute(result, **compute), **prepare)


#: Settings the gallery picks for a metric in place of the metric's defaults.
#: VPIN's default bucket (a day's volume ÷ 50) fills one bucket on a capture
#: of half an hour, so the gallery has ``vpin`` fit it to the capture.
_GALLERY_SETTINGS: dict[str, dict[str, Any]] = {"vpin": {"bucket_volume": "fit"}}


def _metric_cards(result: PipelineResult) -> list[PlotSpec]:
    """Build one card per registered metric that applies to *result*.

    A metric registered in :data:`~ob_analytics.metrics.METRICS` becomes a
    gallery card without any edit here: its ``name`` is the level-less plot
    concept the card dispatches on, so a renderer registered at
    ``(name, None, backend)`` draws it.  A metric that does not apply to
    this run (see :func:`~ob_analytics.metrics.skip_reason`) is skipped.

    The metric is computed when its card is prepared, not here, so building
    the model (which every :func:`plot_result` call does) costs nothing for
    the metrics it does not draw.  A metric that raises fails only its own
    card, which says why.

    *result* is the run as it is, not converted with :func:`display_result`:
    every metric reports in the instrument's own units from the run's own
    frames, so a card shows the numbers ``result.metric(name)`` returns.  The
    one exception is a setting in :data:`_GALLERY_SETTINGS`: VPIN's buckets
    are fitted to the capture, and the card says so when that changed them.
    """
    from ob_analytics.metrics import applicable_metrics

    return [
        PlotSpec(
            metric.name,
            metric.title,
            metric.name,
            partial(_metric_payload, metric, result, **_gallery_settings(metric)),
        )
        for metric in applicable_metrics(result)
    ]


def _gallery_settings(metric: Any) -> dict[str, Any]:
    """The :data:`_GALLERY_SETTINGS` for *metric* that its ``compute`` takes.

    A plug-in metric registered under a built-in name keeps its own defaults
    for any setting its ``compute`` does not name.
    """
    names = _compute_settings(metric)
    settings = _GALLERY_SETTINGS.get(metric.name, {})
    return {k: v for k, v in settings.items() if k in names}


#: Price-valued columns per frame, converted from integer ticks to a
#: quote-currency float for display.  Bps columns are scale-free
#: and are left alone; size-valued columns are handled by
#: :data:`_DISPLAY_SIZE_COLUMNS`.
_DISPLAY_PRICE_COLUMNS: dict[str, tuple[str, ...]] = {
    "events": ("price",),
    "trades": ("price",),
    "depth": ("price",),
    "depth_summary": ("best_bid_price", "best_ask_price"),
}

#: Size-valued columns per frame, converted from integer lots to a base-asset
#: float for display.  ``depth_summary`` carries one volume column
#: per bps bin and the bin count is configurable, so that frame is matched by
#: name rather than listed.
_DISPLAY_SIZE_COLUMNS: dict[str, tuple[str, ...]] = {
    "events": ("volume", "fill"),
    "trades": ("volume",),
    "depth": ("volume",),
}


def _size_columns(df: pd.DataFrame) -> tuple[str, ...]:
    """Return the size-valued ``depth_summary`` columns present in *df*.

    Every one is named ``..._vol`` or ``..._volNNNbps`` — the per-bin depth
    columns, whose count follows ``depth_bins``, plus the two at the touch.
    """
    return tuple(c for c in df.columns if "vol" in c)


def display_result(result: PipelineResult) -> PipelineResult:
    """Return *result* with prices and sizes converted to display units.

    Canonical prices are integer ticks (:mod:`ob_analytics.schemas`); the plots
    show the quote currency, so this converts each price column to the
    ``ticks * tick_size`` float the faces render, reading ``tick_size`` from
    ``result.config``.  :func:`build_gallery_model` calls it once so every face —
    the ``prepare`` helpers, ``order_book`` / ``queue_positions`` called inside
    them, the axes — works in display units; call it yourself before the
    low-level ``prepare.*`` builders when you plot straight from a result.

    Sizes are converted the same way, from integer lots to the base-asset
    ``lots * lot_size`` float.

    Idempotent and legacy-safe: only integer columns are scaled, so a result
    whose prices or sizes are already floats (a pre-tick or pre-lot file, or a
    result already passed through here) is returned unchanged.
    """
    tick_size = getattr(result.config, "tick_size", 1.0)
    decimals = getattr(result.config, "price_decimals", None)
    lot_size = getattr(result.config, "lot_size", 1.0)
    size_decimals = getattr(result.config, "volume_decimals", None)

    def _scale(df: pd.DataFrame, columns: tuple[str, ...], convert) -> pd.DataFrame:
        present = [
            c
            for c in columns
            if c in df.columns and pd.api.types.is_integer_dtype(df[c])
        ]
        if df.empty or not present:
            return df
        out = df.copy()
        for column in present:
            out[column] = convert(out[column].to_numpy())
        return out

    def to_display(
        df: pd.DataFrame, price_columns: tuple[str, ...], size_columns: tuple[str, ...]
    ) -> pd.DataFrame:
        df = _scale(
            df,
            price_columns,
            lambda a: ticks_to_price(a, tick_size, decimals=decimals),
        )
        return _scale(
            df,
            size_columns,
            lambda a: lots_to_size(a, lot_size, decimals=size_decimals),
        )

    return replace(
        result,
        events=to_display(
            result.events,
            _DISPLAY_PRICE_COLUMNS["events"],
            _DISPLAY_SIZE_COLUMNS["events"],
        ),
        trades=to_display(
            result.trades,
            _DISPLAY_PRICE_COLUMNS["trades"],
            _DISPLAY_SIZE_COLUMNS["trades"],
        ),
        depth=to_display(
            result.depth,
            _DISPLAY_PRICE_COLUMNS["depth"],
            _DISPLAY_SIZE_COLUMNS["depth"],
        ),
        depth_summary=to_display(
            result.depth_summary,
            _DISPLAY_PRICE_COLUMNS["depth_summary"],
            _size_columns(result.depth_summary),
        ),
    )


def _hidden_liquidity_overlay(
    raw_result: PipelineResult,
    zoom_start: pd.Timestamp,
    zoom_end: pd.Timestamp,
    focus: _viz_data.FocusWindow,
) -> dict[str, pd.DataFrame]:
    """Iceberg refills and trades against hidden orders, in display units.

    They are overlaid on the depth heatmap and the per-order activity map, so
    hidden liquidity shows without reading the tables.  They are clipped to
    the zoom window: a full LOBSTER day has about 900 suspected icebergs and
    9,600 hidden trades, which would flood a face meant to show a few
    minutes.  A failed detector run, unit conversion or clip must not stop
    the gallery (as a failed metric fails only its own card), so any failure
    drops the overlay and logs a warning.  A run with no per-order events has
    no overlay.
    """
    if raw_result.events.empty:
        return {}
    try:
        from ob_analytics.hidden_liquidity import (
            hidden_trades as find_hidden_trades,
            iceberg_slices,
        )

        # Detected on raw ticks (exact price equality), then its own price
        # columns are converted to display units the same way display_result()
        # converted the core frames -- the overlay must land on the axes it
        # draws over. Legacy-safe like display_result()'s own _scale: only an
        # integer (tick) column is converted, so a pre-tick result whose
        # events/trades already carry display-unit floats is not scaled twice.
        slices = iceberg_slices(raw_result.events, raw_result.trades)
        hidden = find_hidden_trades(
            raw_result.events, raw_result.trades, raw_result.depth_summary
        )
        tick_size = getattr(raw_result.config, "tick_size", 1.0)
        decimals = getattr(raw_result.config, "price_decimals", None)

        slices_display = slices.assign(
            price=ticks_to_price_if_integer(
                slices["price"], tick_size, decimals=decimals
            )
        )
        hidden_display = hidden.assign(
            price=ticks_to_price_if_integer(
                hidden["price"], tick_size, decimals=decimals
            ),
            best_bid_price=ticks_to_price_if_integer(
                hidden["best_bid_price"], tick_size, decimals=decimals
            ),
            best_ask_price=ticks_to_price_if_integer(
                hidden["best_ask_price"], tick_size, decimals=decimals
            ),
        )
        return _viz_data.prepare_hidden_liquidity_overlay(
            slices_display,
            hidden_display,
            start_time=zoom_start,
            end_time=zoom_end,
            price_from=focus.price_from,
            price_to=focus.price_to,
        )
    except Exception as e:  # noqa: BLE001 -- one bad overlay must not sink the gallery
        logger.warning("Gallery: hidden-liquidity overlay failed: {}", e)
        return {}


def build_gallery_model(
    result: PipelineResult,
    *,
    volume_scale: float | None = None,
) -> GalleryModel:
    """Build the default gallery model from pipeline results.

    Each order-book concept comes from the concept table
    (:mod:`ob_analytics.visualization._concepts`), with a variant at each
    level whose data the run has: a price-level (L2) run has no per-order
    events, so it gets the L2 faces only.  A concept with both an L2 and an
    L3 variant is :attr:`~PlotConcept.comparable`.  The analytics are one
    card per registered metric that applies to the run; callers may append
    more (built with the ``*_card`` helpers) to :attr:`GalleryModel.analytics`.

    Parameters
    ----------
    result : PipelineResult
        Pipeline output.
    volume_scale : float or None
        Volume display scale.  ``None`` (default) infers a power-of-10 scale
        via :func:`infer_volume_scale`, from the events' volumes, or on a
        price-level run from the depth's (the trades' when there is no depth).

    Returns
    -------
    GalleryModel
    """
    # The overlay detectors need integer-tick prices (exact equality), and the
    # metrics report in the instrument's own units, so both read the result
    # before it is converted to display units.
    raw_result = result
    result = display_result(result)

    events = result.events
    trades = result.trades
    depth = result.depth
    depth_summary = result.depth_summary

    # The clock that sets the run's length: the events, or the depth on a
    # price-level run, which has no events.
    clock = events if not events.empty else depth

    if volume_scale is None:
        basis = (
            events["volume"]
            if not events.empty
            else (depth["volume"] if not depth.empty else trades.get("volume"))
        )
        volume_scale = (
            infer_volume_scale(basis) if basis is not None and len(basis) else 1.0
        )

    zoom_start, zoom_end = _zoom_window(trades, clock)
    # One mid-anchored price band feeds every clipped face, so faces that get
    # the same bounds stay comparable on shared axes.
    focus = _viz_data.focus_window(trades)

    offset = clock["timestamp"].min() + pd.Timedelta(minutes=1)
    summary = depth_summary[depth_summary["timestamp"] >= offset]
    summary_start, summary_end = _window_over(summary, zoom_start, zoom_end)

    scene = _concepts.Scene(
        events=events,
        trades=trades,
        depth=depth,
        depth_summary=summary,
        spread=get_spread(depth_summary),
        volume_scale=volume_scale,
        zoom_start=zoom_start,
        zoom_end=zoom_end,
        summary_start=summary_start,
        summary_end=summary_end,
        focus=focus,
        overlay=_hidden_liquidity_overlay(raw_result, zoom_start, zoom_end, focus),
    )
    return GalleryModel(
        concepts=_concept_cards(scene), analytics=_metric_cards(raw_result)
    )


def plot_result(
    result: PipelineResult,
    concept: str,
    level: Level | str | None = _UNSET,
    *,
    backend: str = "matplotlib",
    ax: Axes | None = None,
    volume_scale: float | None = None,
    theme: PlotTheme | None = None,
    **overrides: Any,
) -> Any:
    """Render one plot *concept* straight from a :class:`PipelineResult`.

    The one-liner the gallery is built on, without the gallery: it derives the
    same per-face context (spread, focus window, snapshot, volume scale) via
    :func:`build_gallery_model` -- the single source of truth for how each
    concept binds to its prepare function -- then prepares and renders the face.
    So ``plot_result(result, "depth_heatmap")`` replaces calling
    ``prepare.depth_heatmap`` with the run's tables yourself.

    Parameters
    ----------
    result : PipelineResult
        Pipeline output (``events`` / ``trades`` / ``depth`` / ``depth_summary``).
    concept : str
        Concept key, e.g. ``"trade_tape"`` (see :data:`available_concepts`).
        The name of a registered metric works too; a metric is level-less, so
        *level* does not apply to it.  A metric measures *result* as it is,
        not converted with :func:`display_result`, and draws the table
        ``result.metric(concept)`` returns.
    level : Level or str, optional
        ``"L2"`` / ``"L3"`` (or a :class:`Level`).  Leave it out when *result*
        has the concept at one level; it is required when *result* has the
        concept at both L2 and L3 (an L3 run has most concepts at both).
    backend : str
        ``"matplotlib"`` (default), ``"plotly"``, or ``"bokeh"``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on, as in :func:`~ob_analytics.visualization.plot`
        (Matplotlib only).
    volume_scale : float or None
        Display volume scale; ``None`` auto-infers (as the gallery does).
    theme : PlotTheme or None
        Theme for this call, on any backend; ``None`` uses
        :data:`~ob_analytics.visualization.DEFAULT_THEME`.
    **overrides
        Extra keyword arguments merged over the prepare call (e.g.
        ``col_bias=0.1`` for the depth heatmap).  For a metric, each goes to
        its ``compute`` when that names it and to its ``prepare`` otherwise,
        e.g. ``bucket_volume=5.0, threshold=0.8`` for ``"vpin"``.

    Returns
    -------
    A Matplotlib, Plotly or Bokeh figure, per *backend*.

    Raises
    ------
    UnknownNameError
        If *concept* is not built for this result, has no variant at *level*,
        is not drawn by *backend*, or *backend* is not registered.  The
        message names what is available: for a missing backend, the
        backends that draw the plot.
    ConfigError
        If *level* was left out and *result* has *concept* at both L2 and
        L3, or *ax* is given for a backend other than Matplotlib.
    """
    from ob_analytics.metrics import METRICS, skip_reason

    _load_backend(backend)
    _check_ax(ax, backend)

    # A face registered at a level keeps its name, so a metric with the same
    # name cannot hide it.
    is_face = any(lvl is not None for lvl, _ in _concepts.placements(concept))
    metric = None if is_face or concept not in METRICS else METRICS.get(concept)
    if metric is not None and skip_reason(metric, result) is None:
        # A registered metric is a level-less concept: it has no L2/L3
        # variants, and it measures the run's own frames, so it is drawn
        # without building the gallery model or converting the result.
        resolved = _resolve_level(concept, level, backend)
        _concepts.require(concept, resolved, backend)
        data = _metric_payload(metric, result, **overrides)
        if theme is not None:
            data["theme"] = theme
        return plot(concept, resolved, backend=backend, ax=ax, **data)

    model = build_gallery_model(result, volume_scale=volume_scale)
    concept_map = {c.key: c for c in model.concepts}
    pc = concept_map.get(concept)
    if pc is None:
        metric_names = sorted(spec.name for spec in model.analytics)
        built_elsewhere = bool(_concepts.placements(concept)) or concept in METRICS
        reason = None if metric is None else skip_reason(metric, result)
        if reason is not None:
            hint = f"It is not built for this result: {reason}."
        elif built_elsewhere:
            hint = "It is not built for this result."
        else:
            hint = None
        raise UnknownNameError(
            "concept", concept, sorted(concept_map) + metric_names, hint=hint
        )

    resolved = _resolve_level(concept, level, backend, pc.variants)
    spec = None if resolved is None else pc.at(resolved)
    if spec is None:
        have = ", ".join(sorted(lvl.value for lvl in pc.variants))
        raise UnknownNameError(
            f"{_kind_text([resolved]).removeprefix('at ')} concept",
            concept,
            (),
            hint=f"It is drawn at {have}.",
        )
    _concepts.require(concept, resolved, backend)

    data = spec.prepare(**{**spec.prep_kwargs, **overrides})
    if theme is not None:
        data["theme"] = theme
    return plot(concept, resolved, backend=backend, ax=ax, **data)


def available_concepts(result: PipelineResult) -> dict[str, list[str]]:
    """Concept keys built for *result*, mapped to their available level names.

    A discoverability companion to :func:`plot_result`: shows what
    ``plot_result(result, concept, level=...)`` can render for this dataset
    (which varies by format -- e.g. ``hidden_executions`` is LOBSTER-only).

    A registered metric is listed too, with an empty level list: a metric is
    level-less, so ``plot_result(result, "amihud")`` takes no ``level=``.
    """
    model = build_gallery_model(result)
    concepts = {c.key: sorted(lvl.value for lvl in c.variants) for c in model.concepts}
    return {**concepts, **{spec.name: [] for spec in model.analytics}}


def bars_card(bars: pd.DataFrame, *, label: str = "") -> PlotSpec:
    """Build a bars card for :attr:`GalleryModel.analytics`.

    *bars* is a table from :func:`ob_analytics.bars.bars`.  *label* names
    the rule and threshold it was cut with and is shown in the chart title;
    the default reads them off the frame.  Prices are drawn as they arrive, so
    cut the bars from a display-unit trades frame (see :func:`display_result`)
    when the rest of the gallery is in the quote currency.
    """
    label = label or _viz_data.bars_label(bars)
    rule = bars.attrs.get("bar_rule")
    return PlotSpec(
        # The file stem, so two cuts of the same trades do not overwrite each
        # other's image; the concept both of them draw under is "bars".
        f"bars_{rule}" if rule else "bars",
        f"Bars ({label})" if label else "Bars",
        "bars",
        _viz_data.prepare_bars_data,
        {"bars": bars, "label": label},
    )


def ofi_by_window_card(
    trades: pd.DataFrame,
    *,
    windows: tuple[str, ...] = OFI_WINDOWS,
    grid: str = "5s",
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
) -> PlotSpec:
    """Build an OFI-by-window card for :attr:`GalleryModel.analytics` (one band per window)."""
    return PlotSpec(
        "ofi_by_window",
        "Order Flow Imbalance by Window",
        "ofi_by_window",
        _viz_data.prepare_ofi_by_window_data,
        {
            "trades": trades,
            "windows": windows,
            "grid": grid,
            "start_time": start_time,
            "end_time": end_time,
        },
    )


def kyle_lambda_card(kyle_result: Any) -> PlotSpec:
    """Build a Kyle's lambda card for :attr:`GalleryModel.analytics`."""
    return PlotSpec(
        "kyle_lambda",
        f"Kyle's Lambda (lambda={kyle_result.lambda_:.4f})",
        "kyle_lambda",
        _viz_data.prepare_kyle_lambda_data,
        {"kyle_result": kyle_result},
    )


def trading_halts_card(trades: pd.DataFrame, halts: pd.DataFrame) -> PlotSpec:
    """Build a trading-halts card for :attr:`GalleryModel.analytics`.

    LOBSTER halts are not part of the slim :class:`PipelineResult`; read them
    from :attr:`~ob_analytics.lobster.LobsterLoader.trading_halts` and pass
    them here.
    """
    return PlotSpec(
        "trading_halts",
        "Trading Halts",
        "trading_halts",
        _viz_data.prepare_trading_halts_data,
        {"trades": trades, "halts": halts},
    )


# ---------------------------------------------------------------------------
# View projection: GalleryModel -> gallery cards
# ---------------------------------------------------------------------------


@dataclass
class _Panel:
    """One column within a gallery card: what to render and how to display it.

    The render coordinate (``concept``/``level``/``prepare``) drives
    :func:`ob_analytics.visualization.plot`; the display fields
    (``backend``/``stem``/``label``/...) drive the HTML.  ``rendered`` and
    ``reason`` are set by the render loop and consumed by :func:`_render_panel`.
    """

    concept: str
    level: Level | None
    prepare: Callable[..., dict]
    prep_kwargs: dict[str, Any]
    backend: str
    stem: str
    label: str
    panel_cls: str
    role: str  # "primary" | "secondary" | "equal"
    rendered: bool = False
    reason: str = ""  # why the panel was not drawn, shown on the card


@dataclass
class _Card:
    """A gallery card: a title, one panel per column, and a how-to-read note."""

    title: str
    panels: list[_Panel]
    note: str = ""


_LEVEL_LABEL: dict[Level, str] = {
    Level.L2: "L2 -- MBP (aggregate)",
    Level.L3: "L3 -- MBO (per order)",
}


def _panel_cls(backend: str) -> str:
    style = _BACKEND_STYLES.get(backend)
    return style.panel_cls if style else f"{backend}-panel"


def _backend_panels(
    concept: str,
    level: Level | None,
    spec: PlotSpec,
    stem: str,
    backends: list[str],
) -> list[_Panel]:
    """One panel per backend (the column axis for l2/l3/both views)."""
    panels: list[_Panel] = []
    for i, backend in enumerate(backends):
        panels.append(
            _Panel(
                concept=concept,
                level=level,
                prepare=spec.prepare,
                prep_kwargs=spec.prep_kwargs,
                backend=backend,
                stem=stem,
                label=_backend_label(backend),
                panel_cls=_panel_cls(backend),
                role="primary" if i == 0 else "secondary",
            )
        )
    return panels


def _comparison_backend(backends: list[str]) -> str:
    """The single backend used for the side-by-side L2|L3 comparison view."""
    return "plotly" if "plotly" in backends else backends[0]


def _project(
    model: GalleryModel,
    view: str,
    backends: list[str],
) -> list[_Card]:
    """Project *model* into gallery cards for *view*.

    ``l2`` / ``l3`` / ``both`` lay each selected face across the *backend*
    column axis (plus analytics for ``l2`` / ``both``); ``comparison`` pairs the
    L2 and L3 faces of each :attr:`~PlotConcept.comparable` concept in a single
    card with an L2|L3 column axis on one backend.
    """
    if view not in VIEWS:
        raise UnknownNameError("gallery view", view, VIEWS)

    if view == "comparison":
        backend = _comparison_backend(backends)
        cards = []
        for concept in model.concepts:
            # Comparable concepts get an L2|L3 column pair; single-variant
            # concepts get a one-column card labeled "L2-only" / "L3-only"
            # rather than being dropped from the view silently.
            panels = []
            for level, spec in sorted(
                concept.variants.items(), key=lambda kv: kv[0].value
            ):
                label = _LEVEL_LABEL[level]
                if not concept.comparable:
                    label = f"{label}  ·  {level.value}-only"
                panels.append(
                    _Panel(
                        concept=concept.key,
                        level=level,
                        prepare=spec.prepare,
                        prep_kwargs=spec.prep_kwargs,
                        backend=backend,
                        stem=f"{concept.key}.{level}",
                        label=label,
                        panel_cls=_panel_cls(backend),
                        role="equal",
                    )
                )
            cards.append(_Card(concept.title, panels, note=concept.note))
        return cards

    levels_for_view = {
        "l2": (Level.L2,),
        "l3": (Level.L3,),
        "both": (Level.L2, Level.L3),
    }[view]

    cards = []
    for concept in model.concepts:
        for level in levels_for_view:
            spec = concept.at(level)
            if spec is None:
                continue
            stem = f"{concept.key}.{level}"
            title = f"{concept.title} -- {level}"
            cards.append(
                _Card(
                    title,
                    _backend_panels(concept.key, level, spec, stem, backends),
                    note=concept.note,
                )
            )

    # Analytics are level-less: every leveled view includes them.
    for spec in model.analytics:
        cards.append(
            _Card(
                spec.title,
                _backend_panels(spec.plot_name, None, spec, spec.name, backends),
            )
        )
    return cards


def generate_gallery(
    result: PipelineResult | None,
    output_dir: str | Path,
    *,
    model: GalleryModel | None = None,
    view: str = "both",
    volume_scale: float | None = None,
    backends: list[str] | None = None,
    title: str = "ob-analytics Plot Gallery",
) -> Path:
    """Render a gallery view and write a standalone HTML page.

    Parameters
    ----------
    result : PipelineResult or None
        Pipeline output.  May be ``None`` when *model* is supplied (the result
        is only consulted by :func:`build_gallery_model`).
    output_dir : str or Path
        Root directory for gallery output.
    model : GalleryModel, optional
        Inventory to render.  Defaults to :func:`build_gallery_model`.
    view : str
        One of ``"l2"``, ``"l3"``, ``"both"`` (default), ``"comparison"``.
    volume_scale : float or None
        Volume display scale, forwarded to :func:`build_gallery_model` when
        *model* is built here.  Ignored when *model* is supplied.
    backends : list of str, optional
        Backends to render.  Defaults to ``["plotly", "matplotlib"]`` when
        plotly is installed (plotly is the primary column), else
        ``["matplotlib"]``.  Pass ``"bokeh"`` explicitly to add a Bokeh
        column -- it is not auto-detected like plotly since it only covers
        the core concepts (``trade_tape``, ``depth_heatmap``,
        ``book_snapshot``, ``depth_chart``); other concepts render as "Not
        available" in that column.  In ``comparison`` view the backend axis
        collapses to a single backend (plotly if available) so the two
        columns carry L2 vs L3.
    title : str
        Gallery page title.

    Returns
    -------
    Path
        Path to the generated HTML gallery file.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if backends is None:
        try:
            import plotly  # noqa: F401

            backends = ["plotly", "matplotlib"]
        except ImportError:
            backends = ["matplotlib"]

    if model is None:
        if result is None:
            raise ConfigError(
                "generate_gallery requires either `result` or an explicit `model`."
            )
        model = build_gallery_model(result, volume_scale=volume_scale)

    cards = _project(model, view, backends)

    # Load each backend the cards draw once, before the placement check reads
    # the registry.  The comparison view draws one backend, so it loads one.
    drawn = list(dict.fromkeys(p.backend for card in cards for p in card.panels))
    unloaded: dict[str, str] = {}
    for backend in drawn:
        try:
            _load_backend(backend)
        except Exception as e:  # noqa: BLE001 -- the other backends still draw
            unloaded[backend] = (
                str(e)  # a name that was never registered: the error names it
                if backend not in _BACKEND_MODULES
                else f"The {backend!r} backend could not be loaded: {e}"
            )
    loaded = frozenset(drawn) - unloaded.keys()

    # Render each panel once and persist it under <backend>/<stem>.{png,html}.
    rendered_dirs: set[str] = set()
    logged: set[str] = set()
    for card in cards:
        logger.info("Gallery: generating {}", card.title)
        # The backend columns of a card share one prepared payload, so the
        # prepare step runs (and logs) once per face, not once per backend.
        prepared: dict[tuple[int, int], dict | Exception] = {}
        for panel in card.panels:
            panel.reason = unloaded.get(panel.backend) or _misplaced(panel, loaded)
            if panel.reason:
                if panel.reason not in logged:
                    logger.warning("Gallery: {}", panel.reason)
                    logged.add(panel.reason)
                continue
            drawn_by = _concepts.backends(panel.concept, panel.level)
            if drawn_by and panel.backend not in drawn_by:
                # The registry says other backends draw this plot and this one
                # does not (the book replay is Plotly only), so this backend is
                # not asked for it and there is no failure to warn about.  When
                # the gallery was not asked for any of those backends, the card
                # says which one would draw it.  (One that failed to load says
                # so in its own column.)
                if not set(drawn) & set(drawn_by):
                    panel.reason = _drawn_elsewhere(drawn_by)
                continue
            if panel.backend not in rendered_dirs:
                (out / panel.backend).mkdir(parents=True, exist_ok=True)
                rendered_dirs.add(panel.backend)
            panel.rendered = _render_and_save(panel, out, plt, prepared)
        if not card.note:
            card.note = _payload_note(prepared)

    html_path = out / "gallery.html"
    _write_gallery_html(html_path, cards, title)
    logger.info("Gallery: {} cards ({} view) saved to {}", len(cards), view, out)
    return html_path


def _misplaced(panel: _Panel, backends: frozenset[str]) -> str:
    """Say why *panel* does not match its registered renderers, or return ``""``.

    The model decides how a panel is drawn: a :class:`PlotConcept` in
    :attr:`GalleryModel.concepts` is drawn at the level of each variant, and a
    :class:`PlotSpec` in :attr:`GalleryModel.analytics` is drawn level-less.
    The concept table (with any plug-in renderers) decides it separately,
    from the levels it declares the plot at.  Only the renderers on *backends*, the gallery's own
    loaded backends, are read, so the answer does not depend on what else was
    imported.  The panel does not match when it is the other kind, or when
    none of *backends* has a renderer at its level.  The text says how to fix
    the model.  A plot drawn by another of *backends* but not by this one is
    not a mismatch, and neither is a plot none of them draws, so both return
    ``""``.
    """
    levels = {lvl for lvl, b in _concepts.placements(panel.concept) if b in backends}
    if not levels:
        return ""
    registered = f"{panel.concept!r} is registered {_kind_text(levels)}"
    if not _same_kind(next(iter(levels)), panel.level):
        if panel.level is None:
            return (
                f"{registered}, but the gallery model has it in analytics, which "
                "is drawn level-less. Add it to GalleryModel.concepts as a "
                "PlotConcept instead."
            )
        return (
            f"{registered}, but the gallery model has it in concepts, which is "
            "drawn at a level. Add it to GalleryModel.analytics as a PlotSpec "
            "instead."
        )
    if panel.level is not None and panel.level not in levels:
        return (
            f"{registered}, but its PlotConcept has a variant at {panel.level}. "
            f"Remove that variant, or register a renderer at {panel.level}."
        )
    return ""


def _drawn_elsewhere(backends: tuple[str, ...]) -> str:
    """Say that only *backends*, none of them in the gallery, draw a plot."""
    names = " or ".join(_backend_label(b) for b in backends)
    verb, them = ("draws", "it") if len(backends) == 1 else ("draw", "them")
    return f"Only {names} {verb} this plot, and this gallery does not use {them}."


def _backend_label(backend: str) -> str:
    style = _BACKEND_STYLES.get(backend)
    return style.label if style else backend.capitalize()


def _payload_note(prepared: dict[tuple[int, int], dict | Exception]) -> str:
    """The first ``note`` a card's prepared payloads carry, or ``""``.

    A prepare step that changed a setting for the gallery says so in its
    payload's ``note``, and the card shows it under the plot.
    """
    for payload in prepared.values():
        if isinstance(payload, dict) and payload.get("note"):
            return str(payload["note"])
    return ""


def _render_and_save(
    panel: _Panel,
    out: Path,
    plt: Any,
    prepared: dict[tuple[int, int], dict | Exception],
) -> bool:
    """Render one panel to a figure and persist it; return success.

    *prepared* caches payloads by (prepare function, keyword arguments) so
    panels of one card that differ only by backend prepare once.  Renderers
    do not modify their payload, so sharing it is safe.  A prepare that
    raises is cached as its error, so it is not run again for the next
    backend.
    """
    key = (id(panel.prepare), id(panel.prep_kwargs))
    if key not in prepared:
        try:
            prepared[key] = panel.prepare(**panel.prep_kwargs)
        except Exception as e:  # noqa: BLE001 -- one bad panel must not sink the gallery
            logger.warning("Gallery: {} {} failed: {}", panel.backend, panel.stem, e)
            prepared[key] = e
    payload = prepared[key]
    if isinstance(payload, Exception):
        # Where a metric is computed, so the card says why it has no plot.
        panel.reason = f"Preparing the data failed: {payload}"
        return False
    try:
        fig = plot(panel.concept, panel.level, backend=panel.backend, **payload)
    except Exception as e:  # noqa: BLE001
        logger.warning("Gallery: {} {} failed: {}", panel.backend, panel.stem, e)
        panel.reason = f"Drawing the plot failed: {e}"
        return False

    target = out / panel.backend / panel.stem
    try:
        if panel.backend == "matplotlib":
            save_figure(fig, f"{target}.png")
            plt.close(fig)
        elif panel.backend == "plotly":
            fig.write_html(f"{target}.html", include_plotlyjs="cdn")
        elif panel.backend == "bokeh":
            from bokeh.io import save as bokeh_save

            bokeh_save(
                fig, filename=f"{target}.html", resources="cdn", title=panel.stem
            )
        else:  # custom backend: best-effort PNG
            save_figure(fig, f"{target}.png")
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Gallery: persisting {} {} failed: {}", panel.backend, panel.stem, e
        )
        panel.reason = f"Saving the plot failed: {e}"
        return False


@dataclass(frozen=True)
class _BackendStyle:
    """Per-backend display metadata for the gallery HTML."""

    label: str  # human-readable name, e.g. "Plotly"
    panel_cls: str  # CSS class for the panel div, e.g. "plotly-panel"


_BACKEND_STYLES: dict[str, _BackendStyle] = {
    "plotly": _BackendStyle("Plotly", "plotly-panel"),
    "matplotlib": _BackendStyle("Matplotlib", "mpl-panel"),
    "bokeh": _BackendStyle("Bokeh", "bokeh-panel"),
}


def _render_panel(panel: _Panel, escaped_title: str) -> str:
    """Render one ``<div class="panel">`` for a card column."""
    classes = f"panel {panel.panel_cls} panel-{panel.role}"

    if not panel.rendered:
        body = '<p class="na">Not available</p>'
        if panel.reason:
            body += f'<p class="na-reason">{html_mod.escape(panel.reason)}</p>'
    elif panel.backend == "plotly":
        body = (
            f'<iframe src="plotly/{panel.stem}.html" loading="lazy" '
            f'title="{escaped_title} ({panel.label})"></iframe>'
        )
    elif panel.backend == "bokeh":
        body = (
            f'<iframe src="bokeh/{panel.stem}.html" loading="lazy" '
            f'title="{escaped_title} ({panel.label})"></iframe>'
        )
    elif panel.backend == "matplotlib":
        body = (
            f'<img src="matplotlib/{panel.stem}.png" alt="{escaped_title}" '
            f'onclick="zoom(this.src)">'
        )
    else:
        body = (
            f'<img src="{panel.backend}/{panel.stem}.png" alt="{escaped_title}" '
            f'onclick="zoom(this.src)">'
        )

    return f'<div class="{classes}"><h3>{panel.label}</h3>{body}</div>'


def _write_gallery_html(
    path: Path,
    cards: list[_Card],
    title: str,
) -> None:
    """Write the standalone HTML gallery page from projected *cards*."""
    rendered_cards: list[str] = []
    for card in cards:
        escaped_title = html_mod.escape(card.title)
        panels = "".join(_render_panel(p, escaped_title) for p in card.panels)
        note_html = (
            f'<div class="card-note">{html_mod.escape(card.note)}</div>'
            if card.note
            else ""
        )
        rendered_cards.append(
            '<div class="card">'
            f'<div class="card-title">{escaped_title}</div>'
            f"{note_html}"
            f'<div class="card-body">{panels}</div></div>'
        )

    escaped_title = html_mod.escape(title)
    content = (
        "\n".join(rendered_cards)
        if rendered_cards
        else ('<p class="empty">No plots for this view.</p>')
    )

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{escaped_title}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:#1a1a2e;color:#e0e0e0;padding:20px}}
h1{{text-align:center;margin-bottom:24px;color:#e94560}}
.card{{margin-bottom:32px;border:1px solid #333;border-radius:8px;
  overflow:hidden;background:#16213e}}
.card-title{{background:#0f3460;padding:12px 20px;font-size:1.1em;
  font-weight:600;color:#e94560}}
.card-note{{padding:8px 20px;font-size:.9em;color:#9fb3c8;background:#10182e;
  border-bottom:1px solid #333;font-style:italic}}
.card-body{{display:flex;gap:0}}
.panel{{text-align:center;padding:10px;min-width:0}}
.panel-primary{{flex:2}}
.panel-secondary{{flex:1}}
.panel-equal{{flex:1}}
.panel+.panel{{border-left:1px solid #333}}
.panel h3{{margin-bottom:8px;font-size:.85em;text-transform:uppercase;letter-spacing:1px}}
.mpl-panel h3{{color:#81c784}}
.plotly-panel h3{{color:#ffb74d}}
.bokeh-panel h3{{color:#4fc3f7}}
.panel img{{max-width:100%;height:auto;border-radius:4px;cursor:pointer;transition:transform .2s}}
.panel img:hover{{transform:scale(1.02)}}
.panel iframe{{width:100%;border:none;border-radius:4px;background:#1e1e1e}}
.panel-primary iframe{{height:600px}}
.panel-secondary iframe{{height:300px}}
.panel-equal iframe{{height:500px}}
.panel .na{{color:#666;font-style:italic;margin-top:40px}}
.panel .na-reason{{color:#9fb3c8;font-size:.9em;margin:8px 16px 0}}
.empty{{text-align:center;color:#888;font-style:italic;margin-top:40px}}
.overlay{{display:none;position:fixed;top:0;left:0;width:100%;height:100%;
  background:rgba(0,0,0,.9);z-index:1000;justify-content:center;
  align-items:center;cursor:zoom-out}}
.overlay.active{{display:flex}}
.overlay img{{max-width:95%;max-height:95%}}
</style>
</head>
<body>
<h1>{escaped_title}</h1>
{content}
<div class="overlay" id="lb" onclick="this.classList.remove('active')">
  <img id="lb-img" src="">
</div>
<script>
function zoom(s){{document.getElementById("lb-img").src=s;document.getElementById("lb").classList.add("active")}}
</script>
</body>
</html>"""

    # The page declares UTF-8, so write it as UTF-8 whatever the locale is.
    path.write_text(html, encoding="utf-8")
