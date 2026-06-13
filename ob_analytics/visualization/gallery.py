"""Reusable plot gallery generator for ob-analytics.

A :class:`GalleryModel` is the single source of truth: leveled order-book
:class:`PlotConcept` objects (each with up to one :class:`PlotSpec` variant per
:class:`~ob_analytics.visualization.Level`) plus a separate list of level-less
analytic panels.  A *view* (``l2`` / ``l3`` / ``both`` / ``comparison``)
projects that model into gallery cards; :func:`generate_gallery` renders every
selected face across the backend (or L2|L3) column axis, saves the figures, and
writes a standalone HTML page.

Usage::

    from ob_analytics.visualization.gallery import generate_gallery

    gallery_path = generate_gallery(result, "output/gallery/", view="both")
"""

from __future__ import annotations

import html as html_mod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from loguru import logger

from ob_analytics.depth import get_spread
from ob_analytics.analytics import order_book
from ob_analytics.pipeline import PipelineResult
from ob_analytics.visualization import _data as _viz_data
from ob_analytics.visualization import (
    Level,
    infer_volume_scale,
    plot,
    save_figure,
)

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
        Identifier / file stem for analytic panels (e.g. ``"vpin"``).  For a
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


def _auto_zoom_window(
    events: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Derive a sensible (start, end) zoom window from the data's time range."""
    t_min = events["timestamp"].min()
    t_max = events["timestamp"].max()
    duration = t_max - t_min

    if duration < pd.Timedelta(hours=2):
        mid = t_min + duration / 2
        half = duration / 4
        return (mid - half, mid + half)

    quarter = duration / 4
    return (t_min + quarter, t_min + 2 * quarter)


def _l2(
    key: str,
    title: str,
    plot_name: str,
    prepare: Callable[..., dict],
    prep_kwargs: dict[str, Any],
    *,
    note: str = "",
) -> PlotConcept:
    """Build a single-variant (L2-only) concept."""
    spec = PlotSpec(key, title, plot_name, prepare, prep_kwargs)
    return PlotConcept(key, title, {Level.L2: spec}, note=note)


def _l3(
    key: str,
    title: str,
    plot_name: str,
    prepare: Callable[..., dict],
    prep_kwargs: dict[str, Any],
    *,
    note: str = "",
) -> PlotConcept:
    """Build a single-variant (L3-only) concept.

    Used for faces that require persistent order identity and have no aggregate
    (L2/MBP) counterpart -- e.g. competing-risks order outcomes.
    """
    spec = PlotSpec(key, title, plot_name, prepare, prep_kwargs)
    return PlotConcept(key, title, {Level.L3: spec}, note=note)


def _comparable(
    key: str,
    title: str,
    prepare: Callable[..., dict],
    base_kwargs: dict[str, Any],
    *,
    note: str = "",
) -> PlotConcept:
    """Build a comparable (L2 + L3) concept from one prepare + shared kwargs.

    Both faces share *base_kwargs* and differ only in the ``per_order`` flag the
    prepare/renderer pair keys off of: ``False`` aggregates per price (L2/MBP),
    ``True`` keeps one primitive per order (L3/MBO).
    """
    l2 = PlotSpec(key, title, key, prepare, {**base_kwargs, "per_order": False})
    l3 = PlotSpec(key, title, key, prepare, {**base_kwargs, "per_order": True})
    return PlotConcept(key, title, {Level.L2: l2, Level.L3: l3}, note=note)


def _paired(
    key: str,
    title: str,
    l2: PlotSpec,
    l3: PlotSpec,
    *,
    note: str = "",
) -> PlotConcept:
    """Build a comparable concept from two explicit faces.

    Unlike :func:`_comparable` (one prepare keyed by ``per_order``), the L2 and
    L3 faces here use *different* prepares -- e.g. an aggregate volume map (L2)
    paired with a per-order scatter (L3).
    """
    return PlotConcept(key, title, {Level.L2: l2, Level.L3: l3}, note=note)


def build_gallery_model(
    result: PipelineResult,
    *,
    volume_scale: float | None = None,
) -> GalleryModel:
    """Build the default gallery model from pipeline results.

    Every order-book concept is registered with its ``Level.L2`` variant (the
    L3 faces are added by later renderers; a concept becomes
    :attr:`~PlotConcept.comparable` automatically once both exist).  Analytics
    are *not* derived here -- callers append computed analytic panels (built
    with the ``*_panel`` helpers) to :attr:`GalleryModel.analytics`.

    Parameters
    ----------
    result : PipelineResult
        Pipeline output.
    volume_scale : float or None
        Volume display scale.  ``None`` (default) auto-infers a power-of-10
        scale from ``result.events['volume']`` via :func:`infer_volume_scale`.

    Returns
    -------
    GalleryModel
    """
    events = result.events
    trades = result.trades
    depth = result.depth
    depth_summary = result.depth_summary

    if volume_scale is None:
        volume_scale = infer_volume_scale(events["volume"])

    spread = get_spread(depth_summary)
    zoom_start, zoom_end = _auto_zoom_window(events)

    # One shared mid-anchored window feeds every clipped face (roadmap §3.0);
    # faces receiving the same bounds stay comparable on shared axes.
    focus = _viz_data.focus_window(trades)
    price_from = focus.price_from
    price_to = focus.price_to

    offset = events["timestamp"].min() + pd.Timedelta(minutes=1)
    depth_summary_offset = depth_summary[depth_summary["timestamp"] >= offset]

    concepts: list[PlotConcept] = [
        _paired(
            "trade_tape",
            "Trade Tape",
            PlotSpec(
                "trade_tape",
                "Trade Tape (price tape)",
                "trade_tape",
                _viz_data.prepare_trades_data,
                {"trades": trades},
            ),
            PlotSpec(
                "trade_tape",
                "Trade tape with maker order lifecycles",
                "trade_tape",
                _viz_data.prepare_trade_tape_l3_data,
                {"events": events, "trades": trades, "volume_scale": volume_scale},
            ),
            note=(
                "Executions over time. L2: price as a step line. L3: each "
                "execution dot sits at the end of a bar tracing how long the "
                "consumed maker order had rested; color = aggressor side."
            ),
        ),
        _l2(
            "depth_heatmap",
            "Depth Heatmap",
            "depth_heatmap",
            _viz_data.prepare_price_levels_data,
            {
                "depth": depth,
                "spread": spread,
                "trades": trades,
                "volume_scale": volume_scale,
                "price_from": price_from,
                "price_to": price_to,
            },
            note=(
                "Resting liquidity through time: one horizontal line per "
                "price level, colored by available volume; the pale line is "
                "the midprice. Triangles mark executions (aggressor side). "
                "Gaps mean the level emptied. Pass col_bias<1 to brighten "
                "thin levels and reveal near-touch structure."
            ),
        ),
        _paired(
            "order_activity",
            "Order Activity",
            PlotSpec(
                "order_activity",
                "Order Activity (event map)",
                "order_activity",
                _viz_data.prepare_event_map_data,
                # Share the depth heatmap's mid-anchored window so the L2 event
                # map clips around the touch rather than its own raw-price
                # percentile (far flashed orders otherwise squish the activity).
                {
                    "events": events,
                    "volume_scale": volume_scale,
                    "price_from": price_from,
                    "price_to": price_to,
                },
            ),
            PlotSpec(
                "order_activity",
                "Order lifecycles (place → outcome)",
                "order_activity",
                _viz_data.prepare_order_activity_l3_data,
                # Reuses the depth heatmap's trades-median ±3σ window so the
                # Gantt clips around the touch.  A shared FocusWindow primitive
                # replaces these per-face clips in roadmap §3.0 (docs/plans/).
                {
                    "events": events,
                    "volume_scale": volume_scale,
                    "price_from": price_from,
                    "price_to": price_to,
                },
            ),
            note=(
                "Order placement and removal. L2: created/deleted events "
                "scattered at their price. L3: each order is one lifespan "
                "from placement to outcome - orange = pulled (flashed), "
                "green = rested/filled."
            ),
        ),
        _paired(
            "cancellations",
            "Cancellations",
            PlotSpec(
                "cancellations",
                "Cancellations (log)",
                "cancellations",
                _viz_data.prepare_volume_map_data,
                {"events": events, "volume_scale": volume_scale, "log_scale": True},
            ),
            PlotSpec(
                "cancellations",
                "Cancellations (age x distance)",
                "cancellations",
                _viz_data.prepare_cancellations_l3_data,
                {"events": events, "volume_scale": volume_scale},
            ),
            note=(
                "Pulled orders. L2: cancelled volume over time (log scale). "
                "L3: every cancellation as how-long-it-rested vs "
                "how-far-from-the-touch-it-sat, sized by volume."
            ),
        ),
    ]

    # Book snapshot + depth chart share one order-book slice; each ships an
    # L2 (aggregate-per-price) and L3 (per-order) face, so both are comparable.
    # Requires the per-order 'type' column from set_order_types.
    if "type" in events.columns:
        snap_time = zoom_end
        snap_book = order_book(events, tp=snap_time, bps_range=150)
        snap_label = snap_time.strftime("%H:%M")
        concepts.append(
            _comparable(
                "book_snapshot",
                f"Book Snapshot ({snap_label})",
                _viz_data.prepare_book_snapshot_data,
                {"order_book": snap_book, "volume_scale": volume_scale},
                note=(
                    "The resting book at one instant. L2: total size per "
                    "price level. L3: the same bars segmented per order - "
                    "one whale and a crowd of small orders look identical "
                    "on L2 and completely different here."
                ),
            )
        )
        concepts.append(
            _comparable(
                "depth_chart",
                f"Depth Chart ({snap_label})",
                _viz_data.prepare_book_snapshot_data,
                {"order_book": snap_book, "volume_scale": volume_scale},
                note=(
                    "Cumulative liquidity walking away from the touch: the "
                    "cost to sweep X units. L3 marks each individual order "
                    "along the curve."
                ),
            )
        )

    # Order outcome is L3-only: it asks where each *order* was placed (distance
    # from the touch, from order_aggressiveness) and how it ended (competing-risks
    # fate).  No aggregate counterpart, and it needs the per-event bps column.
    if "aggressiveness_bps" in events.columns:
        concepts.append(
            _l3(
                "order_outcome",
                "Order Outcome",
                "order_outcome",
                _viz_data.prepare_order_outcome_l3_data,
                {"events": events, "volume_scale": volume_scale},
                note=(
                    "Where orders were placed vs how they ended. x = signed "
                    "distance from the best price at placement (right of 0 "
                    "improved the touch), y = size; green filled, purple "
                    "partial, orange cancelled. Still-resting orders are "
                    "not shown."
                ),
            )
        )

    if not depth_summary_offset.empty:
        # Liquidity at the touch is L2 (MBP): best bid/ask resting size over time.
        # The L3 counterpart needs FIFO queue reconstruction and is deferred, so
        # this concept ships single-variant (not comparable) for now.
        concepts.append(
            _l2(
                "liquidity_at_touch",
                "Liquidity at the Touch",
                "liquidity_at_touch",
                _viz_data.prepare_liquidity_at_touch_data,
                {
                    "depth_summary": depth_summary_offset,
                    "start_time": zoom_start,
                    "end_time": zoom_end,
                    "volume_scale": volume_scale,
                },
                note=(
                    "Resting size at the best bid and best ask over a "
                    "zoomed window - the liquidity a marketable order "
                    "meets first."
                ),
            )
        )
        concepts.append(
            _l2(
                "volume_percentiles",
                "Volume Percentiles",
                "volume_percentiles",
                _viz_data.prepare_volume_percentiles_data,
                {
                    "depth_summary": depth_summary_offset,
                    "start_time": zoom_start,
                    "end_time": zoom_end,
                    "volume_scale": volume_scale,
                },
                note=(
                    "Book depth through time, stacked by distance from the "
                    "touch in bps bands: asks above zero, bids below; band "
                    "thickness = resting volume in that bps ring."
                ),
            )
        )

    # Events histogram (price). Clip to the shared mid-anchored focus window so
    # the near-touch distribution stays legible: even q01-q99 of a heavy-tailed
    # book still spans the far-from-touch flashed orders, collapsing the face to
    # a single 1px spike.  The prepare clips; here we only size the bandwidth.
    hist_events = events[["timestamp", "direction", "price", "volume"]].copy()
    hist_events["volume"] = hist_events["volume"] * volume_scale
    if price_from is not None and price_to is not None:
        price_window = price_to - price_from
    else:  # no focus window (degenerate trades): fall back to the tail span
        price_window = hist_events["price"].quantile(0.99) - hist_events[
            "price"
        ].quantile(0.01)
    price_bw = max(0.01, round(price_window / 100, 2))
    concepts.append(
        _l2(
            "events_histogram",
            "Events Histogram (price)",
            "events_histogram",
            _viz_data.prepare_events_histogram_data,
            {
                "events": hist_events,
                "val": "price",
                "bw": price_bw,
                "price_from": price_from,
                "price_to": price_to,
            },
            note=(
                "Where order activity concentrated: event counts binned by "
                "price within the focus window, split by side."
            ),
        )
    )

    # Hidden executions are LOBSTER-only (raw_event_type == 5).
    if "raw_event_type" in events.columns:
        hidden = events[events["raw_event_type"] == 5]
        if not hidden.empty:
            concepts.append(
                _l2(
                    "hidden_executions",
                    "Hidden Executions",
                    "hidden_executions",
                    _viz_data.prepare_hidden_executions_data,
                    {"events": events, "trades": trades},
                    note=(
                        "LOBSTER-only: executions against hidden orders "
                        "(type 5) scattered over the trade price line - "
                        "liquidity that never showed in the visible book."
                    ),
                )
            )

    return GalleryModel(concepts=concepts, analytics=[])


def vpin_panel(vpin_df: pd.DataFrame, *, threshold: float = 0.7) -> PlotSpec:
    """Build a VPIN analytic panel for :attr:`GalleryModel.analytics`."""
    return PlotSpec(
        "vpin",
        "VPIN",
        "vpin",
        _viz_data.prepare_vpin_data,
        {"vpin_df": vpin_df, "threshold": threshold},
    )


def ofi_panel(ofi_df: pd.DataFrame, trades: pd.DataFrame | None = None) -> PlotSpec:
    """Build an order-flow-imbalance analytic panel."""
    return PlotSpec(
        "ofi",
        "Order Flow Imbalance",
        "order_flow_imbalance",
        _viz_data.prepare_ofi_data,
        {"ofi_df": ofi_df, "trades": trades},
    )


def kyle_panel(kyle_result: Any) -> PlotSpec:
    """Build a Kyle's-Lambda analytic panel."""
    return PlotSpec(
        "kyle_lambda",
        f"Kyle's Lambda (lambda={kyle_result.lambda_:.4f})",
        "kyle_lambda",
        _viz_data.prepare_kyle_lambda_data,
        {"kyle_result": kyle_result},
    )


def trading_halts_panel(trades: pd.DataFrame, halts: pd.DataFrame) -> PlotSpec:
    """Build a trading-halts analytic panel.

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
    (``backend``/``stem``/``label``/...) drive the HTML.  ``rendered`` is set by
    the render loop and consumed by :func:`_render_panel`.
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
        style = _BACKEND_STYLES.get(backend)
        panels.append(
            _Panel(
                concept=concept,
                level=level,
                prepare=spec.prepare,
                prep_kwargs=spec.prep_kwargs,
                backend=backend,
                stem=stem,
                label=style.label if style else backend.capitalize(),
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
        raise ValueError(f"Unknown view {view!r}. Available: {list(VIEWS)}")

    if view == "comparison":
        backend = _comparison_backend(backends)
        cards: list[_Card] = []
        for concept in model.concepts:
            if not concept.comparable:
                continue
            panels = []
            for level in (Level.L2, Level.L3):
                spec = concept.at(level)
                assert spec is not None  # comparable => both variants exist
                panels.append(
                    _Panel(
                        concept=concept.key,
                        level=level,
                        prepare=spec.prepare,
                        prep_kwargs=spec.prep_kwargs,
                        backend=backend,
                        stem=f"{concept.key}.{level}",
                        label=_LEVEL_LABEL[level],
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
        ``["matplotlib"]``.  In ``comparison`` view the backend axis collapses
        to a single backend (plotly if available) so the two columns carry
        L2 vs L3.
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
            raise ValueError(
                "generate_gallery requires either `result` or an explicit `model`."
            )
        model = build_gallery_model(result, volume_scale=volume_scale)

    cards = _project(model, view, backends)

    # Render each panel once and persist it under <backend>/<stem>.{png,html}.
    rendered_dirs: set[str] = set()
    for card in cards:
        logger.info("Gallery: generating {}", card.title)
        for panel in card.panels:
            if panel.backend not in rendered_dirs:
                (out / panel.backend).mkdir(parents=True, exist_ok=True)
                rendered_dirs.add(panel.backend)
            panel.rendered = _render_and_save(panel, out, plt)

    html_path = out / "gallery.html"
    _write_gallery_html(html_path, cards, title)
    logger.info("Gallery: {} cards ({} view) saved to {}", len(cards), view, out)
    return html_path


def _render_and_save(panel: _Panel, out: Path, plt: Any) -> bool:
    """Render one panel to a figure and persist it; return success."""
    try:
        data = panel.prepare(**panel.prep_kwargs)
        fig = plot(panel.concept, panel.level, backend=panel.backend, **data)
    except Exception as e:  # noqa: BLE001 -- one bad panel must not sink the gallery
        logger.warning("Gallery: {} {} failed: {}", panel.backend, panel.stem, e)
        return False

    target = out / panel.backend / panel.stem
    try:
        if panel.backend == "matplotlib":
            save_figure(fig, f"{target}.png")
            plt.close(fig)
        elif panel.backend == "plotly":
            fig.write_html(f"{target}.html", include_plotlyjs="cdn")
        else:  # custom backend: best-effort PNG
            save_figure(fig, f"{target}.png")
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Gallery: persisting {} {} failed: {}", panel.backend, panel.stem, e
        )
        return False


@dataclass(frozen=True)
class _BackendStyle:
    """Per-backend display metadata for the gallery HTML."""

    label: str  # human-readable name, e.g. "Plotly"
    panel_cls: str  # CSS class for the panel div, e.g. "plotly-panel"


_BACKEND_STYLES: dict[str, _BackendStyle] = {
    "plotly": _BackendStyle("Plotly", "plotly-panel"),
    "matplotlib": _BackendStyle("Matplotlib", "mpl-panel"),
}


def _render_panel(panel: _Panel, escaped_title: str) -> str:
    """Render one ``<div class="panel">`` for a card column."""
    classes = f"panel {panel.panel_cls} panel-{panel.role}"

    if not panel.rendered:
        body = '<p class="na">Not available</p>'
    elif panel.backend == "plotly":
        body = (
            f'<iframe src="plotly/{panel.stem}.html" loading="lazy" '
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
.panel img{{max-width:100%;height:auto;border-radius:4px;cursor:pointer;transition:transform .2s}}
.panel img:hover{{transform:scale(1.02)}}
.panel iframe{{width:100%;border:none;border-radius:4px;background:#1e1e1e}}
.panel-primary iframe{{height:600px}}
.panel-secondary iframe{{height:300px}}
.panel-equal iframe{{height:500px}}
.panel .na{{color:#666;font-style:italic;margin-top:40px}}
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

    path.write_text(html)
