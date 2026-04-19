"""Reusable plot gallery generator for ob-analytics.

Generates all standard plots across matplotlib and plotly backends,
saves them to disk, and produces a standalone HTML gallery page.

Usage::

    from ob_analytics.visualization.gallery import generate_gallery

    gallery_path = generate_gallery(result, "output/gallery/")
"""

from __future__ import annotations

import html as html_mod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from loguru import logger

from ob_analytics.depth import get_spread
from ob_analytics.flow_toxicity import (
    compute_kyle_lambda,
    compute_vpin,
    order_flow_imbalance,
)
from ob_analytics.analytics import order_book
from ob_analytics.pipeline import PipelineResult
from ob_analytics.visualization import (
    infer_volume_scale,
    plot_current_depth,
    plot_event_map,
    plot_events_histogram,
    plot_hidden_executions,
    plot_kyle_lambda,
    plot_order_flow_imbalance,
    plot_price_levels,
    plot_trades,
    plot_trading_halts,
    plot_volume_map,
    plot_volume_percentiles,
    plot_vpin,
    save_figure,
)


@dataclass
class PlotSpec:
    """Specification for a single plot in the gallery.

    Parameters
    ----------
    name : str
        File stem, e.g. ``"01_trades_full"``.
    title : str
        Human-readable title for the gallery card.
    plot_fn : Callable
        The ``plot_*`` function from ``ob_analytics.visualization``.
    kwargs : dict
        Keyword arguments to pass to *plot_fn*.
    needs : str or None
        If set, the plot is skipped when this key is missing or empty
        in the data context (e.g. ``"vpin"`` for VPIN-dependent plots).
    """

    name: str
    title: str
    plot_fn: Callable[..., Any]
    kwargs: dict[str, Any] = field(default_factory=dict)
    needs: str | None = None


def _auto_time_slices(
    events: pd.DataFrame,
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Derive sensible zoom windows from the data's time range."""
    t_min = events["timestamp"].min()
    t_max = events["timestamp"].max()
    duration = t_max - t_min

    if duration < pd.Timedelta(hours=2):
        mid = t_min + duration / 2
        half = duration / 4
        return {"zoom": (mid - half, mid + half)}

    quarter = duration / 4
    return {
        "zoom": (t_min + quarter, t_min + 2 * quarter),
    }


def default_specs(
    result: PipelineResult,
    *,
    volume_scale: float | None = None,
    vpin_bucket_volume: float | None = None,
) -> list[PlotSpec]:
    """Build the default set of plot specifications from pipeline results.

    Parameters
    ----------
    result : PipelineResult
        Pipeline output.
    volume_scale : float or None
        Scaling factor for volume display.  ``None`` (default)
        auto-infers a power-of-10 scale from ``result.events['volume']``
        via :func:`infer_volume_scale` so the gallery works without
        per-asset tuning.
    vpin_bucket_volume : float or None
        Bucket volume for VPIN computation.  When None, VPIN/OFI/Kyle
        plots are skipped.

    Returns
    -------
    list of PlotSpec
    """
    events = result.events
    trades = result.trades
    depth = result.depth
    depth_summary = result.depth_summary

    if volume_scale is None:
        volume_scale = infer_volume_scale(events["volume"])

    spread = get_spread(depth_summary)
    slices = _auto_time_slices(events)
    zoom_start, zoom_end = slices["zoom"]

    price_mid = trades["price"].median()
    price_std = trades["price"].std()
    price_from = max(0, price_mid - 3 * price_std) if price_std > 0 else None

    offset = events["timestamp"].min() + pd.Timedelta(minutes=1)
    depth_summary_offset = depth_summary[depth_summary["timestamp"] >= offset]

    specs: list[PlotSpec] = [
        PlotSpec(
            "01_trades_full",
            "Trades (Full Range)",
            plot_trades,
            {"trades": trades},
        ),
        PlotSpec(
            "02_trades_zoom",
            "Trades (Zoomed)",
            plot_trades,
            {"trades": trades, "start_time": zoom_start, "end_time": zoom_end},
        ),
        PlotSpec(
            "03_price_levels_full",
            "Price Levels / Depth Heatmap (Full)",
            plot_price_levels,
            {
                "depth": depth,
                "spread": spread,
                "volume_scale": volume_scale,
                "col_bias": 0.1,
                "price_from": price_from,
            },
        ),
        PlotSpec(
            "04_price_levels_zoom",
            "Price Levels (Zoomed + Trades)",
            plot_price_levels,
            {
                "depth": depth,
                "spread": spread,
                "trades": trades,
                "start_time": zoom_start,
                "end_time": zoom_end,
                "volume_scale": volume_scale,
            },
        ),
        PlotSpec(
            "05_event_map_full",
            "Event Map (Full Range)",
            plot_event_map,
            {"events": events, "volume_scale": volume_scale},
        ),
        PlotSpec(
            "06_event_map_zoom",
            "Event Map (Zoomed)",
            plot_event_map,
            {
                "events": events,
                "start_time": zoom_start,
                "end_time": zoom_end,
                "volume_scale": volume_scale,
            },
        ),
        PlotSpec(
            "07_volume_map_deleted",
            "Volume Map -- Cancelled (log)",
            plot_volume_map,
            {
                "events": events,
                "volume_scale": volume_scale,
                "log_scale": True,
            },
        ),
        PlotSpec(
            "08_volume_map_created",
            "Volume Map -- Created (log)",
            plot_volume_map,
            {
                "events": events,
                "action": "created",
                "volume_scale": volume_scale,
                "log_scale": True,
            },
        ),
    ]

    # Hidden executions and trading halts (LOBSTER-enriched, degrade gracefully)
    specs.append(
        PlotSpec(
            "08b_hidden_executions",
            "Hidden Order Executions",
            plot_hidden_executions,
            {"events": events, "trades": trades},
        )
    )
    specs.append(
        PlotSpec(
            "08c_trading_halts",
            "Trading Halts",
            plot_trading_halts,
            {"trades": trades, "events": events},
        )
    )

    # Order book snapshot (requires 'type' column from set_order_types)
    if "type" in events.columns:
        snap_time = zoom_end
        specs.append(
            PlotSpec(
                "09_current_depth",
                f"Current Depth ({snap_time.strftime('%H:%M')})",
                plot_current_depth,
                {
                    "order_book": order_book(events, tp=snap_time, bps_range=150),
                    "volume_scale": volume_scale,
                },
            )
        )

    # Volume percentiles
    if not depth_summary_offset.empty:
        specs.append(
            PlotSpec(
                "10_volume_percentiles",
                "Volume Percentiles",
                plot_volume_percentiles,
                {
                    "depth_summary": depth_summary_offset,
                    "start_time": zoom_start,
                    "end_time": zoom_end,
                    "volume_scale": volume_scale,
                },
            )
        )

    # Event histograms
    hist_events = events[["timestamp", "direction", "price", "volume"]].copy()
    hist_events["volume"] = hist_events["volume"] * volume_scale
    q01 = hist_events["price"].quantile(0.01)
    q99 = hist_events["price"].quantile(0.99)
    hist_price = hist_events[
        (hist_events["price"] >= q01) & (hist_events["price"] <= q99)
    ]
    price_range = q99 - q01
    price_bw = max(0.01, round(price_range / 100, 2))

    specs.extend(
        [
            PlotSpec(
                "11_events_hist_price",
                "Events Histogram -- Price",
                plot_events_histogram,
                {"events": hist_price, "val": "price", "bw": price_bw},
            ),
            PlotSpec(
                "12_events_hist_volume",
                "Events Histogram -- Volume",
                plot_events_histogram,
                {
                    "events": hist_events[
                        hist_events["volume"] < hist_events["volume"].quantile(0.99)
                    ],
                    "val": "volume",
                    "bw": max(
                        0.01,
                        round(hist_events["volume"].quantile(0.99) / 20, 2),
                    ),
                },
            ),
        ]
    )

    # Flow toxicity plots
    if vpin_bucket_volume is not None and len(trades) > 10:
        try:
            vpin_df = compute_vpin(trades, vpin_bucket_volume)
            specs.append(
                PlotSpec(
                    "13_vpin",
                    "VPIN",
                    plot_vpin,
                    {"vpin_df": vpin_df, "threshold": 0.7},
                )
            )
        except Exception:
            logger.warning("Could not compute VPIN -- skipping plot")

        try:
            ofi_df = order_flow_imbalance(trades, window="5min")
            specs.append(
                PlotSpec(
                    "14_ofi",
                    "Order Flow Imbalance",
                    plot_order_flow_imbalance,
                    {"ofi_df": ofi_df, "trades": trades},
                )
            )
        except Exception:
            logger.warning("Could not compute OFI -- skipping plot")

        try:
            kyle = compute_kyle_lambda(trades, window="5min")
            specs.append(
                PlotSpec(
                    "15_kyle_lambda",
                    f"Kyle's Lambda (lambda={kyle.lambda_:.4f})",
                    plot_kyle_lambda,
                    {"kyle_result": kyle},
                )
            )
        except Exception:
            logger.warning("Could not compute Kyle's lambda -- skipping plot")

    return specs


def generate_gallery(
    result: PipelineResult | None,
    output_dir: str | Path,
    *,
    specs: list[PlotSpec] | None = None,
    volume_scale: float | None = None,
    vpin_bucket_volume: float | None = None,
    backends: list[str] | None = None,
    title: str = "ob-analytics Plot Gallery",
) -> Path:
    """Generate all plots and a standalone HTML gallery.

    Parameters
    ----------
    result : PipelineResult or None
        Pipeline output.  May be ``None`` when *specs* is provided
        explicitly (the result is only consulted by :func:`default_specs`).
    output_dir : str or Path
        Root directory for gallery output.
    specs : list of PlotSpec, optional
        Plot specifications.  Defaults to :func:`default_specs`.
    volume_scale : float or None
        Volume display scale factor.  ``None`` (default) auto-infers a
        sensible power-of-10 scale from the events.
    vpin_bucket_volume : float or None
        Bucket volume for VPIN.  None skips flow toxicity plots.
    backends : list of str, optional
        Backends to render.  Defaults to ``["plotly", "matplotlib"]``
        when ``plotly`` is installed (Plotly is rendered as the primary
        column), or ``["matplotlib"]`` otherwise.
    title : str
        Gallery page title.

    Returns
    -------
    Path
        Path to the generated HTML gallery file.

    Notes
    -----
    When ``backends`` is left at the default and ``plotly`` is installed,
    the gallery renders Plotly **first** (the primary/larger column in the
    HTML layout) so the interactive view is front-and-center.  Pass
    ``backends=["matplotlib"]`` to opt out.
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

    if specs is None:
        if result is None:
            raise ValueError(
                "generate_gallery requires either `result` or explicit `specs`."
            )
        specs = default_specs(
            result,
            volume_scale=volume_scale,
            vpin_bucket_volume=vpin_bucket_volume,
        )

    backend_dirs = {b: out / b for b in backends}

    # generated[i] = (name, title, {backend: success})
    generated: list[tuple[str, str, dict[str, bool]]] = []

    for spec in specs:
        logger.info("Gallery: generating {}", spec.name)
        statuses: dict[str, bool] = {}

        for backend in backends:
            backend_dirs[backend].mkdir(parents=True, exist_ok=True)
            try:
                fig = spec.plot_fn(**spec.kwargs, backend=backend)
            except Exception as e:
                logger.warning("Gallery: {} {} failed: {}", backend, spec.name, e)
                statuses[backend] = False
                continue

            try:
                if backend == "matplotlib":
                    save_figure(fig, str(backend_dirs[backend] / f"{spec.name}.png"))
                    plt.close(fig)
                elif backend == "plotly":
                    fig.write_html(
                        str(backend_dirs[backend] / f"{spec.name}.html"),
                        include_plotlyjs="cdn",
                    )
                else:  # custom backend: best-effort
                    save_figure(fig, str(backend_dirs[backend] / f"{spec.name}.png"))
                statuses[backend] = True
            except Exception as e:
                logger.warning(
                    "Gallery: persisting {} {} failed: {}", backend, spec.name, e
                )
                statuses[backend] = False

        generated.append((spec.name, spec.title, statuses))

    html_path = out / "gallery.html"
    _write_gallery_html(html_path, generated, title, backends)
    logger.info("Gallery: {} plots saved to {}", len(generated), out)
    return html_path


_BACKEND_LABELS = {"plotly": "Plotly", "matplotlib": "Matplotlib"}
_PANEL_CLASSES = {"plotly": "plotly-panel", "matplotlib": "mpl-panel"}


def _render_panel(
    backend: str,
    name: str,
    rendered: bool,
    role: str,
    escaped_title: str,
) -> str:
    """Render one ``<div class="panel">`` for a given backend.

    *role* is ``"primary"`` for the first listed backend (larger column)
    and ``"secondary"`` for the rest.
    """
    label = _BACKEND_LABELS.get(backend, backend.capitalize())
    panel_cls = _PANEL_CLASSES.get(backend, f"{backend}-panel")
    classes = f"panel {panel_cls} panel-{role}"

    if not rendered:
        body = '<p class="na">Not available</p>'
    elif backend == "plotly":
        body = (
            f'<iframe src="plotly/{name}.html" loading="lazy" '
            f'title="{escaped_title} (Plotly)"></iframe>'
        )
    elif backend == "matplotlib":
        body = (
            f'<img src="matplotlib/{name}.png" alt="{escaped_title}" '
            f'onclick="zoom(this.src)">'
        )
    else:
        body = (
            f'<img src="{backend}/{name}.png" alt="{escaped_title}" '
            f'onclick="zoom(this.src)">'
        )

    return f'<div class="{classes}"><h3>{label}</h3>{body}</div>'


def _write_gallery_html(
    path: Path,
    plots: list[tuple[str, str, dict[str, bool]]],
    title: str,
    backends: list[str],
) -> None:
    """Write the standalone HTML gallery page.

    The first entry in *backends* is rendered as the primary (wider)
    panel; subsequent backends become secondary panels.
    """
    cards: list[str] = []
    for name, plot_title, statuses in plots:
        escaped_title = html_mod.escape(plot_title)
        panels = [
            _render_panel(
                backend,
                name,
                statuses.get(backend, False),
                "primary" if i == 0 else "secondary",
                escaped_title,
            )
            for i, backend in enumerate(backends)
        ]
        cards.append(
            '<div class="card">'
            f'<div class="card-title">{escaped_title}</div>'
            '<div class="card-body">' + "".join(panels) + "</div></div>"
        )

    escaped_title = html_mod.escape(title)
    content = "\n".join(cards)

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
.card-body{{display:flex;gap:0}}
.panel{{text-align:center;padding:10px;min-width:0}}
.panel-primary{{flex:2}}
.panel-secondary{{flex:1}}
.panel+.panel{{border-left:1px solid #333}}
.panel h3{{margin-bottom:8px;font-size:.85em;text-transform:uppercase;letter-spacing:1px}}
.mpl-panel h3{{color:#81c784}}
.plotly-panel h3{{color:#ffb74d}}
.panel img{{max-width:100%;height:auto;border-radius:4px;cursor:pointer;transition:transform .2s}}
.panel img:hover{{transform:scale(1.02)}}
.panel iframe{{width:100%;border:none;border-radius:4px;background:#1e1e1e}}
.panel-primary iframe{{height:600px}}
.panel-secondary iframe{{height:300px}}
.panel .na{{color:#666;font-style:italic;margin-top:40px}}
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
