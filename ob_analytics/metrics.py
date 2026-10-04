"""The metric registry and its plug-in discovery.

A metric measures a finished run and draws as a level-less plot.  Every metric
registers itself here under a name, the way sources register with ``SOURCES``
and plot backends with ``RENDERERS``.

Third-party metrics ship in their own package and are found through the
``ob_analytics.metrics`` entry-point group — :func:`load_metric_plugins`
discovers and registers them with no edit to this core.

A registered value is a :class:`~ob_analytics.protocols.Metric` *instance*, not
a class: a metric carries no per-run construction, so the object registered is
the object called.

Five metrics ship with the package and are registered on import:

* ``"l1_ticker"`` — the Level 1 quote: best bid, best ask and last trade.
* ``"vpin"`` — VPIN (:func:`~ob_analytics.flow_toxicity.compute_vpin`).
* ``"kyle_lambda"`` — Kyle's λ
  (:func:`~ob_analytics.flow_toxicity.compute_kyle_lambda`).
* ``"order_flow_imbalance"`` — order flow imbalance per window
  (:func:`~ob_analytics.flow_toxicity.order_flow_imbalance`).
* ``"ofi_horizon"`` — order flow imbalance over several horizons
  (:func:`~ob_analytics.flow_toxicity.ofi_by_horizon`).

Each uses the defaults of the function it wraps.  Pass that function's
settings as keyword arguments to change them, for example
``result.metric("vpin", n_buckets=20)`` or
``result.plot("vpin", n_buckets=20, threshold=0.8)``.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

import pandas as pd
from loguru import logger

from ob_analytics._registry import Registry
from ob_analytics.depth import get_spread
from ob_analytics.flow_toxicity import (
    OFI_HORIZONS,
    KyleLambdaResult,
    compute_kyle_lambda,
    compute_vpin,
    ofi_by_horizon,
    order_flow_imbalance,
)
from ob_analytics.protocols import Level, Metric

#: The entry-point group a third-party package advertises a metric under, e.g.
#: ``[project.entry-points."ob_analytics.metrics"]`` with ``amihud =
#: my_pkg.amihud:AmihudMetric``.
ENTRY_POINT_GROUP = "ob_analytics.metrics"

#: Registry of metric name → :class:`~ob_analytics.protocols.Metric` instance.
METRICS: Registry[str, Metric] = Registry("metric")


def register_metric(metric: Metric) -> None:
    """Register *metric* under its own :attr:`~ob_analytics.protocols.Metric.name`.

    Case-insensitive; overwriting an existing registration is allowed (handy
    for tests and for a plug-in that intentionally shadows a built-in).
    """
    METRICS.register(metric.name, metric)


def list_metrics() -> list[str]:
    """Return a sorted list of registered metric names."""
    return METRICS.list()


def get_metric(name: str) -> Metric:
    """Return the metric registered under *name* (case-insensitive).

    Raises
    ------
    KeyError
        If no metric is registered under *name*; the message lists the
        registered names.
    """
    return METRICS.get(name)


_plugins_loaded = False


def load_metric_plugins(*, force: bool = False) -> list[str]:
    """Discover and register metrics advertised via entry points.

    Scans the :data:`ENTRY_POINT_GROUP` entry-point group; each entry's value
    is loaded to a :class:`~ob_analytics.protocols.Metric` class, instantiated
    with no arguments, and registered under its own ``name``.  This is what
    lets a metric live in a separate installable package without editing
    ob-analytics.

    Idempotent: the scan runs once per process unless *force* is set (the test
    suite forces a re-scan after monkeypatching the entry points).  A plug-in
    that fails to import is logged and skipped, so one broken package cannot
    stop the rest from loading.

    Returns
    -------
    list of str
        The names newly registered by this call.
    """
    global _plugins_loaded
    if _plugins_loaded and not force:
        return []

    registered: list[str] = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            metric = ep.load()()
        except Exception as exc:  # noqa: BLE001 - one bad plug-in must not break the rest
            logger.warning(
                "Metric plug-in {!r} ({}) failed to load: {!r}",
                ep.name,
                getattr(ep, "value", "?"),
                exc,
            )
            continue
        register_metric(metric)
        registered.append(metric.name)
        logger.debug("Registered metric plug-in {!r} from entry point", metric.name)

    _plugins_loaded = True
    return registered


# ---------------------------------------------------------------------------
# Built-in metrics
# ---------------------------------------------------------------------------
#
# Each one wraps a function a user can also call directly.  compute() takes
# that function's settings as keyword arguments, and prepare() takes the
# display settings, so plot_result can send each override to the right one.
# The payload builders are imported inside prepare(): importing the
# visualization package loads matplotlib, which a run that only computes
# metrics does not need.

_EVERY_LEVEL: tuple[Level, ...] = (Level.L2, Level.L3)


class L1TickerMetric:
    """The Level 1 quote: best bid, best ask and last trade through the run.

    The table has one row each time the touch or the last trade changes.  Its
    face draws the three prices over time, or, given ``at=``, the quote card
    for that instant (see
    :func:`~ob_analytics.visualization._data.prepare_l1_ticker_data`).
    """

    name = "l1_ticker"
    title = "Level 1 Quote (best bid, best ask, last trade)"
    levels: tuple[Level, ...] = _EVERY_LEVEL

    def compute(self, result: Any) -> pd.DataFrame:
        """Return ``timestamp``, the touch prices and sizes, and the last trade.

        Columns: ``timestamp``, ``best_bid_price``, ``best_bid_vol``,
        ``best_ask_price``, ``best_ask_vol``, ``last_price``, ``last_volume``.
        Each row carries every value forward from the rows before it, so a
        row is the whole quote at that time.  ``last_price`` is ``NaN`` before
        the first trade, and a side with no orders has a ``NaN`` price and
        size (the depth summary writes ``0`` there).
        """
        quotes = get_spread(result.depth_summary).astype(
            {
                c: float
                for c in (
                    "best_bid_price",
                    "best_bid_vol",
                    "best_ask_price",
                    "best_ask_vol",
                )
            }
        )
        last = result.trades[["timestamp", "price", "volume"]].rename(
            columns={"price": "last_price", "volume": "last_volume"}
        )
        # A quote and a trade at the same time keep that order: stable sort.
        frame = pd.concat([quotes, last], ignore_index=True)
        frame = frame.sort_values("timestamp", kind="stable").ffill()
        # After the fill, so an empty side is carried forward as empty rather
        # than filled from the quote before it.
        for side in ("bid", "ask"):
            empty = frame[f"best_{side}_price"] == 0
            frame.loc[empty, [f"best_{side}_price", f"best_{side}_vol"]] = float("nan")
        return frame.reset_index(drop=True)

    def prepare(self, frame: pd.DataFrame, **options: Any) -> dict[str, Any]:
        """Payload for the face; *options* are those of ``prepare_l1_ticker_data``."""
        from ob_analytics.visualization._data import prepare_l1_ticker_data

        return prepare_l1_ticker_data(frame, **options)


class VpinMetric:
    """VPIN, from :func:`~ob_analytics.flow_toxicity.compute_vpin`."""

    name = "vpin"
    title = "VPIN"
    levels: tuple[Level, ...] = _EVERY_LEVEL

    def compute(
        self,
        result: Any,
        *,
        bucket_volume: float | None = None,
        n_buckets: int = 50,
        sign_method: str | None = None,
    ) -> pd.DataFrame:
        """Return the per-bucket VPIN table; the settings are ``compute_vpin``'s."""
        return compute_vpin(
            result.trades,
            bucket_volume=bucket_volume,
            n_buckets=n_buckets,
            sign_method=sign_method,
        )

    def prepare(self, frame: pd.DataFrame, *, threshold: float = 0.7) -> dict[str, Any]:
        """Payload for the face; *threshold* is the alert line."""
        from ob_analytics.visualization._data import prepare_vpin_data

        return prepare_vpin_data(frame, threshold=threshold)


class KyleLambdaMetric:
    """Kyle's λ, from :func:`~ob_analytics.flow_toxicity.compute_kyle_lambda`.

    The table is the per-window regression data.  The fit itself is in the
    table's ``attrs``: ``lambda_``, ``t_stat``, ``r_squared``, ``n_windows``,
    ``ci_low``, ``ci_high``, ``ci_level`` and ``diagnostics``, as on
    :class:`~ob_analytics.flow_toxicity.KyleLambdaResult`.
    """

    name = "kyle_lambda"
    title = "Kyle's Lambda"
    levels: tuple[Level, ...] = _EVERY_LEVEL

    def compute(
        self,
        result: Any,
        *,
        window: str = "5min",
        n_boot: int = 1000,
        ci_level: float = 0.95,
        seed: int | None = 0,
    ) -> pd.DataFrame:
        """Return the regression table; the settings are ``compute_kyle_lambda``'s."""
        kyle = compute_kyle_lambda(
            result.trades, window, n_boot=n_boot, ci_level=ci_level, seed=seed
        )
        frame = kyle.regression_df.copy()
        frame.attrs.update(
            lambda_=kyle.lambda_,
            t_stat=kyle.t_stat,
            r_squared=kyle.r_squared,
            n_windows=kyle.n_windows,
            ci_low=kyle.ci_low,
            ci_high=kyle.ci_high,
            ci_level=kyle.ci_level,
            diagnostics=kyle.diagnostics,
        )
        return frame

    def prepare(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Payload for the regression scatter."""
        from ob_analytics.visualization._data import prepare_kyle_lambda_data

        fit = frame.attrs
        return prepare_kyle_lambda_data(
            KyleLambdaResult(
                lambda_=fit["lambda_"],
                t_stat=fit["t_stat"],
                r_squared=fit["r_squared"],
                n_windows=fit["n_windows"],
                regression_df=frame,
                ci_low=fit["ci_low"],
                ci_high=fit["ci_high"],
                ci_level=fit["ci_level"],
            )
        )


class OrderFlowImbalanceMetric:
    """Order flow imbalance, from :func:`~ob_analytics.flow_toxicity.order_flow_imbalance`.

    The table adds ``last_price``, the last trade price in each window, which
    the face draws as a price line behind the bars.
    """

    name = "order_flow_imbalance"
    title = "Order Flow Imbalance"
    levels: tuple[Level, ...] = _EVERY_LEVEL

    def compute(
        self,
        result: Any,
        *,
        window: str = "1min",
        sign_method: str | None = None,
    ) -> pd.DataFrame:
        """Return the per-window table; the settings are ``order_flow_imbalance``'s."""
        trades = result.trades
        frame = order_flow_imbalance(trades, window=window, sign_method=sign_method)
        in_time_order = trades.sort_values("timestamp", kind="stable")
        last = in_time_order.groupby(pd.Grouper(key="timestamp", freq=window))[
            "price"
        ].last()
        return frame.merge(
            last.rename("last_price"), left_on="timestamp", right_index=True, how="left"
        )

    def prepare(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Payload for the bars, with the window prices as the price line."""
        from ob_analytics.visualization._data import prepare_ofi_data

        prices = frame[["timestamp", "last_price"]].rename(
            columns={"last_price": "price"}
        )
        return prepare_ofi_data(frame, trades=prices)


class OfiHorizonMetric:
    """Order flow imbalance over several horizons, from :func:`~ob_analytics.flow_toxicity.ofi_by_horizon`."""

    name = "ofi_horizon"
    title = "Order Flow Imbalance vs Horizon"
    levels: tuple[Level, ...] = _EVERY_LEVEL

    def compute(
        self,
        result: Any,
        *,
        horizons: tuple[str, ...] = OFI_HORIZONS,
        grid: str = "5s",
        sign_method: str | None = None,
    ) -> pd.DataFrame:
        """Return the grid table; the settings are ``ofi_by_horizon``'s."""
        return ofi_by_horizon(
            result.trades, horizons=horizons, grid=grid, sign_method=sign_method
        )

    def prepare(
        self,
        frame: pd.DataFrame,
        *,
        start_time: pd.Timestamp | None = None,
        end_time: pd.Timestamp | None = None,
    ) -> dict[str, Any]:
        """Payload for the horizon graph, for the steps from *start_time* to *end_time*."""
        from ob_analytics.visualization._data import prepare_ofi_horizon_grid_data

        return prepare_ofi_horizon_grid_data(
            frame, start_time=start_time, end_time=end_time
        )


for _metric in (
    L1TickerMetric(),
    VpinMetric(),
    KyleLambdaMetric(),
    OrderFlowImbalanceMetric(),
    OfiHorizonMetric(),
):
    register_metric(_metric)
