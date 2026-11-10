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

Nine metrics ship with the package and are registered on import:

* ``"l1_ticker"`` — the Level 1 quote: best bid, best ask and last trade.
* ``"book_signals"`` — mid, micro-price and order-book imbalance
  (:func:`~ob_analytics.depth.depth_signals`).
* ``"vpin"`` — VPIN (:func:`~ob_analytics.flow_toxicity.compute_vpin`).
* ``"kyle_lambda"`` — Kyle's λ
  (:func:`~ob_analytics.flow_toxicity.compute_kyle_lambda`).
* ``"order_flow_imbalance"`` — order flow imbalance per window
  (:func:`~ob_analytics.flow_toxicity.order_flow_imbalance`).
* ``"ofi_horizon"`` — order flow imbalance over several horizons
  (:func:`~ob_analytics.flow_toxicity.ofi_by_horizon`).
* ``"transaction_costs"`` — effective and realized spread and price impact
  per trade (:func:`~ob_analytics.cost.transaction_costs`).
* ``"amihud"`` — Amihud illiquidity (:func:`~ob_analytics.cost.amihud`).
* ``"roll_spread"`` — Roll's implied spread
  (:func:`~ob_analytics.cost.roll_spread`).

Each uses the defaults of the function it wraps, and the run's own
``depth_summary`` wherever that function takes quotes.  Pass that function's
settings as keyword arguments to change them, for example
``result.metric("vpin", n_buckets=20)`` or
``result.plot("vpin", n_buckets=20, threshold=0.8)``.

Units.  Every metric reports in the instrument's own units — prices in the
quote currency and sizes in BTC, shares or contracts — and so does every
function it wraps, whether it is given a run's integer ticks and lots or
display prices and sizes.  ``result.metric(name)``, ``result.plot(name)`` and
the bare function therefore give the same numbers.  A size setting is in the
instrument's own units too, so ``bucket_volume=0.5`` means 0.5 BTC in all
three.
"""

from __future__ import annotations

from importlib.metadata import EntryPoint
from typing import Any, Literal

import pandas as pd

from ob_analytics._registry import Registry, load_plugins
from ob_analytics._utils import price_scale, size_scale
from ob_analytics.cost import amihud, roll_spread, transaction_costs
from ob_analytics.depth import _quoted, depth_signals, get_spread, imbalance_levels
from ob_analytics.flow_toxicity import (
    OFI_HORIZONS,
    KyleLambdaResult,
    _ofi_windows,
    compute_kyle_lambda,
    compute_vpin,
    ofi_by_horizon,
)
from ob_analytics.protocols import Level, Metric

#: The entry-point group a third-party package advertises a metric under, e.g.
#: ``[project.entry-points."ob_analytics.metrics"]`` with ``amihud =
#: my_pkg.amihud:AmihudMetric``.
ENTRY_POINT_GROUP = "ob_analytics.metrics"

#: The members of the :class:`~ob_analytics.protocols.Metric` protocol, named
#: in the error for a metric that lacks one.
_METRIC_MEMBERS = ("name", "title", "levels", "needs", "compute", "prepare")

#: Registry of metric name → :class:`~ob_analytics.protocols.Metric` instance.
METRICS: Registry[str, Metric] = Registry("metric")


def register_metric(metric: Metric) -> None:
    """Register *metric* under its own :attr:`~ob_analytics.protocols.Metric.name`.

    Case-insensitive; overwriting an existing registration is allowed (handy
    for tests and for a plug-in that intentionally shadows a built-in).

    Raises
    ------
    TypeError
        If *metric* lacks a member of the
        :class:`~ob_analytics.protocols.Metric` protocol, such as ``needs``.
        Checked here, so one incomplete plug-in fails when it loads instead
        of breaking every metric a run computes.
    """
    if not isinstance(metric, Metric):
        missing = [m for m in _METRIC_MEMBERS if not hasattr(metric, m)]
        raise TypeError(
            f"{type(metric).__name__} is not a Metric: it lacks "
            f"{', '.join(missing)}. A metric has {', '.join(_METRIC_MEMBERS)}."
        )
    METRICS.register(metric.name, metric)


def list_metrics() -> list[str]:
    """Return a sorted list of registered metric names."""
    return METRICS.list()


def get_metric(name: str) -> Metric:
    """Return the metric registered under *name* (case-insensitive).

    Raises
    ------
    UnknownNameError
        If no metric is registered under *name*; the message names
        the registered names closest to it.
    """
    return METRICS.get(name)


def skip_reason(metric: Metric, result: Any) -> str | None:
    """Return why *metric* does not apply to *result*, or ``None`` when it does.

    A metric applies when *result* is at one of its
    :attr:`~ob_analytics.protocols.Metric.levels` and every table it
    :attr:`~ob_analytics.protocols.Metric.needs` has rows.  This is the one
    test :meth:`~ob_analytics.pipeline.PipelineResult.metrics`, the gallery
    and :func:`~ob_analytics.visualization.plot_result` use.
    """
    if result.level not in metric.levels:
        drawn = " and ".join(str(level) for level in metric.levels)
        return f"it applies at {drawn}, and this run is {result.level}"
    empty = [name for name in metric.needs if getattr(result, name).empty]
    if empty:
        return f"it needs {' and '.join(empty)}, and this run has none"
    return None


def applicable_metrics(result: Any) -> list[Metric]:
    """Return the registered metrics that apply to *result*, by name.

    See :func:`skip_reason` for what applies.
    """
    found = (METRICS.get(name) for name in METRICS.list())
    return [metric for metric in found if skip_reason(metric, result) is None]


def _register_entry_point(ep: EntryPoint) -> str:
    metric = ep.load()()
    register_metric(metric)
    return metric.name


def load_metric_plugins(*, force: bool = False) -> list[str]:
    """Discover and register metrics advertised via entry points.

    Scans the :data:`ENTRY_POINT_GROUP` entry-point group; each entry's value
    is loaded to a :class:`~ob_analytics.protocols.Metric` class, instantiated
    with no arguments, and registered under its own ``name``.  This is what
    lets a metric live in a separate installable package without editing
    ob-analytics.

    The scan runs once per process unless *force* is set.  A plug-in that
    fails to load is skipped with a warning, so one broken package cannot stop
    the rest from loading.

    Returns
    -------
    list of str
        The names newly registered by this call.
    """
    return load_plugins(ENTRY_POINT_GROUP, _register_entry_point, force=force)


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


def _in_units(
    frame: pd.DataFrame,
    units_of: pd.DataFrame,
    prices: tuple[str, ...],
    sizes: tuple[str, ...],
    context: str,
) -> pd.DataFrame:
    """Return *frame* with its *prices* and *sizes* columns in the instrument's units.

    The scales are read from *units_of*, the frame the columns were taken
    from: its ``best_bid_price`` for prices and ``best_bid_vol`` for sizes,
    so a column derived from them (a mid, a depth bin) is scaled as they are.
    A float frame is already in those units and is returned unchanged in
    value.
    """
    price = price_scale(units_of, context, "best_bid_price")
    size = size_scale(units_of, context, "best_bid_vol")
    columns = {c: price for c in prices if c in frame.columns}
    columns.update({c: size for c in sizes if c in frame.columns})
    return frame.assign(
        **{c: frame[c].to_numpy(dtype=float) / scale for c, scale in columns.items()}
    )


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
    needs: tuple[str, ...] = ("depth_summary",)

    def compute(self, result: Any) -> pd.DataFrame:
        """Return ``timestamp``, the touch prices and sizes, and the last trade.

        Columns: ``timestamp``, ``best_bid_price``, ``best_bid_vol``,
        ``best_ask_price``, ``best_ask_vol``, ``last_price``, ``last_volume``,
        prices in the quote currency and sizes in the instrument's units.
        Each row carries every value forward from the rows before it, so a
        row is the whole quote at that time.  ``last_price`` is ``NaN`` before
        the first trade, and a side with no orders has a ``NaN`` price and
        size (the depth summary writes ``0`` there), read with the same test
        as :func:`~ob_analytics.depth.readable_quotes`.  The other side is
        still shown.
        """
        quotes = _in_units(
            get_spread(result.depth_summary),
            result.depth_summary,
            ("best_bid_price", "best_ask_price"),
            ("best_bid_vol", "best_ask_vol"),
            "l1_ticker",
        )
        trades = result.trades
        last = pd.DataFrame(
            {
                "timestamp": trades["timestamp"],
                "last_price": trades["price"].to_numpy(dtype=float)
                / price_scale(trades, "l1_ticker"),
                "last_volume": trades["volume"].to_numpy(dtype=float)
                / size_scale(trades, "l1_ticker"),
            }
        )
        # A quote and a trade at the same time keep that order: stable sort.
        frame = pd.concat([quotes, last], ignore_index=True)
        frame = frame.sort_values("timestamp", kind="stable").ffill()
        # After the fill, so an empty side is carried forward as empty rather
        # than filled from the quote before it.
        for side in ("bid", "ask"):
            empty = ~_quoted(frame[f"best_{side}_price"])
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
    needs: tuple[str, ...] = ("trades",)

    def compute(
        self,
        result: Any,
        *,
        bucket_volume: float | Literal["fit"] | None = None,
        n_buckets: int = 50,
        sign_method: str | None = None,
    ) -> pd.DataFrame:
        """Return the per-bucket VPIN table; the settings are ``compute_vpin``'s.

        *bucket_volume* is in the instrument's own units, as
        ``compute_vpin`` takes it, or ``"fit"`` to size it to the capture.
        """
        return compute_vpin(
            result.trades,
            bucket_volume=bucket_volume,
            n_buckets=n_buckets,
            sign_method=sign_method,
            quotes=result.depth_summary,
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
    needs: tuple[str, ...] = ("trades",)

    def compute(
        self,
        result: Any,
        *,
        window: str = "5min",
        sign_method: str | None = None,
        n_boot: int = 1000,
        ci_level: float = 0.95,
        seed: int | None = 0,
    ) -> pd.DataFrame:
        """Return the regression table; the settings are ``compute_kyle_lambda``'s."""
        kyle = compute_kyle_lambda(
            result.trades,
            window,
            sign_method=sign_method,
            quotes=result.depth_summary,
            n_boot=n_boot,
            ci_level=ci_level,
            seed=seed,
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

    The table adds ``last_price``, the last trade price in each window in the
    quote currency, which the face draws as a price line behind the bars.
    """

    name = "order_flow_imbalance"
    title = "Order Flow Imbalance"
    levels: tuple[Level, ...] = _EVERY_LEVEL
    needs: tuple[str, ...] = ("trades",)

    def compute(
        self,
        result: Any,
        *,
        window: str = "1min",
        sign_method: str | None = None,
    ) -> pd.DataFrame:
        """Return the per-window table; the settings are ``order_flow_imbalance``'s."""
        # The same time bars that cut the windows give each one's last price;
        # the bars keep the trades' units, so the price is converted here.
        trades = result.trades
        frame, windows = _ofi_windows(trades, window, sign_method, result.depth_summary)
        close = windows["close"].to_numpy(dtype=float)
        return frame.assign(last_price=close / price_scale(trades, "ofi"))

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
    needs: tuple[str, ...] = ("trades",)

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
            result.trades,
            horizons=horizons,
            grid=grid,
            sign_method=sign_method,
            quotes=result.depth_summary,
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


class BookSignalsMetric:
    """Mid, micro-price and order-book imbalance, from :func:`~ob_analytics.depth.depth_signals`.

    The table is the run's ``depth_summary`` with the signal columns added,
    prices in the quote currency and sizes in the instrument's units.  Its
    face draws the micro-price against the mid over an imbalance strip.
    """

    name = "book_signals"
    title = "Book Signals (micro-price & OBI)"
    levels: tuple[Level, ...] = _EVERY_LEVEL
    needs: tuple[str, ...] = ("depth_summary",)

    def compute(self, result: Any, *, depth_levels: int = 5) -> pd.DataFrame:
        """Return the summary with ``mid_price``, ``micro_price``, ``obi`` and ``obi_depth``.

        *depth_levels* is ``depth_signals``'s: the depth of ``obi_depth``.  The
        level count used, after clamping to the bins present, is in
        ``attrs["depth_levels"]``.
        """
        summary = result.depth_summary
        signals = depth_signals(summary, depth_levels=depth_levels)
        frame = _in_units(
            signals,
            summary,
            ("best_bid_price", "best_ask_price", "mid_price", "micro_price"),
            tuple(c for c in signals.columns if "vol" in c),
            "book_signals",
        )
        frame.attrs["depth_levels"] = imbalance_levels(summary, depth_levels)
        return frame

    def prepare(
        self,
        frame: pd.DataFrame,
        *,
        start_time: pd.Timestamp | None = None,
        end_time: pd.Timestamp | None = None,
    ) -> dict[str, Any]:
        """Payload for the face, for the rows from *start_time* to *end_time*."""
        from ob_analytics.visualization._data import prepare_book_signals_data

        return prepare_book_signals_data(
            frame,
            levels=frame.attrs.get("depth_levels", 5),
            start_time=start_time,
            end_time=end_time,
        )


class TransactionCostsMetric:
    """Per-trade costs, from :func:`~ob_analytics.cost.transaction_costs`.

    Measured against the run's own ``depth_summary``.  The face draws the
    effective and realized spread in basis points, averaged over windows.
    """

    name = "transaction_costs"
    title = "Transaction Costs (effective, realized, impact)"
    levels: tuple[Level, ...] = _EVERY_LEVEL
    needs: tuple[str, ...] = ("trades",)

    def compute(
        self,
        result: Any,
        *,
        horizon: str = "1min",
        sign_method: str | None = None,
    ) -> pd.DataFrame:
        """Return the per-trade cost table; the settings are ``transaction_costs``'s."""
        return transaction_costs(
            result.trades,
            result.depth_summary,
            horizon=horizon,
            sign_method=sign_method,
        )

    def prepare(self, frame: pd.DataFrame, *, window: str = "1min") -> dict[str, Any]:
        """Payload for the face; *window* is the span the lines average over."""
        from ob_analytics.visualization._data import prepare_transaction_costs_data

        return prepare_transaction_costs_data(frame, window=window)


class AmihudMetric:
    """Amihud illiquidity, from :func:`~ob_analytics.cost.amihud`."""

    name = "amihud"
    title = "Amihud Illiquidity"
    levels: tuple[Level, ...] = _EVERY_LEVEL
    needs: tuple[str, ...] = ("trades",)

    def compute(self, result: Any, *, window: str | None = None) -> pd.DataFrame:
        """Return the per-window table; the setting is ``amihud``'s."""
        return amihud(result.trades, window=window)

    def prepare(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Payload for the face: one point per window."""
        from ob_analytics.visualization._data import prepare_amihud_data

        return prepare_amihud_data(frame)


class RollSpreadMetric:
    """Roll's implied spread, from :func:`~ob_analytics.cost.roll_spread`."""

    name = "roll_spread"
    title = "Roll Spread"
    levels: tuple[Level, ...] = _EVERY_LEVEL
    needs: tuple[str, ...] = ("trades",)

    def compute(self, result: Any, *, window: str | None = None) -> pd.DataFrame:
        """Return the per-window table; the setting is ``roll_spread``'s."""
        return roll_spread(result.trades, window=window)

    def prepare(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Payload for the face: one point per window."""
        from ob_analytics.visualization._data import prepare_roll_spread_data

        return prepare_roll_spread_data(frame)


for _metric in (
    L1TickerMetric(),
    BookSignalsMetric(),
    VpinMetric(),
    KyleLambdaMetric(),
    OrderFlowImbalanceMetric(),
    OfiHorizonMetric(),
    TransactionCostsMetric(),
    AmihudMetric(),
    RollSpreadMetric(),
):
    register_metric(_metric)
