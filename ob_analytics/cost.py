"""Transaction-cost and price-impact measures.

The quoted spread is what the book advertised.  The **effective spread** is
what a taker actually paid, measured from the mid-price the trade crossed.
Part of that payment is the liquidity provider's compensation, and part is the
price moving against them because the trade carried information.  The split
only shows once time has passed, and it is exact::

    effective spread = realized spread + price impact

Two functions measure a trade against the quotes around it, so they need a
quote frame and the taker's aggressor side:

* :func:`transaction_costs` — all three measures, per trade, in price units
  and in basis points.
* :func:`cost_summary` — the same three volume-weighted over a run, as a
  :class:`CostSummary`.

Two more read the trade prices alone, so they run on a tape with no quotes and
no aggressor side at all:

* :func:`amihud` — Amihud (2002) illiquidity: the price move a unit of
  turnover buys.
* :func:`roll_spread` — Roll (1984): the spread implied by bid-ask bounce in
  the price series.

Units.  The absolute columns carry the unit of the ``price`` column handed in,
which on a pipeline frame is integer ticks (see the price policy in
:mod:`ob_analytics.schemas`); multiply by ``tick_size`` for the quote
currency.  The ``_bps`` columns are ratios to the mid-price, so they carry no
unit and are the ones to compare across instruments.  :func:`amihud` has a
unit of its own; its docstring says which.

References
----------
Amihud, Y. (2002). "Illiquidity and stock returns: cross-section and
time-series effects." *Journal of Financial Markets* 5(1), 31-56.

Bessembinder, H. (2003). "Issues in assessing trade execution costs."
*Journal of Financial Markets* 6(3), 233-257.

Huang, R. D., & Stoll, H. R. (1996). "Dealer versus auction markets: a paired
comparison of execution costs on NASDAQ and the NYSE." *Journal of Financial
Economics* 41(3), 313-357.

Roll, R. (1984). "A simple implicit measure of the effective bid-ask spread in
an efficient market." *The Journal of Finance* 39(4), 1127-1139.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.typing import DataFrameGroupBy

from ob_analytics._utils import validate_columns, validate_non_empty
from ob_analytics.trade_sign import prevailing_mid, resolve_direction

#: Basis points in one, the scale every ``_bps`` column is written on.
_BPS = 10_000.0

#: Columns :func:`transaction_costs` returns, in order.
COST_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "price",
    "volume",
    "direction",
    "mid_price",
    "future_mid_price",
    "effective_spread",
    "realized_spread",
    "price_impact",
    "effective_spread_bps",
    "realized_spread_bps",
    "price_impact_bps",
)


@dataclass(frozen=True)
class CostSummary:
    """One run's transaction costs, volume-weighted.

    Each figure weights a trade by its size, so it reports what the average
    unit traded cost, not what the average trade cost.  That is the convention
    the execution-cost literature reports (Bessembinder 2003), and it is the
    number a taker sizing an order needs: one large trade at a wide spread
    costs more than ten small ones at a narrow spread, and an unweighted mean
    would say the opposite.

    The realized-spread and price-impact figures are averaged over the trades
    that have a future mid to measure against, which is fewer than the
    effective-spread figure has — the last *horizon* of the capture has no
    future left in it.  :attr:`n_realized` says how many that was, so a
    summary computed on a short capture cannot quietly look like a full one.

    Attributes
    ----------
    effective_spread : float
        Volume-weighted effective spread, in the price unit of the trades.
    realized_spread : float
        Volume-weighted realized spread, same unit.
    price_impact : float
        Volume-weighted price impact, same unit.  Equal to
        ``effective_spread - realized_spread`` only when both averages cover
        the same trades, which is why it is measured rather than subtracted.
    effective_spread_bps, realized_spread_bps, price_impact_bps : float
        The same three in basis points of the mid-price.
    horizon : str
        The realized-spread horizon the costs were measured at.
    n_trades : int
        Trades with a mid-price to measure the effective spread against.
    n_realized : int
        Trades that also had a mid-price *horizon* later, so the realized
        spread and the impact could be measured.
    volume : float
        Total size of the :attr:`n_trades` trades — the weight behind the
        effective-spread figure.
    """

    effective_spread: float
    realized_spread: float
    price_impact: float
    effective_spread_bps: float
    realized_spread_bps: float
    price_impact_bps: float
    horizon: str
    n_trades: int
    n_realized: int
    volume: float


def transaction_costs(
    trades: pd.DataFrame,
    quotes: pd.DataFrame,
    *,
    horizon: str = "1min",
    sign_method: str | None = None,
) -> pd.DataFrame:
    r"""Measure what each trade cost the taker, and where that cost went.

    For a trade at price :math:`p_t` with aggressor side :math:`D_t` (``+1``
    for a buy, ``-1`` for a sell), against the mid-price :math:`m_t`
    prevailing when it printed and the mid-price :math:`m_{t+\Delta}`
    prevailing one *horizon* later:

    * **effective spread** :math:`= 2 D_t (p_t - m_t)` — the round-trip cost
      of crossing, as the taker experienced it.  Doubled so it is comparable
      with a quoted spread, which also spans both sides of the mid.
    * **realized spread** :math:`= 2 D_t (p_t - m_{t+\Delta})` — the part the
      liquidity provider kept, measured once the trade's information has had
      *horizon* to reach the price.
    * **price impact** :math:`= 2 D_t (m_{t+\Delta} - m_t)` — the rest: how
      far the trade moved the market.  The two parts add back to the
      effective spread exactly, row by row.

    The horizon matters and there is no neutral choice.  Too short and the
    price has not finished reacting, so impact is understated; too long and
    unrelated moves are counted as this trade's impact.  Five minutes is the
    equity convention; the default here is one minute, because captures of a
    fast crypto tape are usually measured in minutes rather than hours.
    Report the horizon alongside the number.

    A trade in the last *horizon* of the quote frame has no future mid to be
    measured against.  Rather than reuse the final quote — which would report
    a shrinking horizon as though it were the full one — those rows get
    ``NaN`` realized spread and impact, and keep their effective spread.

    :math:`m_t` is the mid of the last quote **strictly before** the trade.
    When the quote frame is a ``depth_summary`` built from the same event
    stream, the row stamped at the trade's own instant is the book *after*
    the trade took the touch, and measuring against it would charge the taker
    nothing for a move they caused.  Venue timestamps are coarse, so a quote
    a few events older is the closest honest reference available.  Crossed
    quotes are skipped for the same reason: a book whose best bid is above
    its best ask has no midpoint, so the last uncrossed quote is used instead.

    Every number here inherits the quality of the book it is measured
    against.  On a diff feed a trade can print through resting orders the
    venue never withdrew, which reads as a negative effective spread — the
    taker apparently paying less than the mid.  That is a finding about the
    capture, not about the market: run
    :func:`~ob_analytics.analytics.detect_stale_orders` (or the ``audit``
    command) before reading the costs as execution quality.

    Parameters
    ----------
    trades : pandas.DataFrame
        Trades with ``timestamp``, ``price`` and ``volume``.  A ``direction``
        column (``"buy"`` / ``"sell"``, the taker side) is used when present;
        otherwise it is inferred — see *sign_method*.
    quotes : pandas.DataFrame
        Quote frame supplying the mid-price: ``timestamp`` plus either a mid
        column (``mid`` / ``midprice`` / ``mid_price``) or a bid/ask pair
        (``best_bid_price`` / ``best_ask_price``, ``best_bid`` / ``best_ask``,
        or ``bid`` / ``ask``).  A pipeline ``depth_summary`` satisfies this.
    horizon : str, optional
        Pandas offset string for :math:`\Delta`, the wait before the realized
        spread is read.  Default ``"1min"``.
    sign_method : str, optional
        How to obtain the aggressor side when there is no native
        ``direction``.  ``None`` (default) keeps a native ``direction`` and
        otherwise classifies with Lee-Ready against *quotes*.  ``"tick"`` /
        ``"lee_ready"`` force that classifier, overriding a native
        ``direction``.  See
        :func:`~ob_analytics.trade_sign.classify_trade_sign`.

    Returns
    -------
    pandas.DataFrame
        One row per trade, chronological, with the columns in
        :data:`COST_COLUMNS`.  ``mid_price`` and ``future_mid_price`` are the
        two mids the measures were taken against, kept so a number can be
        traced back to the quotes that produced it.  The horizon is recorded
        in ``frame.attrs["horizon"]``.

    Raises
    ------
    ConfigError
        If a required column is missing, or *sign_method* is ``"bvc"``, which
        labels volume bars rather than trades.
    ObAnalyticsError
        If *trades* is empty.

    Examples
    --------
    >>> from ob_analytics import Pipeline, sample_csv_path, transaction_costs
    >>> result = Pipeline().run(sample_csv_path())
    >>> costs = transaction_costs(result.trades, result.depth_summary, horizon="5s")
    >>> costs["effective_spread_bps"].median()
    0.1276...
    """
    validate_columns(trades, {"timestamp", "price", "volume"}, "transaction_costs")
    validate_non_empty(trades, "transaction_costs")
    validate_columns(quotes, {"timestamp"}, "transaction_costs(quotes)")

    trades = resolve_direction(trades, sign_method, quotes, "transaction_costs")
    df = trades.sort_values("timestamp", kind="stable").reset_index(drop=True)

    price = df["price"].to_numpy(dtype=np.float64)
    sign = np.where(df["direction"].to_numpy() == "buy", 1.0, -1.0)

    # The contemporaneous mid is the last quote *strictly before* the trade:
    # on a quote frame built from the same event stream, the row sharing the
    # trade's instant is the book after that trade took the touch.  The future
    # mid is read at an arbitrary instant, so an exact match is fine there.
    mid = prevailing_mid(
        df["timestamp"].to_numpy(),
        quotes,
        "transaction_costs",
        allow_exact=False,
        skip_crossed=True,
    )
    later = df["timestamp"] + pd.Timedelta(horizon)
    future_mid = prevailing_mid(
        later.to_numpy(), quotes, "transaction_costs", skip_crossed=True
    )
    # Past the last quote a backward join keeps returning the final mid, which
    # would silently measure a shorter horizon than the one asked for.
    beyond = (later > quotes["timestamp"].max()).to_numpy()
    future_mid = np.where(beyond, np.nan, future_mid)

    effective = 2.0 * sign * (price - mid)
    realized = 2.0 * sign * (price - future_mid)
    impact = 2.0 * sign * (future_mid - mid)

    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(mid > 0, _BPS / mid, np.nan)

    out = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "price": df["price"],
            "volume": df["volume"],
            "direction": df["direction"],
            "mid_price": mid,
            "future_mid_price": future_mid,
            "effective_spread": effective,
            "realized_spread": realized,
            "price_impact": impact,
            "effective_spread_bps": effective * scale,
            "realized_spread_bps": realized * scale,
            "price_impact_bps": impact * scale,
        },
        columns=list(COST_COLUMNS),
    )
    out.attrs["horizon"] = horizon
    return out


def cost_summary(costs: pd.DataFrame) -> CostSummary:
    """Reduce a :func:`transaction_costs` frame to one volume-weighted figure each.

    Each measure is averaged over the trades that actually carry it, weighted
    by trade size: the effective spread over every trade with a mid, the
    realized spread and the impact over the smaller set that also had a mid a
    horizon later.  A measure with no usable trade is ``NaN`` rather than an
    error, so a summary of a capture shorter than its own horizon still
    returns and says so through :attr:`~CostSummary.n_realized`.

    Parameters
    ----------
    costs : pandas.DataFrame
        The frame :func:`transaction_costs` returned.  The horizon is read
        from ``costs.attrs["horizon"]`` when present.

    Returns
    -------
    CostSummary

    Raises
    ------
    ConfigError
        If *costs* is missing a column :func:`transaction_costs` writes.
    ObAnalyticsError
        If *costs* is empty.
    """
    validate_columns(costs, COST_COLUMNS, "cost_summary")
    validate_non_empty(costs, "cost_summary")

    weight = costs["volume"].to_numpy(dtype=np.float64)
    measured = costs["effective_spread"].notna().to_numpy()
    realized_measured = costs["realized_spread"].notna().to_numpy()

    def weighted(column: str, mask: np.ndarray) -> float:
        values = costs[column].to_numpy(dtype=np.float64)[mask]
        weights = weight[mask]
        total = weights.sum()
        if values.size == 0 or total <= 0:
            return float("nan")
        return float(np.dot(values, weights) / total)

    return CostSummary(
        effective_spread=weighted("effective_spread", measured),
        realized_spread=weighted("realized_spread", realized_measured),
        price_impact=weighted("price_impact", realized_measured),
        effective_spread_bps=weighted("effective_spread_bps", measured),
        realized_spread_bps=weighted("realized_spread_bps", realized_measured),
        price_impact_bps=weighted("price_impact_bps", realized_measured),
        horizon=str(costs.attrs.get("horizon", "")),
        n_trades=int(measured.sum()),
        n_realized=int(realized_measured.sum()),
        volume=float(weight[measured].sum()),
    )


def amihud(trades: pd.DataFrame, *, window: str | None = None) -> pd.DataFrame:
    """Amihud (2002) illiquidity: price move per unit of turnover.

    ::

        amihud = |return| / turnover

    Read it as the price response a unit of trading buys.  A market where a
    large turnover barely moves the price is deep, and scores low; one where a
    small turnover swings it is thin, and scores high.  It needs no quotes and
    no aggressor side, so it is the one liquidity measure available on a tape
    that carries nothing but prices, sizes and times.

    The return is taken **within** each window, from its first trade price to
    its last, and turnover is the price times size summed over the same
    trades.  Amihud's original is a daily close-to-close return over that
    day's volume; a capture is not a series of trading days, and taking each
    window's own first and last print keeps every window usable — including
    the first, which a close-to-close definition cannot measure — and makes
    the whole-session case (*window* ``None``) the same calculation over one
    window rather than a different one.

    The result carries a unit: one over the turnover unit of the input, which
    on a pipeline frame is ticks times lots.  It is comparable across windows
    of one run, and across runs of one instrument, but not across instruments
    without rescaling.  Published figures are multiplied by a power of ten to
    reach readable digits; this returns the raw ratio and leaves that choice
    to the caller.

    Parameters
    ----------
    trades : pandas.DataFrame
        Trades with ``timestamp``, ``price`` and ``volume``.
    window : str or None, optional
        Pandas offset string for the window, e.g. ``"5min"``.  ``None``
        (default) measures the whole session as a single window and returns
        one row.

    Returns
    -------
    pandas.DataFrame
        One row per window with ``timestamp`` (the window's start),
        ``first_price``, ``last_price``, ``abs_return``, ``turnover``,
        ``n_trades`` and ``amihud``.  Windows with no trade are dropped;
        a window whose turnover is zero has ``NaN`` for ``amihud``.

    Raises
    ------
    ConfigError
        If a required column is missing.
    ObAnalyticsError
        If *trades* is empty.
    """
    validate_columns(trades, {"timestamp", "price", "volume"}, "amihud")
    validate_non_empty(trades, "amihud")

    df = trades.sort_values("timestamp", kind="stable")
    turnover = df["price"].to_numpy(dtype=np.float64) * df["volume"].to_numpy(
        dtype=np.float64
    )
    df = df.assign(_turnover=turnover)

    grouped = _by_window(df, window).agg(
        first_price=("price", "first"),
        last_price=("price", "last"),
        turnover=("_turnover", "sum"),
        n_trades=("price", "size"),
    )
    grouped = grouped[grouped["n_trades"] > 0].reset_index()

    first = grouped["first_price"].to_numpy(dtype=np.float64)
    last = grouped["last_price"].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        grouped["abs_return"] = np.where(first > 0, np.abs(last / first - 1.0), np.nan)
        grouped["amihud"] = np.where(
            grouped["turnover"].to_numpy(dtype=np.float64) > 0,
            grouped["abs_return"].to_numpy(dtype=np.float64)
            / grouped["turnover"].to_numpy(dtype=np.float64),
            np.nan,
        )

    return grouped[
        [
            "timestamp",
            "first_price",
            "last_price",
            "abs_return",
            "turnover",
            "n_trades",
            "amihud",
        ]
    ]


def roll_spread(trades: pd.DataFrame, *, window: str | None = None) -> pd.DataFrame:
    """Roll (1984): the spread implied by bid-ask bounce in the price series.

    A tape with no quotes still shows the spread, because consecutive trades
    alternate between hitting the bid and lifting the ask, and that bounce
    makes successive price changes negatively correlated.  Roll turns the size
    of that correlation back into a spread::

        roll_spread = 2 * sqrt(-cov(dp_t, dp_{t-1}))

    where ``dp`` is the change in trade price.

    The model assumes an efficient price plus a bounce of constant half-spread,
    with order flow that carries no information.  Under it the bounce is the
    *only* source of price change, so the lag-1 autocorrelation of the changes
    is exactly ``-0.5``.  That is the number to check: ``autocorrelation`` is
    returned beside the estimate, and the further it sits from ``-0.5``, the
    less of the price movement the bounce explains.

    Two things push it away.  Real flow carries information, which raises the
    autocovariance and makes Roll read low against a measured effective
    spread.  More decisively, the efficient price moves between trades, and on
    a sparse tape in a volatile instrument it moves much further than half a
    spread — the bounce then accounts for a few per cent of the variance and
    the autocovariance comes out **positive**, where the formula has no real
    root.  The estimate is ``NaN`` there rather than a number the model does
    not support, and the two diagnostic columns say why.  Sampling more often
    than the tape trades will not help: the fix is a denser tape or a wider
    spread, and where quotes exist :func:`transaction_costs` measures the
    spread directly instead of inferring it.

    Parameters
    ----------
    trades : pandas.DataFrame
        Trades with ``timestamp`` and ``price``, chronological or not.
    window : str or None, optional
        Pandas offset string for the window, e.g. ``"5min"``.  ``None``
        (default) estimates over the whole session and returns one row.

    Returns
    -------
    pandas.DataFrame
        One row per window with ``timestamp`` (the window's start),
        ``n_trades``, ``mean_price``, ``autocovariance``, ``autocorrelation``,
        ``roll_spread`` and ``roll_spread_bps``.  ``autocorrelation`` is the
        lag-1 autocorrelation of the price changes, which Roll's model puts at
        ``-0.5``; a window whose value is far from that is one the model does
        not describe.  A window with fewer than four trades — too few for two
        overlapping price changes — or a non-negative autocovariance has
        ``NaN`` for both spread columns.

    Raises
    ------
    ConfigError
        If a required column is missing.
    ObAnalyticsError
        If *trades* is empty.
    """
    validate_columns(trades, {"timestamp", "price"}, "roll_spread")
    validate_non_empty(trades, "roll_spread")

    df = trades.sort_values("timestamp", kind="stable")
    rows = [
        _roll_window(start, group["price"].to_numpy(dtype=np.float64))
        for start, group in _by_window(df, window)
        if len(group) > 0
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "n_trades",
            "mean_price",
            "autocovariance",
            "autocorrelation",
            "roll_spread",
            "roll_spread_bps",
        ],
    )


def _by_window(df: pd.DataFrame, window: str | None) -> DataFrameGroupBy:
    """Group *df* by *window*, or into a single whole-session group.

    A ``None`` window groups on the first timestamp, so the session case is
    one group keyed the same way a window is — the caller's aggregation and
    column layout then do not branch on it.
    """
    if window is None:
        key = pd.Series(df["timestamp"].iloc[0], index=df.index, name="timestamp")
        return df.groupby(key)
    return df.groupby(pd.Grouper(key="timestamp", freq=window))


def _roll_window(start: pd.Timestamp, prices: np.ndarray) -> dict:
    """Roll's estimate over one window's prices, in trade order."""
    n = len(prices)
    mean_price = float(prices.mean()) if n else float("nan")
    row = {
        "timestamp": start,
        "n_trades": n,
        "mean_price": mean_price,
        "autocovariance": float("nan"),
        "autocorrelation": float("nan"),
        "roll_spread": float("nan"),
        "roll_spread_bps": float("nan"),
    }
    # Two overlapping price changes need four prices; with fewer, the lag-1
    # autocovariance is a single product and carries no information.
    if n < 4:
        return row

    # Lag-1 autocovariance of the price-change series, centred on that one
    # series' own mean -- not on each lagged half's mean, which would let a
    # drift the two halves share leak into the estimate.
    changes = np.diff(prices)
    centred = changes - changes.mean()
    autocovariance = float(np.dot(centred[1:], centred[:-1]) / (len(changes) - 1))
    row["autocovariance"] = autocovariance
    # Roll's model puts this at exactly -0.5, because the bounce is then the
    # only thing moving the price.  How far it lands from -0.5 is how much of
    # the movement the model does not explain.
    variance = float(np.dot(centred, centred) / len(changes))
    if variance > 0:
        row["autocorrelation"] = autocovariance / variance
    if autocovariance < 0:
        spread = 2.0 * np.sqrt(-autocovariance)
        row["roll_spread"] = float(spread)
        if mean_price > 0:
            row["roll_spread_bps"] = float(_BPS * spread / mean_price)
    return row
