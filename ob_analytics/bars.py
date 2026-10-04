"""Bars: the trade stream resampled into open/high/low/close/volume rows.

A bar summarises a run of consecutive trades.  What differs between bar types
is only **where the boundaries fall**, so that decision is a *rule* — a
registered :class:`~ob_analytics.protocols.BarRule` — and everything else is
shared.  :func:`bars` applies the rule and builds the same columns every time.

Five rules ship with the package:

* ``"time"`` — a new bar every fixed span of the clock (classic OHLCV).
* ``"tick"`` — a new bar every N trades.
* ``"volume"`` — a new bar every N units of traded size.
* ``"dollar"`` — a new bar every N units of price × size.
* ``"imbalance"`` — a new bar every time signed size drifts N units away from
  where the bar opened (López de Prado's imbalance bars).

The last four sample the market by activity rather than by the clock, which is
why the quant literature prefers them: a quiet hour and a busy minute produce
the same number of bars, so bar returns come much closer to being independent
and identically distributed.

Usage::

    from ob_analytics import Pipeline, bars, sample_csv_path

    result = Pipeline().run(sample_csv_path())
    ohlcv = bars(result.trades, "time", "1min")
    vol_bars = bars(result.trades, "volume", 50)

A rule of your own registers with :func:`register_bar_rule` and is then usable
by name, with no edit to this module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ob_analytics._registry import Registry
from ob_analytics._utils import validate_columns, validate_non_empty
from ob_analytics.exceptions import ConfigError
from ob_analytics.protocols import BarRule
from ob_analytics.trade_sign import resolve_direction

#: Registry of rule name → :class:`~ob_analytics.protocols.BarRule` instance.
#: A rule carries no per-run state, so the object registered is the object
#: called — the same arrangement as :data:`~ob_analytics.metrics.METRICS`.
BAR_RULES: Registry[str, BarRule] = Registry("bar rule")

#: The columns :func:`bars` returns, in order.
BAR_COLUMNS: tuple[str, ...] = (
    "bar",
    "timestamp_start",
    "timestamp_end",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
    "n_trades",
    "vwap",
    "buy_volume",
    "sell_volume",
    "signed_volume",
)


def register_bar_rule(rule: BarRule) -> None:
    """Register *rule* under its own :attr:`~ob_analytics.protocols.BarRule.name`.

    Case-insensitive; overwriting an existing registration is allowed, so a
    rule of your own may deliberately shadow a built-in one.
    """
    BAR_RULES.register(rule.name, rule)


def list_bar_rules() -> list[str]:
    """Return a sorted list of registered bar-rule names."""
    return BAR_RULES.list()


def get_bar_rule(name: str) -> BarRule:
    """Return the bar rule registered under *name* (case-insensitive).

    Raises
    ------
    KeyError
        If no rule is registered under *name*; the message lists the
        registered names.
    """
    return BAR_RULES.get(name)


# ── The clock rule ───────────────────────────────────────────────────


class ClockRule:
    """A new bar every fixed span of the clock — classic OHLCV.

    The threshold is a fixed duration: a string pandas reads as one
    (``"5s"``, ``"1min"``, ``"1h"``) or a :class:`pandas.Timedelta`.  A
    calendar step with no fixed length (a month, a business day) is not a
    duration and is rejected.

    Bars are aligned to the epoch, so ``"1min"`` cuts on the minute whatever
    the first trade's timestamp.  A span in which nothing traded produces no
    bar: every row :func:`bars` returns holds at least one trade.
    """

    name = "time"

    def default_threshold(self, frame: pd.DataFrame, target_bars: int) -> pd.Timedelta:
        """Return the capture's span divided by *target_bars* (at least 1 ms)."""
        span = frame["timestamp"].iloc[-1] - frame["timestamp"].iloc[0]
        return max(span / target_bars, pd.Timedelta("1ms"))

    def normalize(self, threshold: Any) -> pd.Timedelta:
        """Read *threshold* as a fixed duration."""
        try:
            step = pd.Timedelta(threshold)
        except (ValueError, TypeError) as exc:
            raise ConfigError(
                f"bars: the 'time' rule needs a fixed duration such as '5s', "
                f"'1min' or '1h'; got {threshold!r}."
            ) from exc
        if step <= pd.Timedelta(0):
            raise ConfigError(
                f"bars: the 'time' rule needs a positive duration, got {step}"
            )
        return step

    def assign(self, frame: pd.DataFrame, threshold: pd.Timedelta) -> np.ndarray:
        """Return the bar index of each trade: its timestamp floored to *threshold*."""
        floored = frame["timestamp"].dt.floor(threshold)
        # `frame` is in trade order, so factorising the floored timestamps
        # numbers the occupied spans 0, 1, 2, ... and drops the empty ones.
        return pd.factorize(floored)[0].astype(np.int64)


# ── The counting rule ────────────────────────────────────────────────


class TickRule:
    """A new bar every N trades, whatever their size.

    The threshold is that count.  "Tick" here is the market-data sense of one
    printed trade, not the price increment.
    """

    name = "tick"

    def default_threshold(self, frame: pd.DataFrame, target_bars: int) -> int:
        """Return the trade count divided by *target_bars* (at least 1)."""
        return max(1, -(-len(frame) // target_bars))

    def normalize(self, threshold: Any) -> int:
        """Read *threshold* as a trade count."""
        try:
            count = int(threshold)
        except (ValueError, TypeError) as exc:
            raise ConfigError(
                f"bars: the 'tick' rule needs a whole number of trades, "
                f"got {threshold!r}"
            ) from exc
        if count < 1:
            raise ConfigError(
                f"bars: the 'tick' rule needs a count of 1 or more, got {threshold!r}"
            )
        return count

    def assign(self, frame: pd.DataFrame, threshold: int) -> np.ndarray:
        """Return the bar index of each trade: its position divided by *threshold*."""
        return np.arange(len(frame), dtype=np.int64) // threshold


# ── The accumulation rules ───────────────────────────────────────────


@dataclass
class AccumulationRule:
    """A new bar every N units of some quantity the trades carry.

    One shape covers the three activity rules; they differ only in what they
    accumulate and in whether that quantity is signed:

    * ``"volume"`` — traded size.
    * ``"dollar"`` — price × size, in the units of the input frame.
    * ``"imbalance"`` — signed size, ``+`` for buyer-initiated trades and
      ``-`` for seller-initiated ones.

    An unsigned rule closes a bar when the running total reaches the
    threshold.  A signed rule closes one when the running total reaches the
    threshold **in either direction**, so a bar ends on a burst of one-sided
    flow and a balanced stretch of trading stays inside one bar.

    The threshold is fixed, not the moving estimate of López de Prado's
    original imbalance bars: a fixed threshold gives the same bars every time
    the same trades are read, which is what makes a bar table reproducible.

    Attributes
    ----------
    name : str
        Registered rule name.
    quantity : Callable
        Maps the normalized trade frame to the per-trade amount accumulated.
    signed : bool
        Whether that amount carries the aggressor's sign.
    """

    name: str
    quantity: Callable[[pd.DataFrame], np.ndarray] = field(repr=False)
    signed: bool = False

    def default_threshold(self, frame: pd.DataFrame, target_bars: int) -> float:
        """Return the threshold that cuts *frame* into about *target_bars* bars.

        An unsigned quantity simply splits its total *target_bars* ways.  A
        signed one part-cancels, so its total says nothing about how far it
        drifts: the running sum wanders like a random walk, whose distance
        after a share of the trades is the root of that share's squared
        amounts.
        """
        amount = np.abs(self.quantity(frame))
        if self.signed:
            return float(np.sqrt(np.sum(amount**2) / target_bars))
        return float(np.sum(amount) / target_bars)

    def normalize(self, threshold: Any) -> float:
        """Read *threshold* as a positive amount."""
        try:
            size = float(threshold)
        except (ValueError, TypeError) as exc:
            raise ConfigError(
                f"bars: the {self.name!r} rule needs a number, got {threshold!r}"
            ) from exc
        if not size > 0:
            raise ConfigError(
                f"bars: the {self.name!r} rule needs a positive threshold, "
                f"got {threshold!r}"
            )
        return size

    def assign(self, frame: pd.DataFrame, threshold: float) -> np.ndarray:
        """Return the bar index of each trade by accumulating up to *threshold*."""
        amount = self.quantity(frame)
        return (
            _signed_bar_index(amount, threshold)
            if self.signed
            else _cumulative_bar_index(amount, threshold)
        )


def _cumulative_bar_index(amount: np.ndarray, threshold: float) -> np.ndarray:
    """Bar index per trade, closing a bar when the running total reaches *threshold*.

    *amount* must be non-negative, which makes its cumulative sum
    non-decreasing: each bar's last trade is then the first one at or past the
    next threshold, found by a binary search rather than a walk over trades.
    """
    cumulative = np.cumsum(np.asarray(amount, dtype=np.float64))
    index = np.empty(cumulative.size, dtype=np.int64)
    start = 0
    opened_at = 0.0
    bar = 0
    while start < cumulative.size:
        stop = int(np.searchsorted(cumulative, opened_at + threshold, side="left"))
        if stop >= cumulative.size:
            # The remaining trades never reach the threshold: one part-filled
            # closing bar.
            index[start:] = bar
            break
        index[start : stop + 1] = bar
        opened_at = float(cumulative[stop])
        start = stop + 1
        bar += 1
    return index


def _signed_bar_index(amount: np.ndarray, threshold: float) -> np.ndarray:
    """Bar index per trade, closing a bar when the running total reaches ±*threshold*.

    A signed running total is not monotonic, so there is no ordering to search:
    the first trade that takes the total out of the band has to be found by
    walking them in turn.
    """
    values = np.asarray(amount, dtype=np.float64)
    index = np.empty(values.size, dtype=np.int64)
    running = 0.0
    bar = 0
    for i in range(values.size):
        running += values[i]
        index[i] = bar
        if abs(running) >= threshold:
            bar += 1
            running = 0.0
    return index


def _volume(frame: pd.DataFrame) -> np.ndarray:
    return frame["volume"].to_numpy(dtype=np.float64)


def _turnover(frame: pd.DataFrame) -> np.ndarray:
    return frame["turnover"].to_numpy(dtype=np.float64)


def _signed_volume(frame: pd.DataFrame) -> np.ndarray:
    return frame["volume"].to_numpy(dtype=np.float64) * frame["sign"].to_numpy(
        dtype=np.float64
    )


for _rule in (
    ClockRule(),
    TickRule(),
    AccumulationRule("volume", _volume),
    AccumulationRule("dollar", _turnover),
    AccumulationRule("imbalance", _signed_volume, signed=True),
):
    register_bar_rule(_rule)


# ── The entry point ──────────────────────────────────────────────────


def bars(
    trades: pd.DataFrame,
    rule: str = "time",
    threshold: Any = None,
    *,
    target_bars: int = 50,
    sign_method: str | None = None,
    quotes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Resample *trades* into bars.

    Parameters
    ----------
    trades : pandas.DataFrame
        Trades with at least ``timestamp``, ``price`` and ``volume`` — a
        pipeline result's ``trades`` frame, or any frame shaped like it.  A
        ``direction`` column (``"buy"`` / ``"sell"``, the taker's side) is used
        when present; otherwise the aggressor side is classified — see
        *sign_method*.
    rule : str, optional
        Registered rule name: ``"time"`` (the default), ``"tick"``,
        ``"volume"``, ``"dollar"`` or ``"imbalance"``.  See
        :func:`list_bar_rules`.
    threshold : optional
        How much of the rule's own quantity closes a bar: a duration for
        ``"time"``, a trade count for ``"tick"``, an amount for the rest.
        ``None`` (the default) asks the rule for a threshold that yields about
        *target_bars* bars.
    target_bars : int, optional
        How many bars the default threshold aims at.  Ignored when
        *threshold* is given.
    sign_method : str or None, optional
        How to classify the aggressor side: ``None`` (the default) keeps a
        ``direction`` column if the frame has one and otherwise classifies,
        ``"tick"`` or ``"lee_ready"`` always classify. See
        :func:`~ob_analytics.trade_sign.resolve_direction`.
    quotes : pandas.DataFrame, optional
        Book snapshots for the Lee–Ready classifier — a frame with
        ``timestamp`` and either a mid or a bid/ask pair, such as a pipeline
        ``depth_summary``.  Only read when the aggressor side has to be
        classified.

    Returns
    -------
    pandas.DataFrame
        One row per bar, in time order, with the columns in
        :data:`BAR_COLUMNS`:

        ``bar``
            0-based bar number.
        ``timestamp_start`` / ``timestamp_end``
            First and last trade of the bar.
        ``open`` / ``high`` / ``low`` / ``close``
            Trade prices, in the units of the input frame.
        ``volume``
            Total size traded.
        ``turnover``
            Total price × size.
        ``n_trades``
            Number of trades.
        ``vwap``
            ``turnover / volume`` — the volume-weighted average price.
        ``buy_volume`` / ``sell_volume`` / ``signed_volume``
            Size split by aggressor side, and buys minus sells.

        Every bar holds at least one trade: a clock span in which nothing
        traded produces no row.  The last bar is whatever trades were left
        over, so it may not have reached the threshold; drop it with
        ``.iloc[:-1]`` where an equal-size bar matters.

        The frame's ``attrs`` carry ``bar_rule`` and ``bar_threshold``, so a
        caller that let the threshold default can still report what cut the
        bars.

    Raises
    ------
    ConfigError
        If required columns are missing, or the threshold does not suit the
        rule.
    KeyError
        If *rule* is not registered; the message lists the registered names.
    ObAnalyticsError
        If *trades* is empty.

    Notes
    -----
    Prices and sizes pass through in the units they arrive in.  A pipeline
    result holds prices as whole ticks and sizes as whole lots, so a
    ``"dollar"`` threshold there is in ticks × lots, not in the quote
    currency.  Convert the frame first (see
    :func:`~ob_analytics.visualization.display_result`) to work in quote-
    currency amounts.

    Examples
    --------
    >>> from ob_analytics import Pipeline, bars, sample_csv_path
    >>> result = Pipeline().run(sample_csv_path())  # doctest: +SKIP
    >>> bars(result.trades, "volume", 100)[  # doctest: +SKIP
    ...     ["timestamp_end", "open", "close", "vwap", "signed_volume"]
    ... ]
    """
    validate_non_empty(trades, "bars")
    validate_columns(trades, {"timestamp", "price", "volume"}, "bars")
    bar_rule = get_bar_rule(rule)

    frame = _normalize(trades, sign_method, quotes)
    if threshold is None:
        threshold = bar_rule.default_threshold(frame, max(1, int(target_bars)))
    threshold = bar_rule.normalize(threshold)
    index = np.asarray(bar_rule.assign(frame, threshold), dtype=np.int64)
    if index.shape != (len(frame),):
        raise ConfigError(
            f"bars: rule {rule!r} returned {index.shape} bar indices for "
            f"{len(frame)} trades."
        )
    out = _aggregate(frame, index)
    # The rule and the threshold it ran with, normalized to the rule's own
    # type, so a caller can report exactly what cut the bars.
    out.attrs["bar_rule"] = bar_rule.name
    out.attrs["bar_threshold"] = threshold
    return out


def _normalize(
    trades: pd.DataFrame,
    sign_method: str | None,
    quotes: pd.DataFrame | None,
) -> pd.DataFrame:
    """Return the trade-order frame every rule and the aggregation work from.

    Columns: ``timestamp``, ``price``, ``volume``, ``turnover`` (price × size)
    and ``sign`` (``+1`` buyer-initiated / ``-1`` seller-initiated).
    """
    ordered = trades.sort_values("timestamp", kind="stable")
    ordered = resolve_direction(ordered, sign_method, quotes, "bars")
    sign = np.where(np.asarray(ordered["direction"]) == "buy", 1, -1).astype(np.int8)
    price = ordered["price"].to_numpy()
    volume = ordered["volume"].to_numpy()
    return pd.DataFrame(
        {
            "timestamp": ordered["timestamp"].to_numpy(),
            "price": price,
            "volume": volume,
            "turnover": price * volume,
            "sign": sign,
        }
    )


def _aggregate(frame: pd.DataFrame, index: np.ndarray) -> pd.DataFrame:
    """Build the bar table from *frame* grouped by *index*."""
    buy = np.where(frame["sign"].to_numpy() > 0, frame["volume"].to_numpy(), 0)
    sell = np.where(frame["sign"].to_numpy() < 0, frame["volume"].to_numpy(), 0)
    frame = frame.assign(buy_volume=buy, sell_volume=sell)

    grouped = frame.groupby(index, sort=True)
    price = grouped["price"]
    out = pd.DataFrame(
        {
            "timestamp_start": grouped["timestamp"].first(),
            "timestamp_end": grouped["timestamp"].last(),
            "open": price.first(),
            "high": price.max(),
            "low": price.min(),
            "close": price.last(),
            "volume": grouped["volume"].sum(),
            "turnover": grouped["turnover"].sum(),
            "n_trades": grouped.size(),
            "buy_volume": grouped["buy_volume"].sum(),
            "sell_volume": grouped["sell_volume"].sum(),
        }
    ).reset_index(drop=True)

    # Renumber from 0: a rule may leave an index unused (an empty clock span),
    # and a bar table reads as 0, 1, 2, ... whichever rule cut it.
    out.insert(0, "bar", np.arange(len(out), dtype=np.int64))
    out["vwap"] = out["turnover"] / out["volume"].replace(0, np.nan)
    out["signed_volume"] = out["buy_volume"] - out["sell_volume"]
    return out[list(BAR_COLUMNS)]
