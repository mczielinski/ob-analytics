"""The feature table: one row per bar, one column per measurement.

A model or a study wants one tidy table — a point in time on each row and a
microstructure feature in each column.  The library measures all of those
already, but each in its own table on its own clock, so putting them side by
side means a join per measurement.  :func:`features` does that join once.

Two decisions make the table, and they are separate:

* **Where the rows fall** is a bar rule, the same one :func:`bars` uses.  A
  time grid is the ``"time"`` rule; ``"volume"``, ``"dollar"``, ``"tick"`` and
  ``"imbalance"`` sample by activity instead.  :func:`features` takes the same
  sampling arguments as :func:`~ob_analytics.bars.bars`, so the same arguments
  give the same cut in both.
* **What each column measures** is a *feature* — a registered
  :class:`~ob_analytics.protocols.Feature`.  Ten ship with the package, and a
  feature of your own registers with :func:`register_feature` and is then
  usable by name, with no edit to this module.

Each row is stated as of the close of its bar, in the ``timestamp`` column.
The trade columns hold what happened inside the bar, and the book columns hold
the book as it stood at the bar's close.  Nothing from after that instant
reaches the row, so the table carries no look-ahead.  It also carries no
target: a model's target is a forward return or a forward label, and building
one is a deliberate shift the caller makes (see the how-to).

Usage::

    from ob_analytics import Pipeline, features, sample_csv_path

    result = Pipeline().run(sample_csv_path())
    table = features(result.trades, result.depth_summary, "volume", 100)

Without a quotes frame the book features drop out and the table holds the
trade features alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger

from ob_analytics._registry import Registry
from ob_analytics._utils import validate_columns
from ob_analytics.bars import bars
from ob_analytics.depth import bin_volume_columns, book_imbalance, micro_price
from ob_analytics.exceptions import ConfigError
from ob_analytics.protocols import Feature

#: Registry of feature name → :class:`~ob_analytics.protocols.Feature`
#: instance.  A feature carries no per-run state, so the object registered is
#: the object called — the same arrangement as
#: :data:`~ob_analytics.bars.BAR_RULES`.
FEATURES: Registry[str, Feature] = Registry("feature")

#: The columns that identify a row, before any feature's columns.  ``bar``
#: numbers the rows from 0, ``timestamp_start`` is when the bar opened, and
#: ``timestamp`` is when it closed — the instant the whole row is stated as of.
INDEX_COLUMNS: tuple[str, ...] = ("bar", "timestamp_start", "timestamp")

#: The features that ship with the package, in the order their columns are
#: written.  This is what :func:`features` measures when asked for nothing in
#: particular, followed by any further registered feature, sorted by name.
DEFAULT_FEATURES: tuple[str, ...] = (
    "price",
    "returns",
    "flow",
    "spread",
    "mid_price",
    "micro_price",
    "imbalance",
    "depth",
    "vpin",
    "kyle_lambda",
)

#: How many bars back a trailing-window feature looks by default.
DEFAULT_WINDOW: int = 20

#: Basis points in one, the scale ``spread_bps`` is written on.
_BPS = 10_000.0


def register_feature(feature: Feature) -> None:
    """Register *feature* under its own :attr:`~ob_analytics.protocols.Feature.name`.

    Case-insensitive; overwriting an existing registration is allowed, so a
    feature of your own may deliberately shadow a built-in one.
    """
    FEATURES.register(feature.name, feature)


def list_features() -> list[str]:
    """Return a sorted list of registered feature names."""
    return FEATURES.list()


def get_feature(name: str) -> Feature:
    """Return the feature registered under *name* (case-insensitive).

    Raises
    ------
    KeyError
        If no feature is registered under *name*; the message lists the
        registered names.
    """
    return FEATURES.get(name)


# ── Trade features ───────────────────────────────────────────────────
#
# These read the bar's own columns, which hold what traded between the bar's
# open and its close and nothing else.


class PriceFeature:
    """The bar's four prices and its volume-weighted average price.

    ``open`` / ``high`` / ``low`` / ``close``
        Trade prices within the bar, in the units of the trades frame.
    ``vwap``
        Turnover divided by volume.
    """

    name = "price"
    columns = ("open", "high", "low", "close", "vwap")
    requires = frozenset(columns)

    def compute(self, frame: pd.DataFrame) -> dict[str, npt.ArrayLike]:
        """Return the bar's price columns unchanged."""
        return {column: frame[column] for column in self.columns}


@dataclass
class ReturnsFeature:
    """The bar's return, and how much returns have been moving lately.

    ``log_return``
        ``log(close / previous close)``.  The first bar has no previous close,
        so it is ``NaN``; so is a bar whose close is not a positive price.
    ``realized_vol``
        Standard deviation of ``log_return`` over the last :attr:`window`
        bars, the row's own included.  It is per bar and not annualized: only
        clock bars span equal amounts of time, so there is no one factor that
        would scale it to a year.

    Attributes
    ----------
    name : str
        Registered feature name.
    window : int
        How many bars ``realized_vol`` looks back over.
    """

    name: str = "returns"
    window: int = DEFAULT_WINDOW
    columns = ("log_return", "realized_vol")
    requires = frozenset({"close"})

    def compute(self, frame: pd.DataFrame) -> dict[str, npt.ArrayLike]:
        """Return the log return per bar and its trailing standard deviation."""
        close = frame["close"].to_numpy(dtype=float)
        previous = np.roll(close, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            log_return = np.where(
                (close > 0) & (previous > 0), np.log(close / previous), np.nan
            )
        # The roll wrapped the last close onto row 0, which has no previous
        # bar at all.
        log_return[0] = np.nan
        realized = pd.Series(log_return).rolling(self.window, min_periods=2).std(ddof=1)
        return {"log_return": log_return, "realized_vol": realized}


class FlowFeature:
    """How much traded in the bar, and how one-sided it was.

    ``volume`` / ``turnover`` / ``n_trades``
        Size traded, price × size traded, and the number of trades.
    ``signed_volume``
        Buyer-initiated volume minus seller-initiated volume.
    ``trade_imbalance``
        ``signed_volume / volume`` — the signed share of the bar's volume,
        from ``-1`` (every trade a sell) to ``+1`` (every trade a buy).
    """

    name = "flow"
    columns = (
        "volume",
        "turnover",
        "n_trades",
        "signed_volume",
        "trade_imbalance",
    )
    requires = frozenset({"volume", "turnover", "n_trades", "signed_volume"})

    def compute(self, frame: pd.DataFrame) -> dict[str, npt.ArrayLike]:
        """Return the bar's flow columns and the imbalance they imply."""
        volume = frame["volume"].to_numpy(dtype=float)
        signed = frame["signed_volume"].to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            imbalance = np.where(volume > 0, signed / volume, np.nan)
        return {
            "volume": frame["volume"],
            "turnover": frame["turnover"],
            "n_trades": frame["n_trades"],
            "signed_volume": frame["signed_volume"],
            "trade_imbalance": imbalance,
        }


# ── Book features ────────────────────────────────────────────────────
#
# These read the book as it stood at the bar's close, joined on from the
# quotes frame.  Without quotes they are skipped.


class SpreadFeature:
    """What it cost to cross the book at the bar's close.

    ``spread``
        Best ask price minus best bid price, in the units of the quotes frame.
    ``spread_bps``
        The same as a share of the mid-price, in basis points.
    """

    name = "spread"
    columns = ("spread", "spread_bps")
    requires = frozenset({"best_bid_price", "best_ask_price"})

    def compute(self, frame: pd.DataFrame) -> dict[str, npt.ArrayLike]:
        """Return the spread in price units and in basis points."""
        bid = frame["best_bid_price"].to_numpy(dtype=float)
        ask = frame["best_ask_price"].to_numpy(dtype=float)
        spread = ask - bid
        mid = (ask + bid) / 2.0
        with np.errstate(invalid="ignore", divide="ignore"):
            in_bps = np.where(mid > 0, spread / mid * _BPS, np.nan)
        return {"spread": spread, "spread_bps": in_bps}


class MidPriceFeature:
    """The mid-price at the bar's close: the average of the two best prices."""

    name = "mid_price"
    columns = ("mid_price",)
    requires = frozenset({"best_bid_price", "best_ask_price"})

    def compute(self, frame: pd.DataFrame) -> dict[str, npt.ArrayLike]:
        """Return the plain mid-price per bar."""
        bid = frame["best_bid_price"].to_numpy(dtype=float)
        ask = frame["best_ask_price"].to_numpy(dtype=float)
        return {"mid_price": (bid + ask) / 2.0}


class MicroPriceFeature:
    """The mid-price weighted by the size resting on the opposite side.

    ``micro_price``
        The size-weighted mid from :func:`~ob_analytics.depth.micro_price`.  It
        leans toward the side carrying the heavier opposite book, which is the
        direction price is more likely to move.
    ``micro_price_offset``
        ``micro_price - mid_price``: how far it leans, in price units.  This
        is the part a model wants — the micro-price itself tracks the mid so
        closely that the two say almost the same thing.
    """

    name = "micro_price"
    columns = ("micro_price", "micro_price_offset")
    requires = frozenset(
        {"best_bid_price", "best_bid_vol", "best_ask_price", "best_ask_vol"}
    )

    def compute(self, frame: pd.DataFrame) -> dict[str, npt.ArrayLike]:
        """Return the micro-price and its distance from the mid."""
        micro = micro_price(frame).to_numpy(dtype=float)
        bid = frame["best_bid_price"].to_numpy(dtype=float)
        ask = frame["best_ask_price"].to_numpy(dtype=float)
        return {"micro_price": micro, "micro_price_offset": micro - (bid + ask) / 2.0}


@dataclass
class ImbalanceFeature:
    """The signed share of resting volume on the bid side.

    ``obi``
        Book imbalance at the touch — :func:`~ob_analytics.depth.book_imbalance`
        with ``levels=1``.
    ``obi_depth``
        The same measured over :attr:`depth_levels`, so it reads the pressure
        behind the touch as well as on it.  The level count is clamped to the
        depth bins the quotes frame actually carries, so a summary with fewer
        bins gives a shallower reading rather than an error.

    Attributes
    ----------
    name : str
        Registered feature name.
    depth_levels : int
        Depth for ``obi_depth``, counted the way
        :func:`~ob_analytics.depth.book_imbalance` counts it: ``1`` is the
        touch alone and each further level adds one depth bin.
    """

    name: str = "imbalance"
    depth_levels: int = 5
    columns = ("obi", "obi_depth")
    requires = frozenset({"best_bid_vol", "best_ask_vol"})

    def compute(self, frame: pd.DataFrame) -> dict[str, npt.ArrayLike]:
        """Return the touch imbalance and the deeper one."""
        available = min(
            len(bin_volume_columns(frame, "bid")),
            len(bin_volume_columns(frame, "ask")),
        )
        levels = max(1, min(self.depth_levels, available + 1))
        return {
            "obi": book_imbalance(frame, levels=1),
            "obi_depth": book_imbalance(frame, levels=levels),
        }


class DepthFeature:
    """How much size was resting at the bar's close.

    ``best_bid_vol`` / ``best_ask_vol``
        Size at the touch on each side.
    ``bid_depth`` / ``ask_depth``
        Size within every depth bin the quotes frame carries — the innermost
        bin already includes the touch.  A quotes frame with no depth bins
        reports the touch size, which is all the depth it knows about.
    """

    name = "depth"
    columns = (
        "best_bid_vol",
        "best_ask_vol",
        "bid_depth",
        "ask_depth",
    )
    requires = frozenset({"best_bid_vol", "best_ask_vol"})

    def compute(self, frame: pd.DataFrame) -> dict[str, npt.ArrayLike]:
        """Return the touch size and the measured depth on each side."""
        return {
            "best_bid_vol": frame["best_bid_vol"],
            "best_ask_vol": frame["best_ask_vol"],
            "bid_depth": _side_depth(frame, "bid"),
            "ask_depth": _side_depth(frame, "ask"),
        }


def _side_depth(frame: pd.DataFrame, side: str) -> np.ndarray:
    """Total resting size on *side* across the depth bins, or at the touch."""
    bins = bin_volume_columns(frame, side)
    if not bins:
        return frame[f"best_{side}_vol"].to_numpy(dtype=float)
    return frame[bins].to_numpy(dtype=float).sum(axis=1)


# ── Toxicity features ────────────────────────────────────────────────
#
# Both read a trailing window of bars.  They include the row they are on,
# which is past-only: a bar's own trades all happened before it closed.


@dataclass
class VpinFeature:
    """How one-sided the flow has been over the last :attr:`window` bars.

    ``vpin``
        The mean of the absolute trade imbalance over the window.  It runs
        from ``0`` (every bar balanced) to ``1`` (every bar one-sided), and a
        high reading says the flow has been persistently directional, which is
        what a market maker is exposed to.

    Sampled on volume bars this is VPIN, the volume-synchronized probability
    of informed trading of Easley, López de Prado and O'Hara, whose buckets
    are equal amounts of traded volume.  On another rule it is the same
    measure read on that rule's clock.  Either way the imbalance is divided by
    the bar's own volume, so a bar that traded less is not read as a quieter
    one.

    :func:`~ob_analytics.flow_toxicity.compute_vpin` computes VPIN without a
    bar table, and cuts its buckets slightly differently: a trade that
    straddles a bucket boundary is split between the two, where a volume bar
    keeps the trade whole and closes a little past its threshold.  The two
    readings track each other closely rather than matching row for row.

    Attributes
    ----------
    name : str
        Registered feature name.
    window : int
        How many bars the mean covers — the bucket count of the original
        measure, which the paper puts at 50.
    columns : tuple of str
        The one column, named after :attr:`name`, so a second window
        registered under another name writes a column of its own rather than
        overwriting this one.
    """

    name: str = "vpin"
    window: int = DEFAULT_WINDOW
    columns = ("vpin",)
    requires = frozenset({"volume", "signed_volume"})

    def __post_init__(self) -> None:
        self.columns = (self.name,)

    def compute(self, frame: pd.DataFrame) -> dict[str, npt.ArrayLike]:
        """Return the trailing mean absolute trade imbalance."""
        volume = frame["volume"].to_numpy(dtype=float)
        signed = frame["signed_volume"].to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            imbalance = np.where(volume > 0, np.abs(signed) / volume, np.nan)
        rolling = pd.Series(imbalance).rolling(self.window, min_periods=1).mean()
        return {self.name: rolling}


@dataclass
class KyleLambdaFeature:
    """How far price moved per unit of net order flow, lately.

    ``kyle_lambda``
        The slope of ``close - open`` on ``signed_volume``, fitted over the
        last :attr:`window` bars.  A high value says a given imbalance pushed
        the price a long way, which is a book that was expensive to trade
        against.  It is in the price units of the trades frame per unit of
        volume: on a pipeline result that is ticks per lot, so multiply by
        ``config.tick_size`` to read it in the quote currency.

    This is Kyle's λ, estimated on a trailing window rather than over a whole
    run.  :func:`~ob_analytics.flow_toxicity.compute_kyle_lambda` fits the
    same regression once across every window of a run, and reports the
    t-statistic and R² that say whether the fit means anything; a rolling
    slope reports neither, so read it as a level that moves, not as a test.

    Attributes
    ----------
    name : str
        Registered feature name.
    window : int
        How many bars each fit covers.
    min_bars : int
        The fewest bars that produce a slope.  Earlier rows are ``NaN``.
    columns : tuple of str
        The one column, named after :attr:`name`, so a second window
        registered under another name writes a column of its own rather than
        overwriting this one.
    """

    name: str = "kyle_lambda"
    window: int = DEFAULT_WINDOW
    min_bars: int = 3
    columns = ("kyle_lambda",)
    requires = frozenset({"open", "close", "signed_volume"})

    def __post_init__(self) -> None:
        self.columns = (self.name,)

    def compute(self, frame: pd.DataFrame) -> dict[str, npt.ArrayLike]:
        """Return the trailing regression slope per bar."""
        flow = pd.Series(frame["signed_volume"].to_numpy(dtype=float))
        move = pd.Series(
            frame["close"].to_numpy(dtype=float) - frame["open"].to_numpy(dtype=float)
        )
        periods = min(self.min_bars, self.window)
        window = flow.rolling(self.window, min_periods=periods)
        covariance = window.cov(move)
        variance = window.var(ddof=1)
        # A window in which every bar carried the same net flow has no slope
        # to fit, rather than an infinite one.
        slope = covariance.where(variance > 0) / variance.where(variance > 0)
        return {self.name: slope}


for _feature in (
    PriceFeature(),
    ReturnsFeature(),
    FlowFeature(),
    SpreadFeature(),
    MidPriceFeature(),
    MicroPriceFeature(),
    ImbalanceFeature(),
    DepthFeature(),
    VpinFeature(),
    KyleLambdaFeature(),
):
    register_feature(_feature)


# ── The entry point ──────────────────────────────────────────────────


def features(
    trades: pd.DataFrame,
    quotes: pd.DataFrame | None = None,
    rule: str = "time",
    threshold: Any = None,
    *,
    target_bars: int = 50,
    include: Sequence[str] | None = None,
    sign_method: str | None = None,
) -> pd.DataFrame:
    """Build a feature table from *trades* and the book in *quotes*.

    The trades are cut into bars by *rule* and *threshold* — the same cut
    :func:`~ob_analytics.bars.bars` makes from the same arguments — and each
    bar becomes one row.  Every registered feature that the inputs support
    then writes its columns onto that row.

    Parameters
    ----------
    trades : pandas.DataFrame
        Trades with at least ``timestamp``, ``price`` and ``volume`` — a
        pipeline result's ``trades`` frame, or any frame shaped like it.  A
        ``direction`` column (``"buy"`` / ``"sell"``, the taker's side) is used
        when present; otherwise the aggressor side is classified — see
        *sign_method*.
    quotes : pandas.DataFrame, optional
        Book snapshots supplying the book columns: ``timestamp`` plus the
        touch (``best_bid_price``, ``best_bid_vol``, ``best_ask_price``,
        ``best_ask_vol``) and any depth bins.  A pipeline ``depth_summary``
        satisfies this.  ``None`` (the default) leaves the book features out
        and measures the trades alone.
    rule : str, optional
        Registered bar rule deciding where the rows fall: ``"time"`` (the
        default), ``"tick"``, ``"volume"``, ``"dollar"`` or ``"imbalance"``.
        See :func:`~ob_analytics.bars.list_bar_rules`.
    threshold : optional
        How much of the rule's own quantity closes a bar.  ``None`` (the
        default) asks the rule for a threshold that yields about *target_bars*
        rows.
    target_bars : int, optional
        How many rows the default threshold aims at.  Ignored when *threshold*
        is given.
    include : sequence of str, optional
        The features to measure, in the order their columns are written.
        ``None`` (the default) measures every registered feature the inputs
        support: the ten in :data:`DEFAULT_FEATURES` first, then any further
        registration, sorted by name.  A feature named here whose inputs are
        missing raises; one selected by default is skipped instead.
    sign_method : str or None, optional
        How to classify the aggressor side: ``None`` (the default) keeps a
        ``direction`` column if the trades have one and otherwise classifies,
        ``"tick"`` or ``"lee_ready"`` always classify.  See
        :func:`~ob_analytics.trade_sign.resolve_direction`.

    Returns
    -------
    pandas.DataFrame
        One row per bar, in time order.  The first three columns are
        :data:`INDEX_COLUMNS` — ``bar``, ``timestamp_start`` and
        ``timestamp`` — and the rest are the features' own, in the order they
        were measured.

        ``timestamp`` is the close of the bar, and the instant the whole row
        is stated as of: the trade columns hold what happened between
        ``timestamp_start`` and it, and the book columns hold the book as it
        stood at it.  Nothing later reaches the row, whichever rule cut the
        bars, so the table carries no look-ahead.

        A row with no readable quote behind it has no book to read, and a
        trailing window with too little history behind it has nothing to
        measure; both are ``NaN`` rather than a filled-in value.  Two quote
        states are not readable as a book and are skipped rather than taken at
        face value — an empty side, and a crossed one.  See
        :func:`readable_quotes`.

        The frame's ``attrs`` carry ``bar_rule`` and ``bar_threshold``, the
        cut the rows were made on; ``features``, the names measured; and
        ``features_skipped``, the names left out for want of their inputs.

    Raises
    ------
    ConfigError
        If required columns are missing, if the threshold does not suit the
        rule, if a feature named in *include* cannot read what it needs, or if
        two features would write the same column.
    KeyError
        If *rule*, or a name in *include*, is not registered; the message
        lists the registered names.
    ObAnalyticsError
        If *trades* is empty.

    Notes
    -----
    Prices and sizes pass through in the units they arrive in.  A pipeline
    result holds prices as whole ticks and sizes as whole lots, so ``spread``
    is in ticks and ``vwap`` is a tick count; ``spread_bps``, ``obi`` and
    ``trade_imbalance`` are ratios and do not depend on the unit.

    The table is features only.  A model also needs a target, and a target
    looks forward: build one from the table with a negative shift, which is
    the one place look-ahead belongs.

    No row reads data from after its own close, whatever the threshold.  The
    *choice* of threshold is another matter: left to default it is worked out
    from the whole trades frame, so where the boundaries fall depends on the
    whole capture.  Pass a threshold when the cut itself has to be something
    the rows could have been given at the time.

    Examples
    --------
    >>> from ob_analytics import Pipeline, features, sample_csv_path
    >>> result = Pipeline().run(sample_csv_path())  # doctest: +SKIP
    >>> table = features(  # doctest: +SKIP
    ...     result.trades, result.depth_summary, "volume", 100
    ... )
    >>> table[["timestamp", "close", "spread_bps", "obi", "vpin"]]  # doctest: +SKIP
    """
    bar_table = bars(
        trades,
        rule,
        threshold,
        target_bars=target_bars,
        sign_method=sign_method,
        quotes=quotes,
    )
    frame = bar_table.rename(columns={"timestamp_end": "timestamp"})
    if quotes is not None:
        frame = _join_book(frame, quotes)

    names = list(include) if include is not None else _default_selection()
    table = frame[list(INDEX_COLUMNS)].copy()
    measured: list[str] = []
    skipped: list[str] = []
    # Which feature wrote each column, so a clash names both sides.
    written: dict[str, str] = {}
    present = set(frame.columns)

    for name in names:
        feature = get_feature(name)
        missing = frozenset(feature.requires) - present
        if missing:
            if include is not None:
                raise ConfigError(
                    f"features: {name!r} reads {sorted(missing)}, which the "
                    f"inputs do not carry. Pass a quotes frame, or leave "
                    f"{name!r} out of include."
                )
            skipped.append(name)
            logger.debug("Feature {!r} skipped: no {}", name, sorted(missing))
            continue
        for column, values in _measure(feature, frame).items():
            if column in table.columns:
                raise ConfigError(
                    f"features: {name!r} writes column {column!r}, which "
                    f"{written[column]!r} already wrote. Two features cannot "
                    f"share a column: register one under the other's name to "
                    f"replace it, or give it a column of its own."
                )
            table[column] = values
            written[column] = name
        measured.append(name)

    table.attrs["bar_rule"] = bar_table.attrs["bar_rule"]
    table.attrs["bar_threshold"] = bar_table.attrs["bar_threshold"]
    table.attrs["features"] = measured
    table.attrs["features_skipped"] = skipped
    return table


def _default_selection() -> list[str]:
    """Return every registered feature: the shipped ones first, then the rest."""
    shipped = [name for name in DEFAULT_FEATURES if name in FEATURES]
    extra = sorted(set(list_features()) - set(shipped))
    return shipped + extra


def _measure(feature: Feature, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Run *feature* over *frame* and check it returned what it declared.

    The check is here rather than in each feature so that a feature of your
    own is held to the same contract as a built-in one, and a mistake in it is
    reported against the feature rather than surfacing later as a column of
    the wrong length.
    """
    produced = feature.compute(frame)
    declared = tuple(feature.columns)
    if set(produced) != set(declared):
        raise ConfigError(
            f"features: {feature.name!r} declares columns {sorted(declared)} "
            f"but returned {sorted(produced)}."
        )
    out: dict[str, np.ndarray] = {}
    for column in declared:
        values = np.asarray(produced[column])
        if values.shape != (len(frame),):
            raise ConfigError(
                f"features: {feature.name!r} returned {values.shape} values "
                f"for column {column!r} on {len(frame)} bars."
            )
        out[column] = values
    return out


def readable_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    """Return the rows of *quotes* whose book can be read as a price.

    Two states get through a depth summary that are not books anything could
    have traded against, and both would otherwise reach a row as ordinary
    numbers:

    **A side with nothing resting on it.**  The depth engine writes a price
    and a volume of ``0`` for an empty side, which is a marker and not a
    price.  Taken at face value it makes a spread the width of the whole
    instrument, a mid at half the other side, and a micro-price of zero —
    three finite numbers, none of them true, and none of them marked.

    **A crossed book**, where the best bid is above the best ask.  A diff feed
    can hold genuinely crossed resting orders, so this is an expected state on
    such a feed rather than a fault, but its midpoint is not a price and its
    spread is negative.  The test is ``bid > ask``, the same one
    :func:`~ob_analytics.trade_sign.prevailing_mid` applies: a *locked* book,
    bid equal to ask, is a real state at a spread of zero and is kept.

    Dropping these from the reference series is what makes a bar reach back to
    the last quote that could be read, the way it reaches back over any other
    instant with no quote of its own.  A frame carrying no
    ``best_bid_price`` / ``best_ask_price`` pair cannot be tested and is
    returned unchanged.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Book snapshots — a pipeline ``depth_summary``, or any frame shaped
        like one.

    Returns
    -------
    pandas.DataFrame
        The readable rows, in their original order.
    """
    if not {"best_bid_price", "best_ask_price"} <= set(quotes.columns):
        return quotes
    bid = quotes["best_bid_price"].to_numpy(dtype=float)
    ask = quotes["best_ask_price"].to_numpy(dtype=float)
    return quotes[(bid > 0) & (ask > 0) & (bid <= ask)]


def _join_book(frame: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """Attach the book as it stood at each bar's close.

    A backward as-of join: each bar takes the last *readable* quote published
    at or before it closed — see :func:`readable_quotes`.  A quote stamped at
    exactly that instant counts, because it is part of what had already
    happened when the bar ended.  A bar with no readable quote behind it gets
    ``NaN``, so "nothing to read yet" stays distinguishable from a real
    reading.
    """
    validate_columns(quotes, {"timestamp"}, "features(quotes)")
    # A bar column and a quote column of the same name would collide in the
    # join; the bar's own is the one the features were written against.
    keep = ["timestamp"] + [
        column
        for column in quotes.columns
        if column != "timestamp" and column not in frame.columns
    ]
    book = readable_quotes(quotes)[keep].sort_values("timestamp", kind="stable")
    joined = pd.merge_asof(
        frame,
        book,
        on="timestamp",
        direction="backward",
        allow_exact_matches=True,
    )
    if book.empty:
        # merge_asof against an empty right frame keeps that frame's dtypes,
        # so an integer price column comes back as an all-zero int rather
        # than the "no reading" the row actually has.
        for column in keep:
            if column != "timestamp":
                joined[column] = np.nan
    return joined
