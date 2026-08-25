"""Synthetic L3 (market-by-order) session generator.

A seedable simulator that emits a per-order event stream — ``created`` /
``changed`` / ``deleted`` rows plus matching trades — in the exact column
layout and volume/fill semantics of a real loader (see
:mod:`ob_analytics.schemas` and :class:`~ob_analytics.bitstamp.BitstampLoader`).
The output satisfies :func:`~ob_analytics.schemas.validate_events_df` and
:func:`~ob_analytics.schemas.validate_trades_df` and flows through the full
pipeline (order-type classification, price-level depth, depth metrics, order
aggressiveness, order-book reconstruction) without special handling.

Why this exists
---------------

* Deterministic test fixtures and CI without shipping licensed capture data.
* Ground truth for benchmarks and for the trade-sign / flow-toxicity harnesses:
  every trade carries its true maker and taker order, so classification can be
  checked against the generator's own labels.
* Teaching: the tutorial series can generate its own books at any size.

The model
---------

Three independent arrival processes drive the simulation over a fixed span of
simulated seconds:

* **Limit orders** rest passively on the book. A new order picks a side, a
  price a whole number of ticks behind the touch (never crossing), and a size.
* **Cancellations** remove one resting order chosen uniformly at random.
* **Market orders** cross the spread and consume resting liquidity best price
  first, oldest order first (price-time priority). Each maker order consumed
  produces one trade and one maker-side event; the aggressor produces its own
  matching events and is fully filled by construction, so it never rests.

Arrivals are Poisson (the baseline) or Hawkes (self-exciting, generated with
Ogata thinning and numpy only). Everything is driven by one seeded
:class:`numpy.random.Generator`, so a given :class:`SynthConfig` produces
byte-identical output every run.

Order types produced
--------------------

Under :func:`~ob_analytics.analytics.set_order_types` the orders classify as
``resting-limit`` (passive orders that trade or survive), ``flashed-limit``
(placed and cancelled without trading), and ``market`` (aggressors). The
generator does not produce ``market-limit`` or ``pre-existing`` orders: every
order has a ``created`` event and no aggressor rests, which is a deliberate
simplification, not a defect.

Optional injections
-------------------

Two effects are available and **off by default**:

* **Iceberg** orders display a small peak and refill from a hidden reserve. A
  refilled slice enters at the back of its price queue (it loses time
  priority), which is the honest L3 footprint of a hidden-size order.
* **Toxic flow** enlarges a fraction of market orders to model informed takers
  that consume more liquidity. This raises order-flow imbalance and VPIN.

Both are simplifications intended for teaching and stress tests, documented as
such.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import pandas as pd

from ob_analytics._utils import UTC_NS_DTYPE
from ob_analytics.analytics import set_order_types
from ob_analytics.exceptions import ConfigError

__all__ = [
    "SynthConfig",
    "SynthSession",
    "SyntheticLoader",
    "SyntheticTradeSource",
    "generate_session",
]

# Event / trade tuples are accumulated as plain Python tuples and turned into
# typed DataFrames once at the end (the same build-at-the-end pattern used by
# ob_analytics.datasets), which keeps the hot simulation loop allocation-light.
_EventRow = tuple[int, int, float, int, float, float, str, str]
_TradeRow = tuple[float, int, float, str, int, int, int, int]

# Module-level default so the frozen dataclass field is a name, not a call
# evaluated in the class body (Timestamp is immutable, so sharing it is safe).
# Tz-aware UTC, matching the schema's timestamp policy (issue #154).
_DEFAULT_START_TIME = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")


@dataclass(frozen=True)
class SynthConfig:
    """Typed parameters for :func:`generate_session`.

    Every field has a default that produces a lively but bounded BTC/USD-style
    book around a mid of 100 over five simulated minutes. Override individual
    fields for other instruments, intensities, or session lengths.

    Parameters
    ----------
    seed : int
        Seed for the single :class:`numpy.random.Generator`. Identical configs
        with the same seed produce byte-identical output.
    duration : float
        Length of the simulated session in seconds.
    start_time : pandas.Timestamp
        Wall-clock anchor for the first possible event.  Emitted timestamps are
        tz-aware UTC nanoseconds, matching the schema's timestamp policy
        (issue #154); a tz-naive anchor is read as UTC.
    arrival_process : {"poisson", "hawkes"}
        Arrival model for all three streams. ``"poisson"`` is the constant-rate
        baseline; ``"hawkes"`` is self-exciting (each event briefly raises the
        arrival rate), generated with Ogata thinning.
    limit_rate, cancel_rate, market_rate : float
        Baseline arrival intensities in events per second for new limit orders,
        cancellations, and market orders. For Hawkes these are the background
        rates ``mu`` before self-excitation.
    hawkes_excitation : float
        Branching ratio of the Hawkes kernel (mean offspring per event). Must
        be below 1 for a stationary process; higher values cluster arrivals
        more tightly. Ignored for Poisson.
    hawkes_decay : float
        Exponential decay rate ``beta`` of the Hawkes kernel (per second): how
        fast the excitation from one event fades. Ignored for Poisson.
    mid_price : float
        Reference mid price. The book is anchored here; new limit orders are
        priced relative to it, so the session is roughly stationary.
    tick_size : float
        Smallest price increment. Prices are whole multiples of this value.
    half_spread_ticks : int
        Distance in ticks from the mid to the best price on each side, so the
        resting spread is ``2 * half_spread_ticks`` ticks.
    depth_levels : int
        Number of distinct price levels available on each side.
    level_decay : float
        Success probability of the geometric distribution used to pick how many
        ticks behind the touch a new order sits. Higher values concentrate
        orders at the touch; must be in ``(0, 1]``.
    mean_size : float
        Mean of the exponential order-size distribution.
    min_size : float
        Lower bound applied to every drawn size.
    limit_bid_prob, market_buy_prob : float
        Probability that a new limit order is a bid, and that a market order is
        a buy. 0.5 each gives a balanced book.
    price_decimals, volume_decimals : int
        Rounding precision for emitted prices and volumes, matching
        :class:`~ob_analytics.config.PipelineConfig`.
    iceberg_fraction : float
        Probability that a new limit order is an iceberg (off at 0.0).
    iceberg_size_multiple : float
        An iceberg's total size is this multiple of its displayed peak; the
        remainder is hidden and refilled one peak at a time.
    toxic_fraction : float
        Probability that a market order is toxic / informed (off at 0.0).
    toxic_size_multiple : float
        Size-mean multiplier applied to a toxic market order.
    """

    seed: int = 0
    duration: float = 300.0
    start_time: pd.Timestamp = _DEFAULT_START_TIME

    arrival_process: Literal["poisson", "hawkes"] = "poisson"
    limit_rate: float = 6.0
    cancel_rate: float = 4.0
    market_rate: float = 1.5
    hawkes_excitation: float = 0.5
    hawkes_decay: float = 1.0

    mid_price: float = 100.0
    tick_size: float = 0.01
    half_spread_ticks: int = 2
    depth_levels: int = 10
    level_decay: float = 0.5

    mean_size: float = 1.0
    min_size: float = 0.01
    limit_bid_prob: float = 0.5
    market_buy_prob: float = 0.5

    price_decimals: int = 2
    volume_decimals: int = 8

    iceberg_fraction: float = 0.0
    iceberg_size_multiple: float = 5.0
    toxic_fraction: float = 0.0
    toxic_size_multiple: float = 3.0

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ConfigError("SynthConfig: duration must be positive.")
        if not 0.0 < self.level_decay <= 1.0:
            raise ConfigError("SynthConfig: level_decay must be in (0, 1].")
        if self.depth_levels < 1:
            raise ConfigError("SynthConfig: depth_levels must be at least 1.")
        if self.half_spread_ticks < 1:
            raise ConfigError("SynthConfig: half_spread_ticks must be at least 1.")
        if self.arrival_process == "hawkes" and not 0.0 <= self.hawkes_excitation < 1.0:
            raise ConfigError(
                "SynthConfig: hawkes_excitation must be in [0, 1) for a "
                "stationary process."
            )


@dataclass(frozen=True)
class SynthSession:
    """A generated session: the two canonical frames plus the config used.

    Attributes
    ----------
    events : pandas.DataFrame
        Per-order events satisfying
        :func:`~ob_analytics.schemas.validate_events_df`, in ``event_id`` order,
        already classified by :func:`~ob_analytics.analytics.set_order_types`.
    trades : pandas.DataFrame
        Executions satisfying
        :func:`~ob_analytics.schemas.validate_trades_df`; every row references
        the ``event_id`` of its maker and taker events.
    config : SynthConfig
        The configuration that produced this session.
    """

    events: pd.DataFrame
    trades: pd.DataFrame
    config: SynthConfig


@dataclass
class _Order:
    """Mutable resting-order record held in the simulator's book."""

    id: int
    side: str
    price_tick: int
    outstanding: float
    created_event_id: int
    hidden: float = 0.0
    peak: float = 0.0


def generate_session(
    config: SynthConfig | None = None, /, **overrides: Any
) -> SynthSession:
    """Generate one synthetic L3 session.

    Parameters
    ----------
    config : SynthConfig, optional
        Base configuration. Defaults to :class:`SynthConfig` with all defaults.
    **overrides
        Field overrides applied on top of *config* (e.g. ``seed=7``,
        ``duration=60``). Convenience for ``generate_session(seed=7)`` without
        constructing a config explicitly.

    Returns
    -------
    SynthSession
        The events frame, the trades frame, and the resolved config.

    Examples
    --------
    >>> from ob_analytics.synth import generate_session
    >>> session = generate_session(seed=1, duration=30)
    >>> sorted(session.events["type"].unique().astype(str))  # doctest: +SKIP
    ['flashed-limit', 'market', 'resting-limit']
    """
    cfg = config or SynthConfig()
    if overrides:
        cfg = replace(cfg, **overrides)
    sim = _Simulator(cfg)
    sim.run()
    events, trades = sim.frames()
    return SynthSession(events=events, trades=trades, config=cfg)


class _Simulator:
    """Event-driven engine that produces the raw event and trade rows.

    Prices are held internally as integer tick counts to avoid floating-point
    drift; they are converted to rounded float prices only when the output
    frames are built.
    """

    def __init__(self, config: SynthConfig) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

        self.mid_tick = round(config.mid_price / config.tick_size)
        self.half_spread = int(config.half_spread_ticks)
        self._vol_eps = 10.0 ** (-(config.volume_decimals + 1))

        self.next_event_id = 1
        self.next_order_id = 1

        self.orders: dict[int, _Order] = {}
        self.bid_levels: dict[int, deque[int]] = {}
        self.ask_levels: dict[int, deque[int]] = {}

        # Live passive orders, held as a list plus a position index so a random
        # cancellation target is an O(1) pick and removal is an O(1) swap.
        self.live_ids: list[int] = []
        self.live_pos: dict[int, int] = {}

        self.events: list[_EventRow] = []
        self.trades: list[_TradeRow] = []

    # ── Public driver ────────────────────────────────────────────────

    def run(self) -> None:
        """Generate all arrivals and process them in time order."""
        limit_times = self._arrival_times(self.cfg.limit_rate)
        cancel_times = self._arrival_times(self.cfg.cancel_rate)
        market_times = self._arrival_times(self.cfg.market_rate)

        # kind 0 = limit, 1 = cancel, 2 = market; the kind tie-break makes the
        # merge order deterministic when two arrivals share a timestamp.
        arrivals: list[tuple[float, int]] = []
        arrivals.extend((t, 0) for t in limit_times)
        arrivals.extend((t, 1) for t in cancel_times)
        arrivals.extend((t, 2) for t in market_times)
        arrivals.sort()

        for t, kind in arrivals:
            if kind == 0:
                self._limit_order(t)
            elif kind == 1:
                self._cancel(t)
            else:
                self._market_order(t)

    def frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the classified events frame and the trades frame."""
        if not self.events:
            raise ConfigError(
                "generate_session: no events were produced. Increase duration "
                "or the arrival rates."
            )
        trades = self._build_trades_frame()
        events = self._build_events_frame()
        # Classify with the same function the pipeline uses, so the session is
        # self-consistent and re-running the pipeline is idempotent.
        events = set_order_types(events, trades)
        return events, trades

    # ── Arrival processes ────────────────────────────────────────────

    def _arrival_times(self, rate: float) -> list[float]:
        if rate <= 0:
            return []
        if self.cfg.arrival_process == "hawkes":
            return self._hawkes_times(rate)
        return self._poisson_times(rate)

    def _poisson_times(self, rate: float) -> list[float]:
        times: list[float] = []
        t = 0.0
        dur = self.cfg.duration
        while True:
            t += float(self.rng.exponential(1.0 / rate))
            if t >= dur:
                break
            times.append(t)
        return times

    def _hawkes_times(self, rate: float) -> list[float]:
        """Univariate exponential-kernel Hawkes process via Ogata thinning.

        Intensity is ``mu + sum(alpha * exp(-beta * (t - t_i)))`` over past
        events. ``alpha = excitation * beta`` so ``excitation`` is the mean
        number of offspring per event (the branching ratio).
        """
        mu = rate
        beta = self.cfg.hawkes_decay
        alpha = self.cfg.hawkes_excitation * beta
        dur = self.cfg.duration
        # Guard against a runaway path in a near-critical configuration.
        cap = int(1000 + 100 * mu * dur)

        times: list[float] = []
        t = 0.0
        excitation = 0.0
        while True:
            bound = mu + excitation  # intensity peaks at the interval's start
            if bound <= 0:
                break
            w = float(self.rng.exponential(1.0 / bound))
            t += w
            if t >= dur:
                break
            excitation *= math.exp(-beta * w)
            if float(self.rng.random()) <= (mu + excitation) / bound:
                times.append(t)
                excitation += alpha
                if len(times) >= cap:
                    break
        return times

    # ── Arrival handlers ─────────────────────────────────────────────

    def _limit_order(self, t: float) -> None:
        is_bid = float(self.rng.random()) < self.cfg.limit_bid_prob
        iceberg_r = float(self.rng.random())
        offset = self._draw_offset()
        peak = self._draw_size(self.cfg.mean_size)

        side = "bid" if is_bid else "ask"
        if is_bid:
            price_tick = self.mid_tick - self.half_spread - offset
        else:
            price_tick = self.mid_tick + self.half_spread + offset

        hidden = 0.0
        stored_peak = 0.0
        if iceberg_r < self.cfg.iceberg_fraction:
            total = self._round_vol(peak * self.cfg.iceberg_size_multiple)
            hidden = self._round_vol(total - peak)
            stored_peak = peak if hidden > self._vol_eps else 0.0

        self._place_order(side, price_tick, peak, hidden, stored_peak, t)

    def _cancel(self, t: float) -> None:
        if not self.live_ids:
            return
        idx = int(self.rng.integers(0, len(self.live_ids)))
        order = self.orders[self.live_ids[idx]]
        # A cancellation removes the full outstanding size with no execution.
        self._emit(
            order.id, order.side, order.price_tick, "deleted", order.outstanding, 0.0, t
        )
        self._remove_live(order.id)

    def _market_order(self, t: float) -> None:
        taker_side = (
            "buy" if float(self.rng.random()) < self.cfg.market_buy_prob else "sell"
        )
        toxic = float(self.rng.random()) < self.cfg.toxic_fraction
        mean = self.cfg.mean_size * (self.cfg.toxic_size_multiple if toxic else 1.0)
        desired = self._draw_size(mean)

        fills, iceberg_parents = self._sweep(taker_side, desired)
        if not fills:
            return

        agg_side = "bid" if taker_side == "buy" else "ask"
        touch_tick = fills[0][0].price_tick
        consumed = self._round_vol(sum(amount for _, amount in fills))
        agg_id = self._new_order_id()
        self._emit(agg_id, agg_side, touch_tick, "created", consumed, 0.0, t)

        agg_out = consumed
        for order, amount in fills:
            if order.outstanding <= self._vol_eps:
                maker_eid = self._emit(
                    order.id, order.side, order.price_tick, "deleted", 0.0, amount, t
                )
            else:
                maker_eid = self._emit(
                    order.id,
                    order.side,
                    order.price_tick,
                    "changed",
                    order.outstanding,
                    amount,
                    t,
                )
            agg_out = self._round_vol(agg_out - amount)
            if agg_out <= self._vol_eps:
                taker_eid = self._emit(
                    agg_id, agg_side, touch_tick, "deleted", 0.0, amount, t
                )
            else:
                taker_eid = self._emit(
                    agg_id, agg_side, touch_tick, "changed", agg_out, amount, t
                )
            self.trades.append(
                (
                    t,
                    order.price_tick,
                    amount,
                    taker_side,
                    maker_eid,
                    taker_eid,
                    order.id,
                    agg_id,
                )
            )

        # Refill any consumed iceberg orders after the sweep, so a refilled
        # slice is not re-crossed by the same aggressor (it loses time
        # priority, entering at the back of its price queue).
        for parent in iceberg_parents:
            self._replenish_iceberg(parent, t)

    # ── Book operations ──────────────────────────────────────────────

    def _sweep(
        self, taker_side: str, desired: float
    ) -> tuple[list[tuple[_Order, float]], list[_Order]]:
        """Consume resting liquidity best price first, oldest order first.

        Returns the list of ``(maker_order, amount)`` fills (with each maker's
        ``outstanding`` already decremented) and the list of fully-consumed
        iceberg orders awaiting a refill.
        """
        if taker_side == "buy":
            levels = self.ask_levels
            ticks = sorted(levels)
        else:
            levels = self.bid_levels
            ticks = sorted(levels, reverse=True)

        fills: list[tuple[_Order, float]] = []
        iceberg_parents: list[_Order] = []
        remaining = desired
        for tick in ticks:
            if remaining <= self._vol_eps:
                break
            queue = levels[tick]
            while queue and remaining > self._vol_eps:
                oid = queue[0]
                if oid not in self.live_pos:
                    queue.popleft()  # cancelled order, skip
                    continue
                order = self.orders[oid]
                take = self._round_vol(min(remaining, order.outstanding))
                if take <= self._vol_eps:
                    queue.popleft()
                    self._remove_live(oid)
                    continue
                order.outstanding = self._round_vol(order.outstanding - take)
                remaining = self._round_vol(remaining - take)
                fills.append((order, take))
                if order.outstanding <= self._vol_eps:
                    queue.popleft()
                    self._remove_live(oid)
                    if order.hidden > self._vol_eps:
                        iceberg_parents.append(order)
                else:
                    break  # maker partially filled; it keeps the front spot
            if not queue:
                del levels[tick]
        return fills, iceberg_parents

    def _replenish_iceberg(self, parent: _Order, t: float) -> None:
        slice_size = self._round_vol(min(parent.peak, parent.hidden))
        if slice_size <= self._vol_eps:
            return
        hidden = self._round_vol(parent.hidden - slice_size)
        stored_peak = parent.peak if hidden > self._vol_eps else 0.0
        self._place_order(
            parent.side, parent.price_tick, slice_size, hidden, stored_peak, t
        )

    def _place_order(
        self,
        side: str,
        price_tick: int,
        size: float,
        hidden: float,
        peak: float,
        t: float,
    ) -> None:
        oid = self._new_order_id()
        eid = self._emit(oid, side, price_tick, "created", size, 0.0, t)
        order = _Order(oid, side, price_tick, size, eid, hidden, peak)
        self.orders[oid] = order
        levels = self.bid_levels if side == "bid" else self.ask_levels
        levels.setdefault(price_tick, deque()).append(oid)
        self._add_live(oid)

    # ── Live-order bookkeeping (O(1) random pick + removal) ──────────

    def _add_live(self, oid: int) -> None:
        self.live_pos[oid] = len(self.live_ids)
        self.live_ids.append(oid)

    def _remove_live(self, oid: int) -> None:
        pos = self.live_pos.pop(oid, None)
        if pos is None:
            return
        last = self.live_ids.pop()
        if last != oid:
            self.live_ids[pos] = last
            self.live_pos[last] = pos

    # ── Draw helpers ─────────────────────────────────────────────────

    def _draw_offset(self) -> int:
        k = int(self.rng.geometric(self.cfg.level_decay)) - 1
        return min(k, self.cfg.depth_levels - 1)

    def _draw_size(self, mean: float) -> float:
        x = float(self.rng.exponential(mean))
        x = max(x, self.cfg.min_size)
        size = self._round_vol(x)
        if size <= 0:
            size = self._round_vol(self.cfg.min_size)
        return size

    def _round_vol(self, x: float) -> float:
        return round(x, self.cfg.volume_decimals)

    def _new_order_id(self) -> int:
        oid = self.next_order_id
        self.next_order_id += 1
        return oid

    def _emit(
        self,
        oid: int,
        side: str,
        price_tick: int,
        action: str,
        volume: float,
        fill: float,
        t: float,
    ) -> int:
        eid = self.next_event_id
        self.next_event_id += 1
        self.events.append((eid, oid, t, price_tick, volume, fill, action, side))
        return eid

    # ── Frame construction ───────────────────────────────────────────

    def _timestamps(self, seconds: np.ndarray) -> pd.DatetimeIndex:
        # ``Timestamp.value`` is UTC nanoseconds since the epoch (a tz-naive
        # anchor is read as UTC), so adding the per-event offsets and parsing
        # back with ``utc=True`` yields the canonical tz-aware UTC ns clock.
        start_ns = self.cfg.start_time.value
        offsets = np.round(seconds * 1_000_000_000).astype(np.int64)
        return pd.to_datetime(start_ns + offsets, utc=True).astype(UTC_NS_DTYPE)

    def _build_events_frame(self) -> pd.DataFrame:
        rows = self.events
        event_id = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
        order_id = np.fromiter((r[1] for r in rows), dtype=np.int64, count=len(rows))
        seconds = np.fromiter((r[2] for r in rows), dtype=np.float64, count=len(rows))
        ticks = np.fromiter((r[3] for r in rows), dtype=np.int64, count=len(rows))
        volume = np.fromiter((r[4] for r in rows), dtype=np.float64, count=len(rows))
        fill = np.fromiter((r[5] for r in rows), dtype=np.float64, count=len(rows))
        actions = [r[6] for r in rows]
        directions = [r[7] for r in rows]

        timestamp = self._timestamps(seconds)
        price = np.round(ticks * self.cfg.tick_size, self.cfg.price_decimals)

        return pd.DataFrame(
            {
                "event_id": event_id,
                "id": order_id,
                "timestamp": timestamp,
                "exchange_timestamp": timestamp.copy(),
                "price": price,
                "volume": volume,
                "action": pd.Categorical(
                    actions,
                    categories=["created", "changed", "deleted"],
                    ordered=True,
                ),
                "direction": pd.Categorical(
                    directions, categories=["bid", "ask"], ordered=True
                ),
                "fill": fill,
                "original_number": event_id.copy(),
                "raw_event_type": pd.NA,
            }
        )

    def _build_trades_frame(self) -> pd.DataFrame:
        rows = self.trades
        n = len(rows)
        seconds = np.fromiter((r[0] for r in rows), dtype=np.float64, count=n)
        ticks = np.fromiter((r[1] for r in rows), dtype=np.int64, count=n)
        volume = np.fromiter((r[2] for r in rows), dtype=np.float64, count=n)
        sides = [r[3] for r in rows]
        maker_eid = np.fromiter((r[4] for r in rows), dtype=np.int64, count=n)
        taker_eid = np.fromiter((r[5] for r in rows), dtype=np.int64, count=n)
        maker = np.fromiter((r[6] for r in rows), dtype=np.int64, count=n)
        taker = np.fromiter((r[7] for r in rows), dtype=np.int64, count=n)

        price = np.round(ticks * self.cfg.tick_size, self.cfg.price_decimals)

        return pd.DataFrame(
            {
                "timestamp": self._timestamps(seconds),
                "price": price,
                "volume": volume,
                "direction": pd.Categorical(
                    sides, categories=["buy", "sell"], ordered=True
                ),
                "maker_event_id": maker_eid,
                "taker_event_id": taker_eid,
                "maker": maker,
                "taker": taker,
                # original_number equals event_id in synthetic output, so the
                # provenance columns mirror the event ids.
                "maker_og": maker_eid.copy(),
                "taker_og": taker_eid.copy(),
            }
        )


class SyntheticLoader:
    """Serve a pre-generated session's events to :class:`~ob_analytics.pipeline.Pipeline`.

    Satisfies the :class:`~ob_analytics.protocols.EventLoader` protocol, so a
    synthetic session runs through the real pipeline::

        from ob_analytics import Pipeline
        from ob_analytics.synth import (
            generate_session, SyntheticLoader, SyntheticTradeSource,
        )

        session = generate_session(seed=1)
        result = Pipeline(
            loader=SyntheticLoader(session),
            trade_source=SyntheticTradeSource(session),
        ).run(source=None)
    """

    def __init__(self, session: SynthSession) -> None:
        self._session = session

    def load(self, source: Any = None) -> pd.DataFrame:
        return self._session.events.copy()


class SyntheticTradeSource:
    """Serve a pre-generated session's trades to :class:`~ob_analytics.pipeline.Pipeline`.

    Satisfies the :class:`~ob_analytics.protocols.TradeSource` protocol. See
    :class:`SyntheticLoader` for the paired usage.
    """

    def __init__(self, session: SynthSession) -> None:
        self._session = session

    def load(self, events: pd.DataFrame, source: Any = None) -> pd.DataFrame:
        return self._session.trades.copy()
