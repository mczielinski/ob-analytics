"""LOBSTER data format support.

Provides loader, matcher, trade inferrer, writer, and format descriptor
for the LOBSTER limit-order-book data set
(https://lobsterdata.com).

LOBSTER message files contain six headerless columns::

    Time, EventType, OrderID, Size, Price, Direction

Event types:
    1 = Submission of a new limit order
    2 = Cancellation (partial deletion)
    3 = Deletion (total cancellation)
    4 = Execution of a visible limit order
    5 = Execution of a hidden limit order
    6 = Cross trade (non-book)
    7 = Trading halt indicator

Prices are integers scaled by 10 000 (e.g. 2459800 = $245.98).
Timestamps are seconds after midnight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ob_analytics._utils import (
    datetime_to_seconds_after_midnight,
    seconds_after_midnight_to_datetime,
)
from ob_analytics.config import PipelineConfig
from ob_analytics.protocols import (
    DataWriter,
    EventLoader,
    Format,
    MatchingEngine,
    TradeInferrer,
)


# ── Constants ─────────────────────────────────────────────────────────

_LOBSTER_COLS = ["time", "event_type", "id", "volume", "price", "direction"]

_EVENT_TYPE_TO_ACTION: dict[int, str] = {
    1: "created",
    2: "changed",
    3: "deleted",
    4: "changed",
    5: "changed",
}

_ACTION_TO_EVENT_TYPE: dict[str, int] = {
    "created": 1,
    "deleted": 3,
}

_DIRECTION_MAP: dict[int, str] = {1: "bid", -1: "ask"}
_DIRECTION_REVERSE: dict[str, int] = {"bid": 1, "ask": -1}

_DUMMY_BID_PRICE = -9999999999
_DUMMY_ASK_PRICE = 9999999999


# ── LobsterLoader ────────────────────────────────────────────────────


class LobsterLoader:
    """Load raw limit-order events from LOBSTER message files.

    Satisfies the :class:`~ob_analytics.protocols.EventLoader` protocol.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration.
    trading_date : str or pd.Timestamp
        The calendar date of the trading session (LOBSTER timestamps are
        seconds after midnight and need a date anchor).
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        trading_date: str | pd.Timestamp,
    ) -> None:
        self._config = config or PipelineConfig()
        self._trading_date = pd.Timestamp(trading_date).normalize()
        self._trading_halts: pd.DataFrame | None = None
        self._cross_trades: pd.DataFrame | None = None

    @property
    def trading_halts(self) -> pd.DataFrame | None:
        """Trading halt events extracted during the last :meth:`load`, if any."""
        return self._trading_halts

    @property
    def cross_trades(self) -> pd.DataFrame | None:
        """Cross-trade events extracted during the last :meth:`load`, if any."""
        return self._cross_trades

    def load(self, source: Any) -> pd.DataFrame:
        """Load LOBSTER message data and return a cleaned events DataFrame.

        Parameters
        ----------
        source : str, Path, or directory
            Path to a LOBSTER message CSV, or a directory containing
            message/orderbook file pairs.  When a directory is given the
            loader auto-discovers files by the LOBSTER naming convention.

        Returns
        -------
        pandas.DataFrame
        """
        source = Path(source)
        msg_path = self._resolve_message_file(source)

        logger.info("LobsterLoader: reading {}", msg_path)
        raw = pd.read_csv(msg_path, header=None, names=_LOBSTER_COLS)

        cfg = self._config
        divisor = cfg.price_divisor

        raw["price"] = (raw["price"] / divisor).round(cfg.price_decimals)
        raw["volume"] = raw["volume"].astype(float).round(cfg.volume_decimals)

        raw["timestamp"] = seconds_after_midnight_to_datetime(
            raw["time"], self._trading_date
        )
        raw["exchange_timestamp"] = raw["timestamp"]

        raw["raw_event_type"] = raw["event_type"]
        raw["direction"] = raw["direction"].map(_DIRECTION_MAP)

        # Separate special event types before mapping actions
        halts = raw[raw["event_type"] == 7].copy()
        cross = raw[raw["event_type"] == 6].copy()
        self._trading_halts = halts if not halts.empty else None
        self._cross_trades = cross if not cross.empty else None

        events = raw[raw["event_type"].isin(_EVENT_TYPE_TO_ACTION)].copy()
        events["action"] = events["event_type"].map(_EVENT_TYPE_TO_ACTION)

        events["action"] = pd.Categorical(
            events["action"],
            categories=["created", "changed", "deleted"],
            ordered=True,
        )
        events["direction"] = pd.Categorical(
            events["direction"],
            categories=["bid", "ask"],
            ordered=True,
        )

        # Fill column: partial cancellations (2) and executions (4, 5)
        # carry a volume delta that price_level_volume subtracts.
        events["fill"] = np.where(
            events["event_type"].isin([2, 4, 5]),
            events["volume"],
            0.0,
        )

        # Hidden order executions (type 5) all share id=0 which causes
        # downstream misclassification as "pacman" (multiple prices for
        # the same id).  Assign each a unique synthetic id so they are
        # handled independently.
        hidden_mask = events["event_type"] == 5
        n_hidden = hidden_mask.sum()
        if n_hidden > 0:
            max_id = events["id"].max()
            events.loc[hidden_mask, "id"] = np.arange(max_id + 1, max_id + 1 + n_hidden)

        events = events.reset_index(drop=True)
        events["event_id"] = np.arange(1, len(events) + 1)
        events["original_number"] = events["event_id"]

        events = events[
            [
                "id",
                "timestamp",
                "exchange_timestamp",
                "price",
                "volume",
                "action",
                "direction",
                "fill",
                "event_id",
                "original_number",
                "raw_event_type",
            ]
        ]

        logger.info(
            "LobsterLoader: {} events ({} executions, {} halts, {} cross trades)",
            len(events),
            (events["raw_event_type"].isin([4, 5])).sum(),
            len(halts),
            len(cross),
        )

        # Discover and store the orderbook file path for depth computation
        self._orderbook_path = self._resolve_orderbook_file(source)

        return events

    @property
    def orderbook_path(self) -> Path | None:
        """Path to the LOBSTER orderbook file discovered during :meth:`load`."""
        return self._orderbook_path

    @staticmethod
    def _resolve_message_file(source: Path) -> Path:
        """Find the message CSV from *source* (file or directory)."""
        if source.is_file():
            return source
        if source.is_dir():
            candidates = sorted(source.glob("*_message*.csv"))
            if not candidates:
                candidates = sorted(source.glob("*message*.csv"))
            if candidates:
                return candidates[0]
            raise FileNotFoundError(f"No LOBSTER message file found in {source}")
        raise FileNotFoundError(f"Path does not exist: {source}")

    @staticmethod
    def _resolve_orderbook_file(source: Path) -> Path | None:
        """Find the orderbook CSV from *source* (file or directory)."""
        source = Path(source)
        if source.is_dir():
            candidates = sorted(source.glob("*_orderbook*.csv"))
            if not candidates:
                candidates = sorted(source.glob("*orderbook*.csv"))
            if candidates:
                return candidates[0]
        return None


# ── LobsterMatcher ───────────────────────────────────────────────────


class LobsterMatcher:
    """Pass-through matcher for LOBSTER data.

    LOBSTER provides single-sided execution events (the resting order
    only), so there are no bid/ask pairs to match.  This matcher simply
    adds the required ``matching_event`` column filled with NaN.

    Satisfies the :class:`~ob_analytics.protocols.MatchingEngine` protocol.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    def match(self, events: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()
        events["matching_event"] = np.nan
        return events


# ── LobsterTradeInferrer ─────────────────────────────────────────────


class LobsterTradeInferrer:
    """Infer trades directly from LOBSTER execution events.

    In LOBSTER, each execution event (type 4 or 5) represents one side
    of a trade -- the **resting** (maker) order.  This inferrer builds
    trade records directly from those events without requiring matched
    pairs.

    Satisfies the :class:`~ob_analytics.protocols.TradeInferrer` protocol.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    def infer_trades(self, events: pd.DataFrame) -> pd.DataFrame:
        """Build a trades DataFrame from LOBSTER execution events.

        Parameters
        ----------
        events : pandas.DataFrame
            Events with ``raw_event_type`` column populated.

        Returns
        -------
        pandas.DataFrame
            Trades with ``timestamp``, ``price``, ``volume``,
            ``direction``, ``maker_event_id``, ``taker_event_id``,
            ``maker``, ``taker``.
        """
        execs = (
            events[events["raw_event_type"].isin([4, 5])].copy().reset_index(drop=True)
        )

        if execs.empty:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "price",
                    "volume",
                    "direction",
                    "maker_event_id",
                    "taker_event_id",
                    "maker",
                    "taker",
                    "maker_og",
                    "taker_og",
                ]
            )

        # Direction inversion: execution of a resting ask = buyer-initiated
        trade_direction = np.where(execs["direction"] == "ask", "buy", "sell")

        maker_event_id = execs["event_id"].values
        maker_id = execs["id"].values
        maker_og = execs["original_number"].values

        # Best-effort taker identification
        taker_event_id = self._find_takers(events, execs)
        id_to_id = dict(zip(events["event_id"], events["id"]))
        id_to_og = dict(zip(events["event_id"], events["original_number"]))
        taker_id = pd.array(
            [id_to_id.get(t) if pd.notna(t) else pd.NA for t in taker_event_id],
            dtype="object",
        )
        taker_og = pd.array(
            [id_to_og.get(t) if pd.notna(t) else pd.NA for t in taker_event_id],
            dtype="object",
        )

        trades = pd.DataFrame(
            {
                "timestamp": execs["timestamp"].values,
                "price": execs["price"].values,
                "volume": execs["volume"].values,
                "direction": pd.Categorical(
                    trade_direction, categories=["buy", "sell"], ordered=True
                ),
                "maker_event_id": maker_event_id,
                "taker_event_id": taker_event_id,
                "maker": maker_id,
                "taker": taker_id,
                "maker_og": maker_og,
                "taker_og": taker_og,
            }
        )
        trades = trades.sort_values("timestamp", kind="stable").reset_index(drop=True)

        logger.info(
            "LobsterTradeInferrer: {} trades ({} with identified taker)",
            len(trades),
            trades["taker_event_id"].notna().sum(),
        )
        return trades

    @staticmethod
    def _find_takers(
        all_events: pd.DataFrame, execs: pd.DataFrame
    ) -> pd.api.extensions.ExtensionArray:
        """Best-effort heuristic to identify taker orders.

        For each execution, look for the most recent type-1 submission
        on the **opposite** side at a marketable price.
        """
        submissions = all_events[all_events["raw_event_type"] == 1]

        if submissions.empty:
            return pd.array([pd.NA] * len(execs), dtype="Int64")

        result = pd.array([pd.NA] * len(execs), dtype="Int64")

        # Position lookup keyed by event_id -- avoids per-row linear scans.
        eid_to_pos = pd.Series(np.arange(len(execs)), index=execs["event_id"])

        for side, opp_side in [("bid", "ask"), ("ask", "bid")]:
            side_execs = execs[execs["direction"] == side]
            if side_execs.empty:
                continue
            opp_subs = submissions[submissions["direction"] == opp_side].sort_values(
                "timestamp", kind="stable"
            )

            if opp_subs.empty:
                continue

            merged = pd.merge_asof(
                side_execs[["event_id", "timestamp", "price"]].sort_values("timestamp"),
                opp_subs[["event_id", "timestamp", "price"]].rename(
                    columns={
                        "event_id": "taker_eid",
                        "price": "sub_price",
                    }
                ),
                on="timestamp",
                direction="backward",
            )

            if side == "bid":
                marketable = merged["sub_price"] <= merged["price"]
            else:
                marketable = merged["sub_price"] >= merged["price"]

            matched = merged.loc[marketable, ["event_id", "taker_eid"]].dropna(
                subset=["taker_eid"]
            )
            if matched.empty:
                continue

            positions = eid_to_pos.loc[matched["event_id"].to_numpy()].to_numpy()
            result[positions] = matched["taker_eid"].astype("int64").to_numpy()

        return result


# ── LobsterWriter ────────────────────────────────────────────────────


class LobsterWriter:
    """Write pipeline events back to LOBSTER dual-file format.

    Satisfies the :class:`~ob_analytics.protocols.DataWriter` protocol.

    Parameters
    ----------
    trading_date : str or pd.Timestamp
        Calendar date of the session.
    price_divisor : int
        Multiplier to convert decimal prices back to LOBSTER integers.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        trading_date: str | pd.Timestamp,
        price_divisor: int | None = None,
    ) -> None:
        self._config = config or PipelineConfig()
        self._trading_date = pd.Timestamp(trading_date).normalize()
        # Explicit price_divisor overrides config (for manual construction)
        self._price_divisor = (
            price_divisor if price_divisor is not None else self._config.price_divisor
        )

    def write(
        self,
        data: dict[str, pd.DataFrame],
        dest: str | Path,
        *,
        ticker: str = "DATA",
        num_levels: int = 10,
        **kwargs: Any,
    ) -> tuple[Path, Path]:
        """Write events to LOBSTER message + orderbook files.

        Parameters
        ----------
        data : dict
            Must contain ``"events"`` key.
        dest : str or Path
            Output directory.
        ticker : str
            Ticker symbol for filename.
        num_levels : int
            Number of orderbook levels to write.

        Returns
        -------
        tuple of Path
            ``(message_path, orderbook_path)``
        """
        events = data["events"].copy()
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)

        date_str = self._trading_date.strftime("%Y-%m-%d")
        base = f"{ticker}_{date_str}_{num_levels}"
        msg_path = dest / f"{base}_message.csv"
        ob_path = dest / f"{base}_orderbook.csv"

        msg_df = self._events_to_message(events)
        msg_df.to_csv(msg_path, index=False, header=False)

        ob_df = self._reconstruct_orderbook(events, num_levels)
        ob_df.to_csv(ob_path, index=False, header=False)

        logger.info(
            "LobsterWriter: wrote {} events to {} and {}",
            len(msg_df),
            msg_path.name,
            ob_path.name,
        )
        return msg_path, ob_path

    def _events_to_message(self, events: pd.DataFrame) -> pd.DataFrame:
        """Convert pipeline events back to LOBSTER message format."""
        if (
            "raw_event_type" in events.columns
            and events["raw_event_type"].notna().any()
        ):
            event_type = events["raw_event_type"].astype(int)
        else:
            event_type = (
                events["action"].map(_ACTION_TO_EVENT_TYPE).fillna(2).astype(int)
            )

        midnight = self._trading_date
        time_seconds = datetime_to_seconds_after_midnight(events["timestamp"], midnight)

        price_int = (events["price"] * self._price_divisor).round(0).astype(int)
        direction_int = events["direction"].astype(str).map(_DIRECTION_REVERSE)

        return pd.DataFrame(
            {
                "time": time_seconds,
                "event_type": event_type,
                "id": events["id"],
                "volume": events["volume"],
                "price": price_int,
                "direction": direction_int,
            }
        )

    def _reconstruct_orderbook(
        self, events: pd.DataFrame, num_levels: int
    ) -> pd.DataFrame:
        """Rebuild the orderbook state at each event timestamp.

        The book is stateful so the per-event update is inherently
        sequential, but DataFrame row access is the dominant cost.  We
        extract every column we need to a numpy array up front and
        iterate via indexed access, which is roughly an order of
        magnitude faster than ``DataFrame.iterrows()``.
        """
        book: dict[str, dict[float, float]] = {"bid": {}, "ask": {}}
        rows: list[list[float]] = []

        n = len(events)
        actions = events["action"].astype(str).to_numpy()
        directions = events["direction"].astype(str).to_numpy()
        prices = events["price"].to_numpy(dtype=np.float64)
        volumes = events["volume"].to_numpy(dtype=np.float64)
        if "raw_event_type" in events.columns:
            raw_types = events["raw_event_type"].to_numpy(dtype=np.float64)
        else:
            raw_types = np.full(n, np.nan, dtype=np.float64)

        # Membership test against a set is O(1); covers both int and
        # float scalars because hash(2) == hash(2.0).
        decrement_raw_types = {2, 4, 5}

        for i in range(n):
            action = actions[i]
            direction = directions[i]
            price = prices[i]
            volume = volumes[i]
            side = book[direction]

            if action == "created":
                side[price] = side.get(price, 0.0) + volume
            elif action == "deleted":
                if price in side:
                    side[price] -= volume
                    if side[price] <= 1e-12:
                        del side[price]
            elif action == "changed":
                raw_type = raw_types[i]
                if raw_type in decrement_raw_types and price in side:
                    side[price] -= volume
                    if side[price] <= 1e-12:
                        del side[price]

            rows.append(self._snapshot_row(book, num_levels))

        cols: list[str] = []
        for i in range(1, num_levels + 1):
            cols.extend(
                [
                    f"ask_price_{i}",
                    f"ask_size_{i}",
                    f"bid_price_{i}",
                    f"bid_size_{i}",
                ]
            )

        ob = pd.DataFrame(rows, columns=cols)

        for col in cols:
            if "price" in col:
                ob[col] = (ob[col] * self._price_divisor).round(0).astype(int)

        return ob

    @staticmethod
    def _snapshot_row(
        book: dict[str, dict[float, float]], num_levels: int
    ) -> list[float]:
        """Build a single orderbook snapshot row."""
        asks_sorted = sorted(book["ask"].items())
        bids_sorted = sorted(book["bid"].items(), reverse=True)

        row: list[float] = []
        for i in range(num_levels):
            if i < len(asks_sorted):
                row.extend([asks_sorted[i][0], asks_sorted[i][1]])
            else:
                row.extend([_DUMMY_ASK_PRICE, 0])
            if i < len(bids_sorted):
                row.extend([bids_sorted[i][0], bids_sorted[i][1]])
            else:
                row.extend([_DUMMY_BID_PRICE, 0])

        return row


# ── LOBSTER depth computation ─────────────────────────────────────────


def lobster_depth_from_orderbook(
    events: pd.DataFrame,
    orderbook_path: Path,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute depth and depth summary from the LOBSTER orderbook file.

    The LOBSTER orderbook file is ground truth: it records the complete
    visible book state after every message event.  This function converts
    it into the ``(depth, depth_summary)`` pair the pipeline expects,
    avoiding the need to reconstruct depth from message events (which
    fails when events reference pre-market orders absent from the
    message file).

    Parameters
    ----------
    events : pandas.DataFrame
        Events DataFrame (used only for timestamps and event IDs).
    orderbook_path : Path
        Path to the LOBSTER orderbook CSV.
    config : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    tuple of (depth DataFrame, depth_summary DataFrame)
    """
    from ob_analytics.depth import DepthMetricsEngine

    ob_raw = pd.read_csv(orderbook_path, header=None).values
    num_levels = ob_raw.shape[1] // 4

    book_events = events[events["raw_event_type"].isin([1, 2, 3, 4, 5])].reset_index(
        drop=True
    )

    n_ob, n_ev = ob_raw.shape[0], len(book_events)
    if n_ob != n_ev:
        logger.warning(
            "lobster_depth: orderbook rows ({}) != book events ({}); using min",
            n_ob,
            n_ev,
        )
    n = min(n_ob, n_ev)

    timestamps = book_events["timestamp"].values[:n]
    event_ids = book_events["event_id"].values[:n]

    # Vectorised extraction of prices and volumes per level, then diff
    # consecutive rows to find changes.
    depth_rows: list[dict] = []
    prev_levels: dict[tuple[str, float], float] = {}

    # Pre-compute rounded price arrays for all levels in one shot
    price_divisor = config.price_divisor
    price_dec = config.price_decimals

    ask_price_cols = [j * 4 for j in range(num_levels)]
    ask_size_cols = [j * 4 + 1 for j in range(num_levels)]
    bid_price_cols = [j * 4 + 2 for j in range(num_levels)]
    bid_size_cols = [j * 4 + 3 for j in range(num_levels)]

    ask_prices_all = ob_raw[:n, ask_price_cols]
    ask_sizes_all = ob_raw[:n, ask_size_cols]
    bid_prices_all = ob_raw[:n, bid_price_cols]
    bid_sizes_all = ob_raw[:n, bid_size_cols]

    for i in range(n):
        curr_levels: dict[tuple[str, float], float] = {}

        for j in range(num_levels):
            ap = ask_prices_all[i, j]
            av = ask_sizes_all[i, j]
            if ap != _DUMMY_ASK_PRICE and av > 0:
                price = round(ap / price_divisor, price_dec)
                curr_levels[("ask", price)] = curr_levels.get(("ask", price), 0) + av

            bp = bid_prices_all[i, j]
            bv = bid_sizes_all[i, j]
            if bp != _DUMMY_BID_PRICE and bv > 0:
                price = round(bp / price_divisor, price_dec)
                curr_levels[("bid", price)] = curr_levels.get(("bid", price), 0) + bv

        all_keys = set(prev_levels) | set(curr_levels)
        for key in all_keys:
            pv = prev_levels.get(key, 0.0)
            cv = curr_levels.get(key, 0.0)
            if pv != cv:
                side, price = key
                depth_rows.append(
                    {
                        "event_id": event_ids[i],
                        "timestamp": timestamps[i],
                        "price": price,
                        "volume": cv,
                        "direction": side,
                    }
                )
        prev_levels = curr_levels

    depth = pd.DataFrame(depth_rows)
    if depth.empty:
        depth = pd.DataFrame(
            columns=["event_id", "timestamp", "price", "volume", "direction"]
        )

    depth["direction"] = pd.Categorical(
        depth["direction"], categories=["bid", "ask"], ordered=True
    )
    depth = depth.sort_values("timestamp", kind="stable").reset_index(drop=True)

    logger.info("lobster_depth: {} depth rows from orderbook", len(depth))

    engine = DepthMetricsEngine(config)
    depth_summary = engine.compute(depth)

    return depth, depth_summary


# ── LobsterFormat descriptor ─────────────────────────────────────────


@dataclass
class LobsterFormat(Format):
    """Format descriptor for LOBSTER limit-order-book data.

    Parameters
    ----------
    trading_date : str or pd.Timestamp
        Calendar date of the trading session.
    """

    name: str = field(default="lobster", init=False, repr=False)
    trading_date: str | pd.Timestamp = field()

    _loader: LobsterLoader | None = field(default=None, repr=False, init=False)

    def create_loader(self, config: PipelineConfig) -> EventLoader:
        self._loader = LobsterLoader(config, trading_date=self.trading_date)
        return self._loader

    def create_matcher(self, config: PipelineConfig) -> MatchingEngine:
        return LobsterMatcher(config)

    def create_trade_inferrer(self, config: PipelineConfig) -> TradeInferrer:
        return LobsterTradeInferrer(config)

    def create_writer(self, config: PipelineConfig) -> DataWriter:
        return LobsterWriter(config, trading_date=self.trading_date)

    def compute_depth(
        self,
        events: pd.DataFrame,
        config: Any,
        source: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        ob_path = self._loader.orderbook_path if self._loader is not None else None
        if ob_path is None:
            ob_path = LobsterLoader._resolve_orderbook_file(Path(source))
        if ob_path is None:
            logger.warning(
                "LobsterFormat: no orderbook file found; "
                "falling back to event-based depth"
            )
            return None
        return lobster_depth_from_orderbook(events, ob_path, config)

    def config_defaults(self) -> dict[str, Any]:
        return {
            "price_decimals": 2,
            "price_divisor": 10_000,
            "volume_decimals": 0,
            "match_cutoff_ms": 100,
            "price_jump_threshold": 5.0,
            "skip_zombie_detection": True,
        }
