"""LOBSTER data format support.

Provides loader, trade reader, writer, and format descriptor
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
    attach_ingest_seq,
    datetime_to_seconds_after_midnight,
    empty_trades,
    lots_to_size,
    price_to_ticks,
    seconds_after_midnight_to_datetime,
    size_to_lots,
    validate_columns,
)
from ob_analytics.config import PipelineConfig, SourceSettings
from ob_analytics.depth import DepthMetricsEngine, PriceLevelBook, price_level_volume
from ob_analytics.exceptions import ConfigError
from ob_analytics.protocols import (
    Clocks,
    DataWriter,
    EventLoader,
    FeedType,
    Level,
    RunContext,
    TradeAttribution,
    TradeSource,
)
from ob_analytics.schemas import attach_instrument_identity, time_order_keys

# ── Constants ─────────────────────────────────────────────────────────

_LOBSTER_COLS = ["time", "event_type", "id", "volume", "price", "direction"]

#: Default venue time zone for a LOBSTER session.  LOBSTER distributes US
#: equity data (NASDAQ), whose trading day is US/Eastern; its timestamps are
#: seconds after *local* midnight with no zone, so converting them to the
#: shared UTC clock (issue #154) needs this.  Override per run via
#: ``RunContext(session_tz=...)`` for a non-US-Eastern capture.
LOBSTER_DEFAULT_TZ = "America/New_York"

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

#: The instrument grid of LOBSTER's own data: a cent tick, prices in
#: ten-thousandths of a dollar, and whole shares.
_LOBSTER_CONFIG_DEFAULTS: dict[str, Any] = {
    "tick_size": 0.01,
    "price_decimals": 2,
    "price_divisor": 10_000,
    # LOBSTER quotes whole shares, so one lot is one share and the canonical
    # integer size is the venue's own count.
    "lot_size": 1.0,
    "volume_decimals": 0,
}


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
    session_tz : str, optional
        The venue's local time zone, used to place the session's seconds after
        midnight on the shared UTC clock.  Defaults to
        :data:`LOBSTER_DEFAULT_TZ` (``"America/New_York"``), correct for LOBSTER's
        US equity data.
    venue, symbol : str, optional
        Optional instrument identity.  When either is supplied, the loaded
        frame gains per-row ``venue`` / ``symbol`` columns; ``venue`` falls
        back to ``"lobster"`` when only ``symbol`` is given.  Both ``None``
        (the default) leaves the frame untagged.  The instrument ticker lives
        in the LOBSTER filename, so pass it as ``symbol`` when you want it on
        the rows.
    """

    #: The source venue used to fill the ``venue`` column when identity tagging
    #: is on but no explicit venue was supplied.
    _VENUE = "lobster"

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        trading_date: str | pd.Timestamp,
        session_tz: str = LOBSTER_DEFAULT_TZ,
        venue: str | None = None,
        symbol: str | None = None,
    ) -> None:
        self._config = config or PipelineConfig()
        self._trading_date = pd.Timestamp(trading_date).normalize()
        self._session_tz = session_tz
        self._venue = venue
        self._symbol = symbol
        #: Trading-halt (event_type 7) and cross-trade (event_type 6) rows,
        #: split out during :meth:`load`. Public so callers can append them to
        #: the gallery model's ``analytics`` (LOBSTER-only; ``None`` if absent).
        self.trading_halts: pd.DataFrame | None = None
        self.cross_trades: pd.DataFrame | None = None
        #: Path to the companion LOBSTER orderbook file, discovered during
        #: :meth:`load`. Public so :class:`LobsterSource` can read it for depth
        #: computation without reaching into a private attribute.
        self.orderbook_path: Path | None = None
        #: Number of rows in the message file read by :meth:`load`, every
        #: event type included: the number of rows its orderbook file must
        #: have.
        self.message_rows: int | None = None

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
        self.message_rows = len(raw)

        cfg = self._config
        divisor = cfg.price_divisor

        # Canonical price is integer ticks (issue #155).  LOBSTER stores a raw
        # integer in ten-thousandths of a dollar, so divide by the feed's
        # encoding scale to reach the quote currency, then quantise to ticks.
        raw["price"] = price_to_ticks(raw["price"] / divisor, cfg.tick_size)
        # Canonical size is integer lots (issue #226).  LOBSTER's Size column is
        # already a whole share count and its ``lot_size`` default is 1, so this
        # keeps the venue's own integer rather than routing it through a float.
        raw["volume"] = size_to_lots(raw["volume"], cfg.lot_size)

        raw["timestamp"] = seconds_after_midnight_to_datetime(
            raw["time"], self._trading_date, self._session_tz
        )
        raw["exchange_timestamp"] = raw["timestamp"]

        raw["raw_event_type"] = raw["event_type"]
        raw["direction"] = raw["direction"].map(_DIRECTION_MAP)

        # Separate special event types before mapping actions
        halts = raw[raw["event_type"] == 7].copy()
        cross = raw[raw["event_type"] == 6].copy()
        self.trading_halts = halts if not halts.empty else None
        self.cross_trades = cross if not cross.empty else None

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

        # Canonical volume semantics (see schemas.py): ``volume`` is the
        # order's outstanding size after the event (created/changed) or the
        # size removed (deleted); ``fill`` is the *executed* delta (visible
        # and hidden executions only).  LOBSTER's raw Size column is a
        # per-event delta for cancels/executions, so outstanding size is
        # derived per order; the raw delta is preserved in ``raw_size``.
        events["raw_size"] = events["volume"]
        sizes = events["volume"].to_numpy()
        etypes = events["event_type"].to_numpy()

        # Signed deltas: submissions add, reductions/executions subtract.
        rem_after = (
            pd.Series(np.where(etypes == 1, sizes, -sizes))
            .groupby(events["id"].to_numpy())
            .cumsum()
            .to_numpy()
        )
        # Deleted rows report the size removed (= outstanding immediately
        # before the delete), matching the Bitstamp convention.
        derived = np.where(etypes == 3, rem_after + sizes, rem_after)

        # Orders first seen mid-stream — the pre-existing opening book and
        # hidden executions (which all share LOBSTER's native id=0) — have no
        # submission to anchor the cumsum; their rows keep the raw delta.
        derivable = (
            events.groupby("id")["event_type"].transform("first").to_numpy() == 1
        )
        events["volume"] = np.where(derivable, derived, sizes)

        # Integer lots, like ``volume`` and ``raw_size`` it is taken from: a
        # ``0.0`` here would widen the whole column to float and hand every
        # consumer base-asset-looking floats that are really lot counts.
        events["fill"] = np.where(
            events["event_type"].isin([4, 5]),
            events["raw_size"],
            0,
        )

        # ``original_number`` captures the 1-based row in the source message
        # file (gaps appear where halt/cross/other event types were filtered
        # out above). ``event_id`` is a contiguous 1..N surrogate. These match
        # the Bitstamp loader's convention so trade provenance (maker_og /
        # taker_og) is comparable across formats.
        events = events.reset_index().rename(columns={"index": "original_number"})
        events["original_number"] = events["original_number"] + 1
        events["event_id"] = np.arange(1, len(events) + 1)

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
                "raw_size",
            ]
        ]

        logger.info(
            "LobsterLoader: {} events ({} executions, {} halts, {} cross trades)",
            len(events),
            (events["raw_event_type"].isin([4, 5])).sum(),
            len(halts),
            len(cross),
        )

        # Local monotonic ingest counter (opt-in, so the default frame is
        # unchanged).  LOBSTER message files carry no venue sequence, so only
        # ``ingest_seq`` is attached; ``sequence`` stays absent.
        if self._config.track_sequence:
            events = attach_ingest_seq(events)

        # Discover and store the orderbook file path for depth computation
        self.orderbook_path = self._resolve_orderbook_file(source)

        # Optional per-row instrument identity (issue #147); a no-op unless a
        # venue/symbol was supplied for the run.
        events = attach_instrument_identity(
            events,
            venue=self._venue,
            symbol=self._symbol,
            default_venue=self._VENUE,
        )

        return events

    @staticmethod
    def _glob_lobster_file(directory: Path, kind: str) -> Path | None:
        """Return the first ``*{kind}*.csv`` in *directory* (or None).

        Prefers the underscored ``*_{kind}*.csv`` form (LOBSTER's own naming)
        and falls back to the looser ``*{kind}*.csv``.
        """
        candidates = sorted(directory.glob(f"*_{kind}*.csv"))
        if not candidates:
            candidates = sorted(directory.glob(f"*{kind}*.csv"))
        return candidates[0] if candidates else None

    @staticmethod
    def _resolve_message_file(source: Path) -> Path:
        """Find the message CSV from *source* (file or directory)."""
        if source.is_file():
            return source
        if source.is_dir():
            found = LobsterLoader._glob_lobster_file(source, "message")
            if found is not None:
                return found
            raise FileNotFoundError(f"No LOBSTER message file found in {source}")
        raise FileNotFoundError(f"Path does not exist: {source}")

    @staticmethod
    def _resolve_orderbook_file(source: Path) -> Path | None:
        """Find the orderbook CSV from *source* (file or directory)."""
        source = Path(source)
        if source.is_dir():
            return LobsterLoader._glob_lobster_file(source, "orderbook")
        return None


# ── LobsterTradeReader ────────────────────────────────────────────────


class LobsterTradeReader:
    """Build trades directly from LOBSTER execution events.

    In LOBSTER, each execution event (type 4 or 5) represents the
    resting (maker) side of a trade.  This reader builds trade records
    directly from those rows in the events frame; no matching is needed
    because the data already pairs maker rows with executions.

    Satisfies the :class:`~ob_analytics.protocols.TradeSource` protocol.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    def load(self, events: pd.DataFrame, source: Any) -> pd.DataFrame:
        """Build a trades DataFrame from LOBSTER execution events.

        Parameters
        ----------
        events : pandas.DataFrame
            Events with ``raw_event_type`` column populated.
        source
            Unused; trade information is embedded in *events*.

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
            return empty_trades()

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
                # ``.array`` keeps the tz-aware UTC dtype (``.values`` would
                # drop the zone to a naive numpy datetime64).
                "timestamp": execs["timestamp"].array,
                "price": execs["price"].values,
                # Executed quantity: `fill` carries the raw executed delta
                # (`volume` is the order's outstanding size after the event).
                "volume": execs["fill"].values,
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
            "LobsterTradeReader: {} trades ({} with identified taker)",
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


def _require_raw_units(
    frame: pd.DataFrame, name: str, *, whole_float_volume: bool = False
) -> None:
    """Refuse a table whose prices or sizes are in display units.

    Integer columns are ticks and lots; float columns are display units.  With
    *whole_float_volume*, a float size column of whole numbers is read as
    lots, because :func:`~ob_analytics.depth.price_level_volume` can return
    one for a depth table.
    """
    volume = frame["volume"].to_numpy()
    if not pd.api.types.is_integer_dtype(frame["price"]) or not (
        pd.api.types.is_integer_dtype(volume)
        or (whole_float_volume and bool(np.all(volume == np.round(volume))))
    ):
        raise ConfigError(
            f"LobsterWriter: the {name} table's price must be integer ticks "
            "and its volume whole lots, as the pipeline returns them, not "
            "display units."
        )


class LobsterWriter:
    """Write a run as a LOBSTER message file and orderbook file.

    The message file has one row per event.  The orderbook file has one row
    per message: the book after that event, taken from the run's depth table.
    So the file holds the book the run computed, whatever source the run came
    from.  Reading the files back with :class:`LobsterSource` gives the run's
    book after every event: the depth summary read back equals the run's, as
    far as the written levels reach.

    Satisfies the :class:`~ob_analytics.protocols.DataWriter` protocol.

    Parameters
    ----------
    config : PipelineConfig, optional
        The run's configuration: its ``tick_size``, ``lot_size`` and
        ``price_divisor``.  Defaults to LOBSTER's own grid (a cent tick,
        ``price_divisor=10_000``, whole shares), as :class:`LobsterSource`
        sets it.
    trading_date : str or pd.Timestamp
        Calendar date of the session.
    session_tz : str, optional
        The venue's local time zone, used to convert the shared UTC clock back
        to LOBSTER's seconds after local midnight.  Defaults to
        :data:`LOBSTER_DEFAULT_TZ`; must match the value used on load.
    price_divisor : int
        Multiplier to convert decimal prices back to LOBSTER integers.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        trading_date: str | pd.Timestamp,
        session_tz: str = LOBSTER_DEFAULT_TZ,
        price_divisor: int | None = None,
    ) -> None:
        self._config = config or PipelineConfig(**_LOBSTER_CONFIG_DEFAULTS)
        self._trading_date = pd.Timestamp(trading_date).normalize()
        self._session_tz = session_tz
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
        """Write a run to a LOBSTER message file and orderbook file.

        Parameters
        ----------
        data : dict
            The run's tables.  ``"events"`` is required.  ``"depth"`` is the
            depth table the orderbook file is written from; when it is
            missing, it is computed from the events with
            :func:`~ob_analytics.depth.price_level_volume`, which needs the
            ``type`` column that
            :func:`~ob_analytics.analytics.set_order_types` adds.
        dest : str or Path
            Output directory.
        ticker : str
            Ticker symbol for filename.
        num_levels : int
            Number of price levels per side in the orderbook file.  A book
            with fewer levels is padded with LOBSTER's dummy prices and zero
            sizes; a book with more is cut to the levels nearest the touch.

        Returns
        -------
        tuple of Path
            ``(message_path, orderbook_path)``

        Raises
        ------
        ConfigError
            If the run has no events, the events lack a column, two events
            share an ``event_id``, the depth table names an event the events
            do not have, the prices and sizes of either table are not whole
            ticks and lots, *num_levels* is below 1, or one tick is not a
            whole number of ``1 / price_divisor`` units.
        """
        if num_levels < 1:
            raise ConfigError(
                f"LobsterWriter: num_levels must be at least 1, got {num_levels}."
            )
        self._price_units()
        # A plain 0..n-1 index: frames joined with ``pd.concat`` can repeat
        # labels, which the depth computation cannot align on.
        events = data["events"].reset_index(drop=True)
        if events.empty:
            raise ConfigError(
                "LobsterWriter: the run has no events to write.  A LOBSTER "
                "message file holds per-order events, which an L2 run has not."
            )
        validate_columns(
            events,
            {"event_id", "id", "timestamp", "price", "volume", "direction", "action"},
            "LobsterWriter",
        )
        _require_raw_units(events, "events")
        if not events["event_id"].is_unique:
            raise ConfigError(
                "LobsterWriter: each event must have its own event_id, as in "
                "the events table of one run."
            )
        depth = data.get("depth")
        if depth is None:
            if "type" not in events.columns:
                raise ConfigError(
                    "LobsterWriter: pass the run's depth table, or events with "
                    "the type column that set_order_types adds, so the depth "
                    "can be computed from them."
                )
            depth = price_level_volume(events)
        # A LOBSTER file is in time order, and the depth summary replays the
        # depth in the canonical event order, so both files follow it.
        events = events.sort_values(by=time_order_keys(events), kind="stable")
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)

        date_str = self._trading_date.strftime("%Y-%m-%d")
        base = f"{ticker}_{date_str}_{num_levels}"
        msg_path = dest / f"{base}_message.csv"
        ob_path = dest / f"{base}_orderbook.csv"

        msg_df = self._events_to_message(events)
        msg_df.to_csv(msg_path, index=False, header=False)

        ob_df = self._orderbook_rows(events, depth, num_levels)
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
        time_seconds = datetime_to_seconds_after_midnight(
            events["timestamp"], midnight, self._session_tz
        )

        price_int = self._raw_prices(events["price"])
        direction_int = events["direction"].astype(str).map(_DIRECTION_REVERSE)

        # LOBSTER's Size column is the per-event delta; loader-produced frames
        # carry it as ``raw_size`` (``volume`` holds outstanding size under
        # the canonical schema).  Frames built without it fall back to
        # ``volume``, which equals the delta for created/deleted rows.
        size = (
            events["raw_size"]
            if "raw_size" in events.columns and events["raw_size"].notna().any()
            else events["volume"]
        )
        # Restore the venue's own share count from integer lots (issue #226).
        # LOBSTER's ``lot_size`` is 1, so this is the identity on a LOBSTER
        # round-trip; it matters for a frame loaded from another venue's grid.
        size = lots_to_size(
            size, self._config.lot_size, decimals=self._config.volume_decimals
        )

        return pd.DataFrame(
            {
                "time": time_seconds,
                "event_type": event_type,
                "id": events["id"],
                "volume": size,
                "price": price_int,
                "direction": direction_int,
            }
        )

    def _price_units(self) -> int:
        """The number of ``1 / price_divisor`` units in one tick.

        A LOBSTER price is a whole number of these units, so a divisor that
        does not make one tick a whole number of them is refused.
        """
        units = self._config.tick_size * self._price_divisor
        if round(units) < 1 or abs(units - round(units)) > 1e-9 * units:
            raise ConfigError(
                f"LobsterWriter: a tick of {self._config.tick_size} is not a "
                f"whole number of 1/{self._price_divisor} units, so prices "
                "would be rounded.  Set price_divisor so that tick_size * "
                "price_divisor is a whole number (10 000 for LOBSTER's own "
                "files)."
            )
        return round(units)

    def _raw_prices(self, ticks: object) -> np.ndarray:
        """Encode integer-tick prices as LOBSTER's raw integer prices."""
        # One tick is a whole number of 1 / price_divisor units, so the
        # encoding is an exact integer multiply, with no rounding through the
        # quote currency.
        return np.asarray(ticks, dtype=np.int64) * self._price_units()

    def _orderbook_rows(
        self, events: pd.DataFrame, depth: pd.DataFrame, num_levels: int
    ) -> pd.DataFrame:
        """The orderbook file: the book after each event, from the depth table.

        The depth rows are replayed through a
        :class:`~ob_analytics.depth.PriceLevelBook`, the book the depth summary
        is built on, and the top *num_levels* price levels of each side are
        copied out after each event.  So each row's touch is the depth
        summary's touch after that event.  Only the LOBSTER encoding is done
        here: raw integer prices, sizes in the instrument's units, and the
        dummy prices LOBSTER writes for an empty level.
        """
        validate_columns(
            depth,
            {"event_id", "timestamp", "price", "volume", "direction"},
            "LobsterWriter",
        )
        _require_raw_units(depth, "depth", whole_float_volume=True)
        # Each depth row is applied with the event it names, events in the
        # order of the files.  The stable sort below keeps the depth table's
        # own order within one event, the order the depth summary replays.
        position = pd.Index(events["event_id"]).get_indexer(pd.Index(depth["event_id"]))
        if (position < 0).any():
            raise ConfigError(
                "LobsterWriter: the depth table names events that are not in "
                "the events table.  Pass the depth of the same run."
            )
        replay = np.argsort(position, kind="stable")
        position = position[replay]
        prices = depth["price"].to_numpy()[replay].astype(np.int64).tolist()
        volumes = depth["volume"].to_numpy()[replay].tolist()
        sides = (depth["direction"].to_numpy()[replay] != "bid").astype(int).tolist()
        # The depth rows of event i are rows[stops[i - 1]:stops[i]].
        stops = np.searchsorted(position, np.arange(len(events)), side="right")

        n = len(events)
        ask_ticks = np.zeros((n, num_levels), dtype=np.int64)
        bid_ticks = np.zeros((n, num_levels), dtype=np.int64)
        ask_lots = np.zeros((n, num_levels), dtype=np.int64)
        bid_lots = np.zeros((n, num_levels), dtype=np.int64)
        ask_depth = np.zeros(n, dtype=np.int64)
        bid_depth = np.zeros(n, dtype=np.int64)

        book = PriceLevelBook()
        row = 0
        for i in range(n):
            for j in range(row, int(stops[i])):
                book.update(prices[j], volumes[j], sides[j])
            row = int(stops[i])
            # Each side best first: asks from the lowest price, bids from the
            # highest.
            k = ask_depth[i] = min(num_levels, book.ask_prices.size)
            ask_ticks[i, :k] = book.ask_prices[:k]
            ask_lots[i, :k] = book.ask_volumes[:k]
            k = bid_depth[i] = min(num_levels, book.bid_prices.size)
            bid_ticks[i, :k] = book.bid_prices[::-1][:k]
            bid_lots[i, :k] = book.bid_volumes[::-1][:k]

        level = np.arange(num_levels)
        ask_empty = level >= ask_depth[:, None]
        bid_empty = level >= bid_depth[:, None]
        cfg = self._config
        lots = {"ask": ask_lots, "bid": bid_lots}
        if float(cfg.lot_size).is_integer():
            # A lot of whole units (one share, for LOBSTER's own data): write
            # whole numbers, as LOBSTER writes its share counts.
            sizes = {side: v * int(cfg.lot_size) for side, v in lots.items()}
        else:
            sizes = {
                side: lots_to_size(v, cfg.lot_size, decimals=cfg.volume_decimals)
                for side, v in lots.items()
            }
        raw = {
            "ask": np.where(ask_empty, _DUMMY_ASK_PRICE, self._raw_prices(ask_ticks)),
            "bid": np.where(bid_empty, _DUMMY_BID_PRICE, self._raw_prices(bid_ticks)),
        }

        columns: dict[str, np.ndarray] = {}
        for i in range(num_levels):
            for side in ("ask", "bid"):
                columns[f"{side}_price_{i + 1}"] = raw[side][:, i]
                columns[f"{side}_size_{i + 1}"] = sizes[side][:, i]
        return pd.DataFrame(columns)


# ── LOBSTER depth computation ─────────────────────────────────────────


def _side_level_changes(
    p: np.ndarray,
    v: np.ndarray,
    dummy: float,
    divisor: int,
    tick_size: float,
    lot_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Volume changes of one book side across consecutive orderbook rows.

    *p*/*v* are the side's ``(n_rows, n_levels)`` price/size arrays, as the
    file states them.  Each size is converted to a whole number of lots
    (:func:`~ob_analytics._utils.size_to_lots`), as the loader converts the
    message sizes.  A level is active when its price is not the *dummy*
    sentinel and its size is at least one lot.  Each unique raw price is
    converted to an integer tick count (``round((price / divisor) /
    tick_size)``) before keying, and duplicate tick prices within a row are
    summed in level order.

    Returns ``(row_indices, prices, volumes_after)`` for every (row, price)
    whose volume differs from the previous row (missing level = 0), the
    ``prices`` being integer tick counts (held as float here, cast to ``int64``
    by the caller), ascending within each row, and the volumes ``int64`` lots.
    """
    n = p.shape[0]
    lots = size_to_lots(v, lot_size)
    valid = (p != dummy) & (lots > 0)

    rp = np.full(p.shape, np.inf)
    if valid.any():
        uniq = np.unique(p[valid])
        rounded = price_to_ticks(uniq / divisor, tick_size).astype(np.float64)
        idx = np.clip(np.searchsorted(uniq, p), 0, uniq.size - 1)
        ok = valid & (uniq[idx] == p)
        rp[ok] = rounded[idx[ok]]
    vv = np.where(valid, lots, 0)

    # Sort each row by rounded price; inactive levels (+inf) sort last.
    order = np.argsort(rp, axis=1, kind="stable")
    rp = np.take_along_axis(rp, order, axis=1)
    vv = np.take_along_axis(vv, order, axis=1)
    counts = valid.sum(axis=1)

    # Collapse duplicate rounded prices within a row (summed, level order).
    dup_rows = np.unique(
        np.nonzero((rp[:, 1:] == rp[:, :-1]) & np.isfinite(rp[:, 1:]))[0]
    )
    collapsed: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for r in dup_rows.tolist():
        pr = rp[r, : counts[r]]
        first = np.ones(pr.size, dtype=bool)
        first[1:] = pr[1:] != pr[:-1]
        starts = np.nonzero(first)[0]
        collapsed[r] = (pr[first], np.add.reduceat(vv[r, : counts[r]], starts))

    out_rows: list[np.ndarray] = []
    out_prices: list[np.ndarray] = []
    out_vols: list[np.ndarray] = []
    prev_p: np.ndarray = np.empty(0)
    prev_v: np.ndarray = np.empty(0, dtype=np.int64)
    for i in range(n):
        if i in collapsed:
            cur_p, cur_v = collapsed[i]
        else:
            k = counts[i]
            cur_p, cur_v = rp[i, :k], vv[i, :k]

        union = np.union1d(prev_p, cur_p)
        va = np.zeros(union.size, dtype=np.int64)
        va[np.searchsorted(union, prev_p)] = prev_v
        vb = np.zeros(union.size, dtype=np.int64)
        vb[np.searchsorted(union, cur_p)] = cur_v
        changed = va != vb
        if changed.any():
            out_rows.append(np.full(int(changed.sum()), i, dtype=np.int64))
            out_prices.append(union[changed])
            out_vols.append(vb[changed])
        prev_p, prev_v = cur_p, cur_v

    if not out_rows:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )
    return (
        np.concatenate(out_rows),
        np.concatenate(out_prices),
        np.concatenate(out_vols),
    )


def lobster_depth_from_orderbook(
    events: pd.DataFrame,
    orderbook_path: Path,
    config: PipelineConfig,
    *,
    message_rows: int | None = None,
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
        The events the :class:`LobsterLoader` read from the message file.
        Each event takes its book from the orderbook row of its own message,
        named by ``original_number``; the time and ``event_id`` of the depth
        rows come from the event.  The orderbook rows of type 6 (cross trade)
        and type 7 (halt) messages, which have no event, are skipped.
    orderbook_path : Path
        Path to the LOBSTER orderbook CSV.
    config : PipelineConfig
        Pipeline configuration.  The file's prices are read on
        ``price_divisor`` and ``tick_size``, and its sizes are converted to
        whole lots of ``lot_size``, as the message sizes are.
    message_rows : int, optional
        The number of rows in the message file, every event type included.
        When given, the orderbook file must have exactly this many rows.
        :class:`LobsterSource` passes the count its loader read.

    Returns
    -------
    tuple of (depth DataFrame, depth_summary DataFrame)
        Sizes in both are ``int64`` lots.

    Raises
    ------
    ConfigError
        If *events* lacks a column this needs, an ``original_number`` is
        missing, not a whole number, below 1 or repeated, or the orderbook file does not have one row per
        message: fewer rows than the events name, or a count other than
        *message_rows*.
    """
    validate_columns(
        events,
        {"event_id", "timestamp", "raw_event_type", "original_number"},
        "lobster_depth_from_orderbook",
    )
    ob_raw = pd.read_csv(orderbook_path, header=None, dtype=np.float64).to_numpy()
    num_levels = ob_raw.shape[1] // 4

    # In message order, so consecutive book states are diffed in the order
    # they happened, whatever order the frame was passed in.
    book_events = (
        events[events["raw_event_type"].isin([1, 2, 3, 4, 5])]
        .sort_values("original_number", kind="stable")
        .reset_index(drop=True)
    )
    # The orderbook file has one row per message row, type 6 (cross trade)
    # and type 7 (halt) messages included, while the events keep only types
    # 1 to 5.  So each event reads the row of its own message, named by
    # ``original_number`` (the 1-based message row), not the row at its
    # position among the events.
    message_row = book_events["original_number"]
    if message_row.isna().any():
        raise ConfigError(
            "lobster_depth: original_number must be the 1-based message row "
            "of each event, as LobsterLoader sets it; some are missing."
        )
    if not pd.api.types.is_integer_dtype(message_row) and not np.all(
        message_row.to_numpy(dtype=np.float64) % 1 == 0
    ):
        raise ConfigError(
            "lobster_depth: original_number must be the 1-based message row "
            "of each event, as LobsterLoader sets it; found a fraction."
        )
    if not message_row.is_unique:
        raise ConfigError(
            "lobster_depth: two events name the same message row "
            "(original_number); pass the events of one message file."
        )
    rows = book_events["original_number"].to_numpy(dtype=np.int64) - 1
    if rows.size and rows.min() < 0:
        raise ConfigError(
            "lobster_depth: original_number must be the 1-based message row "
            "of each event, as LobsterLoader sets it; found a value below 1."
        )
    least = int(rows.max()) + 1 if rows.size else 0
    expected = message_rows if message_rows is not None else least
    if ob_raw.shape[0] < least or (
        message_rows is not None and ob_raw.shape[0] != message_rows
    ):
        raise ConfigError(
            f"lobster_depth: the orderbook file {orderbook_path.name} has "
            f"{ob_raw.shape[0]} rows, but the message file has "
            f"{'' if message_rows is not None else 'at least '}{expected}.  "
            "The two files must have one row per message."
        )
    ob_raw = ob_raw[rows]
    n = len(book_events)

    # ``.array`` keeps the tz-aware UTC dtype through the fancy indexing below
    # (``.values`` would drop the zone to a naive numpy datetime64).
    timestamps = book_events["timestamp"].array
    event_ids = book_events["event_id"].values

    # Diff consecutive book states per side.  LOBSTER columns repeat
    # (ask_price, ask_size, bid_price, bid_size) per level, so each row
    # reshapes to (levels, 4) and each side becomes an (n, levels) array
    # pair handled by :func:`_side_level_changes`.  Within an event the
    # changes are emitted asks first, then bids, each ascending in price
    # (the previous dict-diff emitted them in unspecified set order).
    grid = (config.price_divisor, config.tick_size, config.lot_size)

    lv = ob_raw.reshape(n, num_levels, 4)

    ask_idx, ask_p, ask_v = _side_level_changes(
        lv[:, :, 0], lv[:, :, 1], _DUMMY_ASK_PRICE, *grid
    )
    bid_idx, bid_p, bid_v = _side_level_changes(
        lv[:, :, 2], lv[:, :, 3], _DUMMY_BID_PRICE, *grid
    )

    row_idx = np.concatenate([ask_idx, bid_idx])
    prices = np.concatenate([ask_p, bid_p])
    vols = np.concatenate([ask_v, bid_v])
    side_code = np.concatenate(
        [np.zeros(ask_idx.size, dtype=np.int8), np.ones(bid_idx.size, dtype=np.int8)]
    )
    order = np.lexsort((side_code, row_idx))

    if row_idx.size:
        depth = pd.DataFrame(
            {
                "event_id": event_ids[row_idx[order]],
                "timestamp": timestamps[row_idx[order]],
                # Prices are integer tick counts (issue #155); _side_level_changes
                # holds them as float, so restore the int64 tick dtype here.
                "price": prices[order].astype(np.int64),
                "volume": vols[order],
                "direction": np.where(side_code[order] == 0, "ask", "bid"),
            }
        )
    else:
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


# ── LobsterSource descriptor ─────────────────────────────────────────


def _require_trading_date(td: object, where: str) -> str | pd.Timestamp:
    """Validate a LOBSTER ``trading_date`` (taken from :class:`RunContext`).

    Single source of truth for the loader, writer, and standalone
    writer-factory paths.  Returns *td* unchanged when valid; raises
    :class:`ValueError` when missing and :class:`TypeError` on a bad type.
    """
    if td is None:
        raise ValueError(
            f"LobsterSource.{where}: trading_date is required. "
            f"Pass it via ctx=RunContext(trading_date=...)."
        )
    if not isinstance(td, (str, pd.Timestamp)):
        raise TypeError(
            f"LobsterSource.{where}: trading_date must be str or "
            f"pandas.Timestamp, got {type(td).__name__}"
        )
    return td


@dataclass
class LobsterSource:
    """The LOBSTER source: offline replay of message/orderbook files (L3).

    ``trading_date`` is taken from the per-run
    :class:`~ob_analytics.protocols.RunContext`, not the source
    constructor — so the same ``LobsterSource()`` instance can be reused
    across runs with different sessions.  Offline only: LOBSTER ships as data
    files, so it satisfies :class:`~ob_analytics.protocols.OfflineSource` and
    has no live capability.
    """

    name: str = field(default="lobster", init=False, repr=False)
    # LOBSTER is a venue matched book (exchange matching engine): bids can
    # never rest above asks, so the reconstructed book is never crossed.
    feed_type: FeedType = field(default=FeedType.MATCHED_BOOK, init=False, repr=False)
    # Nasdaq ITCH names only the resting order of an execution.  The taker
    # columns are filled by a guess (LobsterTradeReader._find_takers), which
    # the order types use, but the feed itself shows the maker only.
    trade_attribution: TradeAttribution = field(
        default=TradeAttribution.MAKER_ONLY, init=False, repr=False
    )
    # A LOBSTER message file holds the exchange's time only; the loader copies
    # it into ``timestamp``, so there is no receive time to check it against.
    clocks: Clocks = field(default=Clocks.VENUE_ONLY, init=False, repr=False)
    # Per-order (market-by-order) feed — the full reconstruction model.
    level: Level = field(default=Level.L3, init=False, repr=False)
    # LOBSTER needs no per-source knobs; empty typed settings keep the
    # construction signature uniform with sources that do.
    settings: SourceSettings = field(default_factory=SourceSettings)

    _loader: LobsterLoader | None = field(default=None, repr=False, init=False)

    def create_loader(self, config: PipelineConfig, ctx: RunContext) -> EventLoader:
        td = _require_trading_date(ctx.trading_date, "create_loader")
        self._loader = LobsterLoader(
            config,
            trading_date=td,
            session_tz=ctx.session_tz or LOBSTER_DEFAULT_TZ,
            venue=ctx.venue,
            symbol=ctx.symbol,
        )
        return self._loader

    def create_trade_source(
        self, config: PipelineConfig, ctx: RunContext
    ) -> TradeSource:
        return LobsterTradeReader(config)

    def create_writer(
        self, config: PipelineConfig | None, ctx: RunContext
    ) -> DataWriter:
        return _make_lobster_writer(config, ctx)

    def compute_depth(
        self,
        events: pd.DataFrame,
        config: Any,
        source: Any,
        ctx: RunContext,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        ob_path = self._loader.orderbook_path if self._loader is not None else None
        if ob_path is None:
            ob_path = LobsterLoader._resolve_orderbook_file(Path(source))
        if ob_path is None:
            logger.warning(
                "LobsterSource: no orderbook file found; "
                "falling back to event-based depth"
            )
            return None
        message_rows = self._loader.message_rows if self._loader is not None else None
        return lobster_depth_from_orderbook(
            events, ob_path, config, message_rows=message_rows
        )

    def config_defaults(self) -> dict[str, Any]:
        return dict(_LOBSTER_CONFIG_DEFAULTS)

    def required_context(self) -> list[str]:
        # LOBSTER message/orderbook filenames carry no date, so the trading
        # date must be supplied explicitly.
        return ["trading_date"]


# ── Register this source ──────────────────────────────────────────────
# Registration runs at the bottom, after ``LobsterSource`` is defined.
from ob_analytics.sources import register_source


def _make_lobster_writer(config, ctx):
    td = _require_trading_date(ctx.trading_date, "create_writer")
    return LobsterWriter(
        config, trading_date=td, session_tz=ctx.session_tz or LOBSTER_DEFAULT_TZ
    )


register_source("lobster", LobsterSource)
