"""Export a run to the backtesting engines next door (issue #113).

ob-analytics reconstructs, classifies, measures and draws; Nautilus Trader and
hftbacktest backtest.  A user who has finished the analysis moves to one of
those, and this module writes what each one reads, so the move needs no glue
code of their own.

Neither engine is a dependency.  Nautilus ingests a pandas frame through its
``OrderBookDeltaDataWrangler`` and hftbacktest a structured numpy array, so both
targets are shapes this library can build on its own; installing the engine is
the user's business, and only the cross-check in #224 needs one present.

Both engines take **float** prices, while the canonical schema stores integer
ticks (issue #155), so both writers scale by the run's ``tick_size``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ob_analytics.config import PipelineConfig
from ob_analytics.data import register_writer
from ob_analytics.schemas import time_order_keys

# ── hftbacktest ───────────────────────────────────────────────────────

#: hftbacktest's feed-event record (``hftbacktest.types.event_dtype``).  Written
#: out rather than imported, because the engine is not a dependency.
HFT_EVENT_DTYPE = np.dtype(
    {
        "names": ["ev", "exch_ts", "local_ts", "px", "qty", "order_id", "ival", "fval"],
        "formats": ["<u8", "<i8", "<i8", "<f8", "<f8", "<u8", "<i8", "<f8"],
        "offsets": [0, 8, 16, 24, 32, 40, 48, 56],
        "itemsize": 64,
        "aligned": True,
    }
)

#: The npz member hftbacktest loads: ``np.load(path)["data"]``.
HFT_ARRAY_KEY = "data"

# Event types, and the flag bits packed into ``ev`` alongside them.
_HFT_ADD_ORDER = 10
_HFT_CANCEL_ORDER = 11
_HFT_MODIFY_ORDER = 12
_HFT_EXCH = 1 << 31
_HFT_LOCAL = 1 << 30
_HFT_BUY = 1 << 29
_HFT_SELL = 1 << 28

#: Canonical action to hftbacktest event type.  The engine models a *change* as
#: a modify rather than a cancel-and-replace, which matches the canonical
#: ``changed`` action carrying the order's outstanding size after the event.
_ACTION_TO_HFT_EVENT: dict[str, int] = {
    "created": _HFT_ADD_ORDER,
    "changed": _HFT_MODIFY_ORDER,
    "deleted": _HFT_CANCEL_ORDER,
}

_DIRECTION_TO_HFT_FLAG: dict[str, int] = {"bid": _HFT_BUY, "ask": _HFT_SELL}


def _epoch_nanos(column: pd.Series) -> np.ndarray:
    """Return *column* as int64 nanoseconds since the epoch.

    ``as_unit("ns")`` first, rather than a bare ``astype("int64")``: pandas
    carries a resolution on a datetime column and infers microseconds for some
    inputs, so casting straight to ``int64`` would silently export a timestamp a
    thousand times too small.  The canonical schema is nanoseconds (issue #154)
    and both engines read nanoseconds.
    """
    return column.dt.as_unit("ns").astype("int64").to_numpy()


def _in_canonical_time_order(events: pd.DataFrame) -> pd.DataFrame:
    """Return *events* in the canonical total order (issue #154).

    Both engines replay a timeline and reject an input that steps backwards —
    hftbacktest's ``validate_event_order`` checks exactly this — while the
    canonical frames are not always stored that way: the Bitstamp path is
    ordered by order id.  Sorting here rather than asking the user to sort keeps
    the export correct by default.
    """
    return events.sort_values(time_order_keys(events), kind="stable")


def _hftbacktest_rows(events: pd.DataFrame, *, tick_size: float) -> np.ndarray:
    """Return one unflagged feed record per canonical event, in canonical order."""
    out: np.ndarray = np.zeros(len(events), dtype=HFT_EVENT_DTYPE)
    if len(events) == 0:
        return out

    events = _in_canonical_time_order(events)
    event_type = events["action"].astype(str).map(_ACTION_TO_HFT_EVENT)
    side_flag = events["direction"].astype(str).map(_DIRECTION_TO_HFT_FLAG)

    out["ev"] = event_type.to_numpy(dtype=np.uint64) | side_flag.to_numpy(
        dtype=np.uint64
    )
    out["exch_ts"] = _epoch_nanos(events["exchange_timestamp"])
    out["local_ts"] = _epoch_nanos(events["timestamp"])
    out["px"] = events["price"].to_numpy(dtype=np.float64) * tick_size
    out["qty"] = events["volume"].to_numpy(dtype=np.float64)
    out["order_id"] = events["id"].to_numpy(dtype=np.uint64)
    return out


def to_hftbacktest_array(
    events: pd.DataFrame,
    *,
    tick_size: float,
) -> np.ndarray:
    """Return *events* as an hftbacktest feed array, on two timelines.

    Parameters
    ----------
    events : pandas.DataFrame
        Canonical events.
    tick_size : float
        The run's tick size, used to scale the integer ``price`` back to the
        quote currency, which is what the engine reads.

    Returns
    -------
    numpy.ndarray
        A structured array of :data:`HFT_EVENT_DTYPE`, ordered by the time each
        record enters its own timeline.  ``ival`` and ``fval`` are left at zero:
        they carry per-venue extras the canonical schema does not keep.

    Notes
    -----
    hftbacktest replays **two** clocks: an event reaches the exchange at
    ``exch_ts`` and the strategy at ``local_ts``, which is how it models feed
    latency.  ``ev`` carries a flag for each timeline, and its
    ``validate_event_order`` requires the records flagged for a timeline to be
    non-decreasing in that timeline's clock.

    The two orders are not the same order.  On the bundled Bitstamp sample the
    exchange stamp leads the receive stamp by 0.08 to 2.7 seconds and the two
    disagree on the order of 14,966 of 314,057 events, so no single sequence of
    one record per event can be sorted for both.  An event whose clocks disagree
    is therefore written **twice**, once per timeline, which is the same shape
    the engine's own ``correct_event_order`` produces.  An event whose clocks
    agree stays one record carrying both flags.
    """
    base = _hftbacktest_rows(events, tick_size=tick_size)
    if len(base) == 0:
        return base

    agree = base["exch_ts"] == base["local_ts"]

    both = base[agree].copy()
    both["ev"] |= np.uint64(_HFT_EXCH | _HFT_LOCAL)

    split = base[~agree]
    on_exchange = split.copy()
    on_exchange["ev"] |= np.uint64(_HFT_EXCH)
    on_exchange = on_exchange[np.argsort(on_exchange["exch_ts"], kind="stable")]

    locally = split.copy()
    locally["ev"] |= np.uint64(_HFT_LOCAL)
    locally = locally[np.argsort(locally["local_ts"], kind="stable")]

    # Each record is placed by the clock of the timeline it belongs to, so a
    # stable sort on that key merges three already-sorted streams without
    # disturbing the order within any of them.
    merged = np.concatenate([both, on_exchange, locally])
    entry = np.concatenate(
        [both["exch_ts"], on_exchange["exch_ts"], locally["local_ts"]]
    )
    return merged[np.argsort(entry, kind="stable")]


class HftbacktestWriter:
    """Write a run's events as an hftbacktest ``.npz`` feed file.

    Satisfies the :class:`~ob_analytics.protocols.DataWriter` protocol; reached
    as ``save_data(..., fmt="hftbacktest")``.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    def write(
        self,
        data: dict[str, pd.DataFrame],
        dest: str | Path,
        **kwargs: Any,
    ) -> Path:
        """Write ``data["events"]`` to *dest* as a compressed npz.

        *dest* is a file.  The array lands under the ``"data"`` member, which is
        the one ``hftbacktest`` loads.
        """
        p = Path(dest)
        p.parent.mkdir(parents=True, exist_ok=True)
        array = to_hftbacktest_array(data["events"], tick_size=self._config.tick_size)
        np.savez_compressed(p, **{HFT_ARRAY_KEY: array})  # type: ignore
        return p


# ── Nautilus Trader ───────────────────────────────────────────────────

#: The columns ``OrderBookDeltaDataWrangler.process`` reads, in its own order.
NAUTILUS_DELTA_COLUMNS: tuple[str, ...] = (
    "action",
    "side",
    "price",
    "size",
    "order_id",
    "flags",
    "sequence",
)

#: Canonical action to Nautilus ``BookAction``.  ``CLEAR`` has no canonical
#: counterpart: the schema records what each order did, and never "the book was
#: emptied", so nothing maps to it.
_ACTION_TO_NAUTILUS: dict[str, str] = {
    "created": "ADD",
    "changed": "UPDATE",
    "deleted": "DELETE",
}

_DIRECTION_TO_NAUTILUS: dict[str, str] = {"bid": "BUY", "ask": "SELL"}


def _resting_size_on_delete(events: pd.DataFrame) -> pd.DataFrame:
    """Give every delta a size Nautilus will accept, dropping those that cannot.

    The canonical ``volume`` on a ``deleted`` event is the size *removed*, so an
    order that was fully filled leaves a delete carrying zero: there was nothing
    left to cancel.  Nautilus rejects a delta whose size is not positive, and
    dropping the delete instead would leave the order resting in its book for
    the rest of the session.  So a zero-size row is given the size that order
    last rested at, which is what the engine still believes it holds.

    A row with no earlier size to borrow — an order that never rested — is
    dropped: there is nothing truthful to tell the engine about it.
    """
    volume = events["volume"]
    if (volume > 0).all():
        return events

    # Forward-fill within each order, so a zero borrows that order's own last
    # resting size and never a neighbouring order's.
    last_resting = volume.where(volume > 0).groupby(events["id"].to_numpy()).ffill()
    filled = volume.mask(volume <= 0, last_resting)
    return events.loc[filled > 0].assign(volume=filled[filled > 0])


def to_nautilus_deltas(
    events: pd.DataFrame,
    *,
    tick_size: float,
) -> pd.DataFrame:
    """Return *events* in the frame shape Nautilus' delta wrangler takes.

    Parameters
    ----------
    events : pandas.DataFrame
        Canonical events.
    tick_size : float
        The run's tick size, used to scale the integer ``price`` back to the
        quote currency.

    Returns
    -------
    pandas.DataFrame
        The seven :data:`NAUTILUS_DELTA_COLUMNS`, indexed by the exchange clock
        as a UTC ``DatetimeIndex`` — the wrangler reads the index as the event
        time and passes it through ``as_utc_index``.

    Notes
    -----
    ``flags`` is zero throughout: it carries Nautilus' record flags, of which
    ``F_SNAPSHOT`` is the one that matters, and a canonical event stream is a
    stream of deltas rather than a snapshot.  ``sequence`` uses the venue's own
    sequence when the source published one (issue #146) and falls back to the
    dense ``event_id``, so the column is always a usable order.
    """
    events = _resting_size_on_delete(_in_canonical_time_order(events))
    sequence = (
        events["sequence"]
        if "sequence" in events.columns and events["sequence"].notna().all()
        else events["event_id"]
    )
    # ``.to_numpy()`` on every column, not just the numeric ones: the frame is
    # built against a new DatetimeIndex, and a Series still carrying the events'
    # own index would be realigned against it and come back all NaN.
    frame = pd.DataFrame(
        {
            "action": events["action"].astype(str).map(_ACTION_TO_NAUTILUS).to_numpy(),
            "side": (
                events["direction"].astype(str).map(_DIRECTION_TO_NAUTILUS).to_numpy()
            ),
            "price": events["price"].to_numpy(dtype=np.float64) * tick_size,
            "size": events["volume"].to_numpy(dtype=np.float64),
            "order_id": events["id"].to_numpy(dtype=np.uint64),
            "flags": np.zeros(len(events), dtype=np.uint8),
            "sequence": sequence.to_numpy(dtype=np.uint64),
        },
        index=pd.DatetimeIndex(
            events["exchange_timestamp"].dt.as_unit("ns"), name="timestamp"
        ),
    )
    return frame[list(NAUTILUS_DELTA_COLUMNS)]


class NautilusWriter:
    """Write a run's events as a Nautilus order-book delta file.

    Satisfies the :class:`~ob_analytics.protocols.DataWriter` protocol; reached
    as ``save_data(..., fmt="nautilus")``.  The output is Parquet, so
    ``pd.read_parquet`` hands ``OrderBookDeltaDataWrangler.process`` the frame it
    wants, index and dtypes intact.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    def write(
        self,
        data: dict[str, pd.DataFrame],
        dest: str | Path,
        **kwargs: Any,
    ) -> Path:
        """Write ``data["events"]`` to *dest* as one Parquet file of deltas."""
        p = Path(dest)
        p.parent.mkdir(parents=True, exist_ok=True)
        deltas = to_nautilus_deltas(data["events"], tick_size=self._config.tick_size)
        deltas.to_parquet(p, index=True)
        return p


register_writer("hftbacktest", lambda config, ctx: HftbacktestWriter(config))
register_writer("nautilus", lambda config, ctx: NautilusWriter(config))
