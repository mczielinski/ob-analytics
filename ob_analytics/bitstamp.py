"""Bitstamp data format: event loading, matching, trade inference, and writing.

Contains the full symmetric set of Bitstamp-specific components:

* :class:`BitstampLoader` — loads Bitstamp CSV event data
* :class:`BitstampMatcher` — matches bid/ask fills (Needleman-Wunsch)
* :class:`BitstampTradeInferrer` — infers trades from matched events
* :class:`BitstampWriter` — writes events back to Bitstamp CSV
* :class:`BitstampFormat` — format descriptor bundling all of the above

Format-agnostic analytics (e.g. :func:`~ob_analytics.analytics.order_aggressiveness`)
live in :mod:`ob_analytics.analytics`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from loguru import logger

from ob_analytics._utils import (
    datetime_to_epoch,
    epoch_to_datetime,
    validate_columns,
    validate_non_empty,
)
from ob_analytics.config import PipelineConfig
from ob_analytics.exceptions import MatchingError
from ob_analytics.protocols import (
    DataWriter,
    EventLoader,
    Format,
    MatchingEngine,
    TradeInferrer,
)


# ---------------------------------------------------------------------------
# Needleman-Wunsch sequence alignment (bid/ask fill matching)
# ---------------------------------------------------------------------------


def _create_similarity_matrix(
    a: pd.Series, b: pd.Series, cut_off_ms: int
) -> np.ndarray:
    """Create a similarity matrix based on time differences and cut-off window.

    Parameters
    ----------
    a : pandas.Series
        A pandas Series of timestamps.
    b : pandas.Series
        A pandas Series of timestamps.
    cut_off_ms : int
        The cut-off time window in milliseconds.

    Returns
    -------
    numpy.ndarray
        A NumPy array representing the similarity matrix.
    """
    a_ns = a.to_numpy().astype("datetime64[ns]").astype(np.float64)
    b_ns = b.to_numpy().astype("datetime64[ns]").astype(np.float64)
    diff_ms = np.abs(a_ns.reshape(-1, 1) - b_ns) / 1e6
    safe_diff = np.where(diff_ms != 0, diff_ms, 1.0)
    return np.where(diff_ms != 0, cut_off_ms / safe_diff, float(cut_off_ms))


def _align_sequences(s_matrix: np.ndarray, gap_penalty: int = -1) -> np.ndarray:
    """Perform Needleman-Wunsch alignment and return aligned indices.

    Parameters
    ----------
    s_matrix : numpy.ndarray
        The similarity matrix.
    gap_penalty : int, optional
        The penalty for gaps in the alignment. Default is -1.

    Returns
    -------
    numpy.ndarray
        A NumPy array with aligned indices from the two sequences.
    """
    m, n = s_matrix.shape
    f_matrix = np.zeros((m + 1, n + 1))
    f_matrix[0, :] = np.arange(n + 1) * gap_penalty
    f_matrix[:, 0] = np.arange(m + 1) * gap_penalty

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = f_matrix[i - 1, j - 1] + s_matrix[i - 1, j - 1]
            delete = f_matrix[i - 1, j] + gap_penalty
            insert = f_matrix[i, j - 1] + gap_penalty
            f_matrix[i, j] = max(match, delete, insert)

    aligned_indices = []
    i, j = m, n
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and f_matrix[i, j] == f_matrix[i - 1, j - 1] + s_matrix[i - 1, j - 1]
        ):
            aligned_indices.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and f_matrix[i, j] == f_matrix[i - 1, j] + gap_penalty:
            i -= 1
        else:
            j -= 1

    return np.array(aligned_indices[::-1])


# ---------------------------------------------------------------------------
# NeedlemanWunschMatcher
# ---------------------------------------------------------------------------


class NeedlemanWunschMatcher:
    """Match bid/ask fills using volume equality and Needleman-Wunsch alignment.

    Satisfies the :class:`~ob_analytics.protocols.MatchingEngine` protocol.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration.  ``match_cutoff_ms`` controls the maximum
        time window between fills to consider a match.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    def match(self, events: pd.DataFrame) -> pd.DataFrame:
        """Add a ``matching_event`` column pairing bid/ask fills.

        For each unique fill volume, bid and ask fills within
        ``match_cutoff_ms`` are paired.  When a simple closest-match is
        ambiguous (duplicates or gaps), Needleman-Wunsch sequence
        alignment is used as a fallback.

        Parameters
        ----------
        events : pandas.DataFrame
            Events with columns ``direction``, ``fill``,
            ``original_number``, ``event_id``, ``timestamp``.

        Returns
        -------
        pandas.DataFrame
            Same events with a ``matching_event`` column added.
            Unmatched rows have ``NaN``.
        """
        validate_columns(
            events,
            {"direction", "fill", "original_number", "event_id", "timestamp"},
            "NeedlemanWunschMatcher.match",
        )
        validate_non_empty(events, "NeedlemanWunschMatcher.match")

        matched = self._run_matching(events)

        events["matching_event"] = np.nan
        if len(matched) == 0:
            return events
        matched_bids = pd.DataFrame(matched, columns=["event_id", "matching_event"])
        matched_asks = pd.DataFrame(matched, columns=["matching_event", "event_id"])
        events = pd.merge(events, matched_bids, on="event_id", how="left")
        events = pd.merge(events, matched_asks, on="event_id", how="left")

        events["matching_event"] = (
            events["matching_event_y"]
            .fillna(events["matching_event_x"])
            .fillna(events["matching_event"])
        )

        events = events.drop(columns=["matching_event_x", "matching_event_y"])
        return events

    def _run_matching(self, events: pd.DataFrame) -> np.ndarray:
        cut_off_td = np.timedelta64(self._config.match_cutoff_ms * 1_000_000, "ns")
        res: list[np.ndarray] = []
        cols = ["original_number", "event_id", "fill", "timestamp"]

        bid_fills = events[(events["direction"] == "bid") & (events["fill"] != 0)][cols]
        ask_fills = events[(events["direction"] == "ask") & (events["fill"] != 0)][cols]

        def fill_id(src: pd.DataFrame, dst: pd.DataFrame) -> pd.DataFrame:
            id_fills = src[src["fill"].isin(dst["fill"])]
            return id_fills.sort_values(
                by=["fill", "timestamp"], ascending=[False, True], kind="stable"
            )

        id_bid_fills = fill_id(bid_fills, ask_fills)
        id_ask_fills = fill_id(ask_fills, bid_fills)

        for volume in id_bid_fills["fill"].unique():
            bids = id_bid_fills[id_bid_fills["fill"] == volume]
            asks = id_ask_fills[id_ask_fills["fill"] == volume]

            time_deltas = (
                bids["timestamp"].to_numpy(dtype="datetime64[ns]").reshape(-1, 1)
                - asks["timestamp"].to_numpy(dtype="datetime64[ns]")
            ).astype("timedelta64[ns]")

            if len(asks) == 1:
                time_deltas = time_deltas.reshape(-1, 1)

            closest_ask_indices = np.argmin(np.abs(time_deltas), axis=1)
            ask_event_ids = asks["event_id"].values[closest_ask_indices]

            mask = (
                np.abs(time_deltas[np.arange(len(bids)), closest_ask_indices])
                <= cut_off_td
            )
            ask_event_ids = np.where(mask, ask_event_ids, np.nan)

            if not any(pd.isna(ask_event_ids)) and len(ask_event_ids) == len(
                set(ask_event_ids)
            ):
                matches = np.column_stack((bids["event_id"], ask_event_ids))
                res.extend(matches)
            else:
                similarity_matrix = _create_similarity_matrix(
                    bids["timestamp"], asks["timestamp"], self._config.match_cutoff_ms
                )
                aligned_indices = _align_sequences(similarity_matrix)
                bid_ts = bids["timestamp"].to_numpy(dtype="datetime64[ns]")
                ask_ts = asks["timestamp"].to_numpy(dtype="datetime64[ns]")
                matched_indices = aligned_indices[
                    np.abs(
                        bid_ts[aligned_indices[:, 0]] - ask_ts[aligned_indices[:, 1]]
                    )
                    <= cut_off_td
                ]
                res.extend(
                    np.column_stack(
                        (
                            bids["event_id"].values[matched_indices[:, 0]],
                            asks["event_id"].values[matched_indices[:, 1]],
                        )
                    )
                )

        return np.array(res, dtype=int)


# ---------------------------------------------------------------------------
# BitstampLoader
# ---------------------------------------------------------------------------


class BitstampLoader:
    """Load raw limit-order events from a Bitstamp-format CSV.

    Satisfies the :class:`~ob_analytics.protocols.EventLoader` protocol.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration.  ``price_decimals`` and ``volume_decimals``
        control rounding precision.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    def load(self, source: str | Path) -> pd.DataFrame:
        """Read *source* CSV and return a cleaned events DataFrame.

        Parameters
        ----------
        source : str or Path
            Path to a Bitstamp-format CSV with columns ``id``,
            ``timestamp``, ``exchange_timestamp``,
            ``price``, ``volume``, ``action``, ``direction``.

        Returns
        -------
        pandas.DataFrame
            Cleaned events with columns ``id``, ``timestamp``,
            ``exchange_timestamp``, ``price``, ``volume``, ``action``,
            ``direction``, ``event_id``, ``fill``, ``original_number``.
        """
        price_digits = self._config.price_decimals
        volume_digits = self._config.volume_decimals

        events = pd.read_csv(source)
        validate_columns(
            events,
            {
                "id",
                "timestamp",
                "exchange_timestamp",
                "price",
                "volume",
                "action",
                "direction",
            },
            "BitstampLoader.load",
        )
        validate_non_empty(events, "BitstampLoader.load")

        events = events[events["volume"] >= 0]
        events = events.reset_index().rename(columns={"index": "original_number"})
        events["original_number"] = events["original_number"] + 1
        events["volume"] = events["volume"].round(volume_digits)
        events["price"] = events["price"].round(price_digits)

        ts_unit = self._config.timestamp_unit
        events["timestamp"] = epoch_to_datetime(events["timestamp"], ts_unit)
        events["exchange_timestamp"] = epoch_to_datetime(
            events["exchange_timestamp"], ts_unit
        )
        events["action"] = pd.Categorical(
            events["action"], categories=["created", "changed", "deleted"], ordered=True
        )
        events["direction"] = pd.Categorical(
            events["direction"], categories=["bid", "ask"], ordered=True
        )

        events = events.sort_values(
            by=["id", "volume", "action", "timestamp"],
            ascending=[True, False, True, True],
            kind="stable",
        )

        events["event_id"] = np.arange(1, len(events) + 1)
        events = self._remove_duplicates(events)

        fill_deltas = events.groupby("id")["volume"].diff().fillna(0)
        price_deltas = events.groupby("id")["price"].diff().fillna(0)
        fill_deltas = fill_deltas.where(price_deltas == 0, 0)
        events["fill"] = fill_deltas.abs().round(volume_digits)

        ts_sorted: pd.Series = events.groupby("id")["timestamp"].transform(
            lambda x: np.sort(np.asarray(x.values), kind="stable")
        )
        events["timestamp"] = ts_sorted

        events["raw_event_type"] = pd.NA

        return events

    @staticmethod
    def _remove_duplicates(events: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate delete events (port of R ``removeDuplicates``)."""
        deletes = events[events["action"] == "deleted"].sort_values(
            by=["id", "volume"], kind="stable"
        )
        dup_ids = deletes.loc[deletes["id"].duplicated(), "id"]
        duplicate_deletes = deletes[deletes["id"].isin(dup_ids)]
        duplicate_event_ids = duplicate_deletes.loc[
            duplicate_deletes["id"].duplicated(), "event_id"
        ]

        rem_dup = len(duplicate_event_ids)
        if rem_dup > 0:
            removed_ids = events.loc[events["event_id"].isin(duplicate_event_ids), "id"]
            logger.warning(
                "Removed {} duplicate order cancellations: {}",
                rem_dup,
                " ".join(removed_ids.astype(str)),
            )

        return events[~events["event_id"].isin(duplicate_event_ids)]


# ── BitstampMatcher ───────────────────────────────────────────────────


class BitstampMatcher:
    """Matching engine for Bitstamp event data.

    Thin wrapper around :class:`NeedlemanWunschMatcher` providing a stable,
    exchange-named API entry point.  Symmetric with
    :class:`~ob_analytics.lobster.LobsterMatcher`.

    Satisfies the :class:`~ob_analytics.protocols.MatchingEngine` protocol.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._matcher = NeedlemanWunschMatcher(config)

    def match(self, events: pd.DataFrame) -> pd.DataFrame:
        """Pair bid/ask fills; return events with ``matching_event`` column."""
        return self._matcher.match(events)


# ── BitstampTradeInferrer ─────────────────────────────────────────────


class BitstampTradeInferrer:
    """Construct trade records from matched bid/ask event pairs.

    Satisfies the :class:`~ob_analytics.protocols.TradeInferrer` protocol.
    Symmetric with :class:`~ob_analytics.lobster.LobsterTradeInferrer`.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration.  ``price_jump_threshold`` controls when the
        maker/taker swap heuristic is triggered.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    def infer_trades(self, events: pd.DataFrame) -> pd.DataFrame:
        """Build a trades DataFrame from matched events.

        Determines maker vs taker using exchange timestamps, and applies
        a price-jump heuristic to correct misattributions.

        Parameters
        ----------
        events : pandas.DataFrame
            Events with ``matching_event`` column populated.

        Returns
        -------
        pandas.DataFrame
            Trades with columns ``timestamp``, ``price``, ``volume``,
            ``direction``, ``maker_event_id``, ``taker_event_id``,
            ``maker``, ``taker``.
        """
        validate_columns(
            events,
            {
                "direction",
                "matching_event",
                "event_id",
                "exchange_timestamp",
                "timestamp",
                "price",
                "fill",
                "id",
                "original_number",
            },
            "BitstampTradeInferrer.infer_trades",
        )
        validate_non_empty(events, "BitstampTradeInferrer.infer_trades")

        matching_bids = events[
            (events["direction"] == "bid") & ~pd.isna(events["matching_event"])
        ].sort_values(by="event_id", kind="stable")
        matching_asks = events[
            (events["direction"] == "ask") & ~pd.isna(events["matching_event"])
        ].sort_values(by="matching_event", kind="stable")

        if not np.asarray(
            matching_bids["event_id"].values == matching_asks["matching_event"].values
        ).all():
            raise MatchingError(
                "Bid event IDs do not align with ask matching events. "
                "This indicates a matching error in the upstream matching step."
            )

        matching_bids = matching_bids.reset_index(drop=True)
        matching_asks = matching_asks.reset_index(drop=True)
        bid_exchange_ts = matching_bids["exchange_timestamp"]
        ask_exchange_ts = matching_asks["exchange_timestamp"]
        bid_maker = (bid_exchange_ts < ask_exchange_ts) | (
            (bid_exchange_ts == ask_exchange_ts)
            & (matching_bids["event_id"] < matching_asks["event_id"])
        )

        timestamp = np.where(
            matching_bids["timestamp"] <= matching_asks["timestamp"],
            matching_bids["timestamp"],
            matching_asks["timestamp"],
        )

        price = np.where(bid_maker, matching_bids["price"], matching_asks["price"])
        volume = matching_bids["fill"]

        direction = pd.Categorical(
            np.where(bid_maker, "sell", "buy"), categories=["buy", "sell"], ordered=True
        ).astype(str)

        maker_event_id = np.where(
            bid_maker, matching_bids["event_id"], matching_asks["event_id"]
        )
        taker_event_id = np.where(
            bid_maker, matching_asks["event_id"], matching_bids["event_id"]
        )

        id_to_id = dict(zip(events["event_id"], events["id"]))
        id_to_original_number = dict(zip(events["event_id"], events["original_number"]))

        maker = pd.Series(maker_event_id).map(id_to_id).values
        taker = pd.Series(taker_event_id).map(id_to_id).values
        maker_og = pd.Series(maker_event_id).map(id_to_original_number).values
        taker_og = pd.Series(taker_event_id).map(id_to_original_number).values

        combined = pd.DataFrame(
            {
                "timestamp": timestamp,
                "price": price,
                "volume": volume,
                "direction": direction,
                "maker_event_id": maker_event_id,
                "taker_event_id": taker_event_id,
                "maker": maker,
                "taker": taker,
                "maker_og": maker_og,
                "taker_og": taker_og,
            }
        )
        combined = combined.sort_values(by="timestamp", kind="stable")

        self._fix_price_jumps(combined, events)
        return combined

    def _fix_price_jumps(self, combined: pd.DataFrame, events: pd.DataFrame) -> None:
        """Swap maker/taker when consecutive prices jump beyond threshold."""
        threshold = self._config.price_jump_threshold
        jumps = np.where(abs(np.diff(combined["price"])) > threshold)[0]
        if len(jumps) == 0:
            return

        if jumps[0] == 0:
            jumps = jumps[1:]

        logger.warning(
            "{}: {} jumps > ${:.0f} (swapping makers with takers)",
            combined["timestamp"].iloc[0].strftime("%Y-%m-%d"),
            len(jumps),
            threshold,
        )

        for i in jumps:
            prev_jump, this_jump = combined.iloc[i - 1], combined.iloc[i]
            if abs(this_jump["price"] - prev_jump["price"]) > threshold:
                taker_eid = this_jump["taker_event_id"]
                taker_price = events.loc[events["event_id"] == taker_eid, "price"].iloc[
                    0
                ]
                taker_dir = "sell" if this_jump["direction"] == "buy" else "buy"
                swap = pd.DataFrame(
                    {
                        "price": taker_price,
                        "direction": taker_dir,
                        "maker_event_id": taker_eid,
                        "taker_event_id": this_jump["maker_event_id"],
                        "maker": this_jump["taker"],
                        "taker": this_jump["maker"],
                        "maker_og": this_jump["taker_og"],
                        "taker_og": this_jump["maker_og"],
                    },
                    index=[i],
                )
                combined.loc[
                    i,
                    [
                        "price",
                        "direction",
                        "maker_event_id",
                        "taker_event_id",
                        "maker",
                        "taker",
                        "maker_og",
                        "taker_og",
                    ],
                ] = swap.iloc[0]


# ── BitstampWriter ────────────────────────────────────────────────────


class BitstampWriter:
    """Write pipeline events back to Bitstamp-format CSV.

    Satisfies the :class:`~ob_analytics.protocols.DataWriter` protocol.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    def write(
        self,
        data: dict[str, pd.DataFrame],
        dest: str | Path,
        **kwargs: Any,
    ) -> Path:
        """Write the events DataFrame to a Bitstamp-format CSV.

        Parameters
        ----------
        data : dict of str to DataFrame
            Must contain an ``"events"`` key.
        dest : str or Path
            Output CSV file path.

        Returns
        -------
        Path
            The written file path.
        """
        events = data["events"]
        p = Path(dest)
        ts_unit = self._config.timestamp_unit
        out = pd.DataFrame(
            {
                "id": events["id"],
                "timestamp": datetime_to_epoch(events["timestamp"], ts_unit),
                "exchange_timestamp": datetime_to_epoch(
                    events["exchange_timestamp"], ts_unit
                ),
                "price": events["price"],
                "volume": events["volume"],
                "action": events["action"].astype(str),
                "direction": events["direction"].astype(str),
            }
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(p, index=False)
        return p


# ── BitstampFormat descriptor ─────────────────────────────────────────


@dataclass
class BitstampFormat(Format):
    """Format descriptor for Bitstamp-style CSV data."""

    name: str = "bitstamp"

    def create_loader(self, config: PipelineConfig) -> EventLoader:
        return BitstampLoader(config)

    def create_matcher(self, config: PipelineConfig) -> MatchingEngine:
        return BitstampMatcher(config)

    def create_trade_inferrer(self, config: PipelineConfig) -> TradeInferrer:
        return BitstampTradeInferrer(config)

    def create_writer(self, config: PipelineConfig) -> DataWriter:
        return BitstampWriter(config)

    def config_defaults(self) -> dict[str, Any]:
        return {
            "price_decimals": 2,
            "volume_decimals": 8,
            "timestamp_unit": "ms",
            "match_cutoff_ms": 5000,
            "price_jump_threshold": 10.0,
        }
