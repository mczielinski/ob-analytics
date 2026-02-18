"""Event matching engine.

Contains :class:`NeedlemanWunschMatcher` (the default
:class:`MatchingEngine` implementation) which pairs simultaneous bid and
ask fills to identify trades.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ob_analytics._needleman_wunsch import align_sequences, create_similarity_matrix
from ob_analytics._utils import validate_columns, validate_non_empty
from ob_analytics.config import PipelineConfig


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
        """Add a ``matching.event`` column pairing bid/ask fills.

        For each unique fill volume, bid and ask fills within
        ``match_cutoff_ms`` are paired.  When a simple closest-match is
        ambiguous (duplicates or gaps), Needleman-Wunsch sequence
        alignment is used as a fallback.
        """
        validate_columns(
            events,
            {"direction", "fill", "original_number", "event.id", "timestamp"},
            "NeedlemanWunschMatcher.match",
        )
        validate_non_empty(events, "NeedlemanWunschMatcher.match")

        matched = self._run_matching(events)

        events["matching.event"] = np.nan
        matched_bids = pd.DataFrame(matched, columns=["event.id", "matching.event"])
        matched_asks = pd.DataFrame(matched, columns=["matching.event", "event.id"])
        events = pd.merge(events, matched_bids, on="event.id", how="left")
        events = pd.merge(events, matched_asks, on="event.id", how="left")

        events["matching.event"] = (
            events["matching.event_y"]
            .fillna(events["matching.event_x"])
            .fillna(events["matching.event"])
        )

        events = events.drop(columns=["matching.event_x", "matching.event_y"])
        return events

    def _run_matching(self, events: pd.DataFrame) -> np.ndarray:
        cut_off_td = np.timedelta64(self._config.match_cutoff_ms * 1_000_000, "ns")
        res: list[np.ndarray] = []
        cols = ["original_number", "event.id", "fill", "timestamp"]

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

            distance_matrix_ms = (
                (bids["timestamp"].values.reshape(-1, 1) - asks["timestamp"].values)
                .astype("timedelta64[ns]")
                .astype("int64")
            )

            if len(asks) == 1:
                distance_matrix_ms = distance_matrix_ms.reshape(-1, 1)

            closest_ask_indices = np.argmin(np.abs(distance_matrix_ms), axis=1)
            ask_event_ids = asks["event.id"].values[closest_ask_indices]

            mask = (
                np.abs(distance_matrix_ms[np.arange(len(bids)), closest_ask_indices])
                <= cut_off_td
            )
            ask_event_ids = np.where(mask, ask_event_ids, np.nan)

            if not any(pd.isna(ask_event_ids)) and len(ask_event_ids) == len(
                set(ask_event_ids)
            ):
                matches = np.column_stack((bids["event.id"], ask_event_ids))
                res.extend(matches)
            else:
                similarity_matrix = create_similarity_matrix(
                    bids["timestamp"], asks["timestamp"], cut_off_td
                )
                aligned_indices = align_sequences(similarity_matrix)
                matched_indices = aligned_indices[
                    np.abs(
                        (
                            bids["timestamp"].values[aligned_indices[:, 0]]
                            - asks["timestamp"].values[aligned_indices[:, 1]]
                        )
                    )
                    <= cut_off_td
                ]
                res.extend(
                    np.column_stack(
                        (
                            bids["event.id"].values[matched_indices[:, 0]],
                            asks["event.id"].values[matched_indices[:, 1]],
                        )
                    )
                )

        return np.array(res, dtype=int)


# ── Backward-compatible module-level function ─────────────────────────


def event_match(events: pd.DataFrame, cut_off_ms: int = 5000) -> pd.DataFrame:
    """Match market orders (takers) to limit orders (makers).

    This is a convenience wrapper around :class:`NeedlemanWunschMatcher`.

    Parameters
    ----------
    events : pandas.DataFrame
        A pandas DataFrame of order book events.
    cut_off_ms : int, optional
        The time window in milliseconds for considering candidate matches.
        Default is 5000.

    Returns
    -------
    pandas.DataFrame
        The events DataFrame with a 'matching.event' column indicating
        matched events.
    """
    config = PipelineConfig(match_cutoff_ms=cut_off_ms)
    return NeedlemanWunschMatcher(config).match(events)
