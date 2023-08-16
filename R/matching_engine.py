import pandas as pd
import numpy as np
from R.needleman_wunsch import sMatrix
from R.needleman_wunsch import alignS

def event_match(events: pd.DataFrame, cut_off_ms: int = 5000) -> pd.DataFrame:
    """
    Match bid and ask events based on fill volumes and timestamps.
    
    Parameters
    ----------
    events : pd.DataFrame
        Events dataframe.
    cut_off_ms : int, optional
        Maximum time difference in milliseconds for matching events. Defaults to 5000.

    Returns
    -------
    pd.DataFrame
        Dataframe with an additional 'matching.event' column.
    """
    
    def matcher() -> np.ndarray:
        """
        Match events using simple closest timestamp and alignment algorithm.

        Returns
        -------
        np.ndarray
            Array of matched event ids.
        """
        res = []
        cols = ["event.id", "fill", "timestamp"]

        bid_fills = events[(events["direction"]=="bid") & (events["fill"] != 0)][cols]
        ask_fills = events[(events["direction"]=="ask") & (events["fill"] != 0)][cols]

        # Identifiable bid and ask fills
        def fill_id(src, dst):
            return src[src['fill'].isin(dst['fill'])].sort_values(by=["fill", "timestamp"])

        id_bid_fills = fill_id(bid_fills, ask_fills)
        id_ask_fills = fill_id(ask_fills, bid_fills)

        for volume in id_bid_fills["fill"].unique():
            bids = id_bid_fills[id_bid_fills["fill"] == volume]
            asks = id_ask_fills[id_ask_fills["fill"] == volume]

            distance_matrix_ms = np.abs(np.subtract.outer(bids["timestamp"].astype(int).values, asks["timestamp"].astype(int).values))

            # Simple matching
            ask_event_ids = np.array([asks["event.id"].iloc[np.argmin(row)] if np.min(row) <= cut_off_ms else np.nan for row in distance_matrix_ms])

            if len(np.unique(ask_event_ids[~np.isnan(ask_event_ids)])) == len(ask_event_ids[~np.isnan(ask_event_ids)]):
                matches = np.column_stack((bids["event.id"].values, ask_event_ids))
                matches = matches[~np.isnan(matches[:, 1])]
                res.append(matches)
            else:
                # Alignment algorithm
                sm = sMatrix(bids["timestamp"].astype(int).values, asks["timestamp"].astype(int).values, filter_func=lambda f1, f2: cut_off_ms if abs(f1 - f2) == 0 else cut_off_ms / abs(f1 - f2))
                aligned_idx = alignS(sm, gap=-1)
                aligned_idx -= aligned_idx
                
                matched_bids = bids.iloc[aligned_idx[:, 1]]
                matched_asks = asks.iloc[aligned_idx[:, 0]]
                in_bounds = np.abs(matched_bids["timestamp"].astype(int).values - matched_asks["timestamp"].astype(int).values) <= cut_off_ms

                res.append(np.column_stack((matched_bids[in_bounds]["event.id"].values, matched_asks[in_bounds]["event.id"].values)))

        return np.vstack(res)

    matched = matcher()
    matched = np.unique(matched,axis=0)

    # Ensure no duplicates
    assert len(matched[:, 0]) == len(np.unique(matched[:, 0]))
    assert len(matched[:, 1]) == len(np.unique(matched[:, 1]))

    events["matching.event"] = np.nan
    bid_matches = np.searchsorted(events["event.id"].values, matched[:, 0])
    ask_matches = np.searchsorted(events["event.id"].values, matched[:, 1])

    events.iloc[bid_matches, events.columns.get_loc("matching.event")] = matched[:, 1]
    events.iloc[ask_matches, events.columns.get_loc("matching.event")] = matched[:, 0]

    return events
