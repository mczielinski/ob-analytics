import pandas as pd
import numpy as np
from typing import List, Union, Tuple
from ob_analytics.needleman_wunsch import sMatrix
from ob_analytics.needleman_wunsch import alignS

def event_match(events: pd.DataFrame, cut_off_ms: int = 5000) -> pd.DataFrame:
    """
    Match Market Orders (takers) to Limit Orders (makers).

    Parameters:
    - events : pd.DataFrame
        DataFrame of order book events.
    - cut_off_ms : int
        Events occurring outside of this time (in milliseconds) will be considered as candidate matches.

    Returns:
    - pd.DataFrame
        DataFrame with matched events.
    """
    
    def fillId(src: pd.DataFrame, dst: pd.DataFrame) -> pd.DataFrame:
        return src[src["fill"].isin(dst["fill"])]

    def matcher(events: pd.DataFrame, cut_off_ms: int) -> List[Tuple[int, int]]:
        res = []
        bid_fills = events[(events["direction"] == "bid") & (events["fill"] != 0)]
        ask_fills = events[(events["direction"] == "ask") & (events["fill"] != 0)]
        
        id_bid_fills = fillId(bid_fills, ask_fills)
        id_ask_fills = fillId(ask_fills, bid_fills)

        for volume in id_bid_fills["fill"].unique():
            bids = id_bid_fills[id_bid_fills["fill"] == volume]
            asks = id_ask_fills[id_ask_fills["fill"] == volume]

            distance_matrix_ms = np.abs(np.array([[((bid - ask).total_seconds() * 1000) for ask in asks["timestamp"]] for bid in bids["timestamp"]]))
            # Simple matching: For each bid, find the index of the closest ask
            min_indices = np.argmin(np.where(distance_matrix_ms <= cut_off_ms, distance_matrix_ms, np.inf), axis=1)
            ask_event_ids = [asks.iloc[idx]["event.id"] for idx in min_indices]

            # Check for duplicates in the matched asks
            if not any(pd.Series(ask_event_ids).duplicated()):
                res.extend(zip(bids["event.id"].tolist(), ask_event_ids))
            else:
                sm = sMatrix(bids["timestamp"].to_list(), asks["timestamp"].to_list(),
                            lambda f1, f2: cut_off_ms / np.abs((f1 - f2).total_seconds() * 1000))
                aligned_idx = alignS(sm)
                matched_bids = bids.iloc[aligned_idx["a"].values - 1]
                matched_asks = asks.iloc[aligned_idx["b"].values - 1]
                time_diffs = matched_bids["timestamp"].values - matched_asks["timestamp"].values
                in_bounds = np.abs(pd.Series(time_diffs).dt.total_seconds() * 1000) <= cut_off_ms
                res.extend(zip(matched_bids.iloc[in_bounds.values,0].to_list(), matched_asks.iloc[in_bounds.values,0].to_list()))
        return res

    matched = matcher(events, cut_off_ms)
    events["matching.event"] = np.nan
    for bid, ask in matched:
        events.loc[events["event.id"] == bid, "matching.event"] = ask
        if not np.isnan(ask):
            events.loc[events["event.id"] == ask, "matching.event"] = bid
    return events