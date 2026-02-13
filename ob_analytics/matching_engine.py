import numpy as np
import pandas as pd

from ob_analytics.needleman_wunsch import align_sequences, create_similarity_matrix


def event_match(events: pd.DataFrame, cut_off_ms: int = 5000) -> pd.DataFrame:
    """
    Match market orders (takers) to limit orders (makers).

    Parameters
    ----------
    events : pandas.DataFrame
        A pandas DataFrame of order book events.
    cut_off_ms : int, optional
        The time window in milliseconds for considering candidate matches. Default is 5000.

    Returns
    -------
    pandas.DataFrame
        The events DataFrame with a 'matching.event' column indicating matched events.
    """

    def matcher() -> np.ndarray:
        """
        Internal function to perform the matching logic.

        Returns
        -------
        numpy.ndarray
            An array of matched event IDs.
        """
        cut_off_td = np.timedelta64(cut_off_ms * 1000000, "ns")
        res = []
        cols = ["original_number", "event.id", "fill", "timestamp"]

        bid_fills = events[(events["direction"] == "bid") & (events["fill"] != 0)][cols]
        ask_fills = events[(events["direction"] == "ask") & (events["fill"] != 0)][cols]

        def fill_id(src: pd.DataFrame, dst: pd.DataFrame) -> pd.DataFrame:
            """
            Filter fills based on matching fill amounts and sort them.

            Parameters
            ----------
            src : pandas.DataFrame
                Source DataFrame containing fills.
            dst : pandas.DataFrame
                Destination DataFrame to compare fills with.

            Returns
            -------
            pandas.DataFrame
                Filtered and sorted DataFrame of fills.
            """
            id_fills = src[src["fill"].isin(dst["fill"])]
            return id_fills.sort_values(
                by=["fill", "timestamp"], ascending=[False, True], kind="stable"
            )

        id_bid_fills = fill_id(bid_fills, ask_fills)
        id_ask_fills = fill_id(ask_fills, bid_fills)

        for volume in id_bid_fills["fill"].unique():
            bids = id_bid_fills[id_bid_fills["fill"] == volume]
            asks = id_ask_fills[id_ask_fills["fill"] == volume]

            # Calculate distance matrix in milliseconds
            distance_matrix_ms = (
                (bids["timestamp"].values.reshape(-1, 1) - asks["timestamp"].values)
                .astype("timedelta64[ns]")
                .astype("int64")
            )

            # Handle single ask case
            if len(asks) == 1:
                distance_matrix_ms = distance_matrix_ms.reshape(-1, 1)

            # Find the closest ask indices for each bid
            closest_ask_indices = np.argmin(np.abs(distance_matrix_ms), axis=1)

            # Retrieve event ids of the closest asks
            ask_event_ids = asks["event.id"].values[closest_ask_indices]

            # Create a mask for the valid matches within cutoff
            mask = (
                np.abs(distance_matrix_ms[np.arange(len(bids)), closest_ask_indices])
                <= cut_off_td
            )

            # Apply mask to get the final ask event ids
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

    matched = matcher()

    # pd.DataFrame(matched).to_csv('matched.csv')
    events["matching.event"] = np.nan
    # matched[:, 0] = bid event ids, matched[:, 1] = ask event ids
    matched_bids = pd.DataFrame(matched, columns=["event.id", "matching.event"])
    matched_asks = pd.DataFrame(matched, columns=["matching.event", "event.id"])
    events = pd.merge(events, matched_bids, on="event.id", how="left")
    events = pd.merge(events, matched_asks, on="event.id", how="left")

    # Combine columns into a single 'matching.event' column
    events["matching.event"] = (
        events["matching.event_y"]
        .fillna(events["matching.event_x"])
        .fillna(events["matching.event"])
    )

    # Drop the unnecessary columns
    events = events.drop(columns=["matching.event_x", "matching.event_y"])
    return events
