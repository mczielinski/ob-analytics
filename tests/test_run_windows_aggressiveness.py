"""A windowed Bitstamp run gives the same ``aggressiveness_bps`` as one run.

The ``run_windows`` docstring once warned that this column could differ on
Bitstamp, whose loader numbered events by order id.  The loader now numbers
them in time order, and the aggressiveness lookup reads the quote standing
before each order in time order, so a cut between windows changes nothing.
"""

from __future__ import annotations

import pandas as pd

from ob_analytics import Pipeline, sample_csv_path
from ob_analytics.data import load_data


def test_bitstamp_windowed_run_matches_a_single_run_on_aggressiveness(tmp_path):
    single = Pipeline().run(sample_csv_path()).events
    ordered = single["timestamp"].sort_values().reset_index(drop=True)
    cut = ordered.iloc[len(ordered) // 2].floor("s")

    out = Pipeline().run_windows(sample_csv_path(), [cut], tmp_path / "w")
    windowed = load_data(out)["events"]

    a, b = (
        frame[["event_id", "aggressiveness_bps"]]
        .sort_values("event_id")
        .reset_index(drop=True)
        for frame in (single, windowed)
    )
    pd.testing.assert_frame_equal(b, a)
