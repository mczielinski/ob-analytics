"""Port of R testthat/test-event-match.R to pytest.

The R package has 3 tests for eventMatch; these are direct translations
using the same timestamps, directions, fill volumes, and expected outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ob_analytics.matching_engine import event_match


def _make_events(
    timestamps: list[str],
    directions: list[str],
    event_ids: list[int],
    fill: int = 1234,
) -> pd.DataFrame:
    """Build a minimal events DataFrame matching R's test structure."""
    n = len(timestamps)
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, utc=True),
            "direction": pd.Categorical(
                directions, categories=["bid", "ask"], ordered=True
            ),
            "event.id": event_ids,
            "fill": [fill] * n,
            "original_number": list(range(1, n + 1)),
        }
    )


class TestSimpleEventMatching:
    """R: 'simple event matching works' -- 4 events, 2 bid + 2 ask, clean pairs."""

    def test_all_events_matched(self):
        events = _make_events(
            timestamps=[
                "2015-10-10 21:32:00.000",
                "2015-10-10 21:32:00.010",
                "2015-10-10 21:32:10.000",
                "2015-10-10 21:32:10.010",
            ],
            directions=["bid", "ask", "bid", "ask"],
            event_ids=[1, 2, 3, 4],
        )
        matched = event_match(events, cut_off_ms=1000)
        expected = [2, 1, 4, 3]
        result = matched.sort_values("event.id")["matching.event"].tolist()
        assert result == expected


class TestLeftOverEventMatching:
    """R: 'left over event matching works' -- 5 events, odd ask has no match."""

    def test_unmatched_event_is_nan(self):
        events = _make_events(
            timestamps=[
                "2015-10-10 21:32:00.000",
                "2015-10-10 21:32:00.010",
                "2015-10-10 21:32:10.000",
                "2015-10-10 21:32:10.010",
                "2015-10-10 21:33:00.000",
            ],
            directions=["bid", "ask", "bid", "ask", "ask"],
            event_ids=[1, 2, 3, 4, 5],
        )
        matched = event_match(events, cut_off_ms=1000)
        result = matched.sort_values("event.id")["matching.event"].tolist()
        assert result[0] == 2
        assert result[1] == 1
        assert result[2] == 4
        assert result[3] == 3
        assert np.isnan(result[4])


class TestConflictingEventMatching:
    """R: 'conflicting event matching works' -- ambiguous sequences needing NW alignment."""

    def test_conflict_resolution_case_1(self):
        events = _make_events(
            timestamps=[
                "2015-10-10 21:32:00.000",
                "2015-10-10 21:32:00.010",
                "2015-10-10 21:32:10.000",
                "2015-10-10 21:32:10.080",
                "2015-10-10 21:32:10.090",
            ],
            directions=["bid", "ask", "bid", "bid", "ask"],
            event_ids=[1, 2, 3, 4, 5],
        )
        matched = event_match(events, cut_off_ms=1000)
        result = matched.sort_values("event.id")["matching.event"].tolist()
        assert result[0] == 2
        assert result[1] == 1
        assert np.isnan(result[2])
        assert result[3] == 5
        assert result[4] == 4

    def test_conflict_resolution_case_2(self):
        events = _make_events(
            timestamps=[
                "2015-10-10 21:32:00.000",
                "2015-10-10 21:32:00.010",
                "2015-10-10 21:32:01.000",
                "2015-10-10 21:32:01.010",
                "2015-10-10 21:32:01.090",
            ],
            directions=["bid", "ask", "ask", "bid", "bid"],
            event_ids=[1, 2, 3, 4, 5],
        )
        matched = event_match(events, cut_off_ms=1000)
        result = matched.sort_values("event.id")["matching.event"].tolist()
        assert result[0] == 2
        assert result[1] == 1
        assert result[2] == 4
        assert result[3] == 3
        assert np.isnan(result[4])
