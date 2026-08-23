"""Tests for venue sequence + local ingest counter (issue #146).

Covers the gap-detection helper (dropped / reordered / clean streams, per-channel
grouping, consecutive-duplicate collapsing), the optional ``sequence`` /
``ingest_seq`` columns the loaders attach under ``track_sequence`` (present with
the right dtype; absent for sources with no venue sequence; unchanged by
default), and the summary fields surfaced in ``data_quality_summary``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ob_analytics import (
    FeedType,
    SequenceGapReport,
    data_quality_summary,
    detect_sequence_gaps,
)
from ob_analytics._utils import empty_trades
from ob_analytics.analytics import set_order_types
from ob_analytics.bitstamp import BitstampLoader
from ob_analytics.config import PipelineConfig
from ob_analytics.lobster import LobsterLoader
from ob_analytics.schemas import INGEST_SEQ_COLUMN, SEQUENCE_COLUMN

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _framed(
    seqs,
    *,
    venues: list[str] | None = None,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """A frame carrying a venue ``sequence`` in ingest order (``ingest_seq``)."""
    n = len(seqs)
    data: dict[str, object] = {
        SEQUENCE_COLUMN: pd.array(seqs, dtype="Int64"),
        INGEST_SEQ_COLUMN: np.arange(n, dtype="int64"),
    }
    if venues is not None:
        data["venue"] = venues
    if symbols is not None:
        data["symbol"] = symbols
    return pd.DataFrame(data)


_BASE = pd.Timestamp("2026-01-05 10:00:00")


def _classified(rows: list[tuple]) -> pd.DataFrame:
    """Canonical, type-classified events. Each row is
    ``(event_id, id, t_seconds, price, volume, direction, action, fill)``."""
    ts = pd.Series([_BASE + pd.Timedelta(seconds=r[2]) for r in rows]).astype(
        "datetime64[ns]"
    )
    events = pd.DataFrame(
        {
            "event_id": np.array([r[0] for r in rows], dtype=np.int64),
            "id": np.array([r[1] for r in rows], dtype=np.int64),
            "timestamp": ts,
            "exchange_timestamp": ts.copy(),
            "price": np.array([r[3] for r in rows], dtype=np.float64),
            "volume": np.array([r[4] for r in rows], dtype=np.float64),
            "direction": pd.Categorical(
                [r[5] for r in rows], categories=["bid", "ask"], ordered=True
            ),
            "action": pd.Categorical(
                [r[6] for r in rows],
                categories=["created", "changed", "deleted"],
                ordered=True,
            ),
            "fill": np.array([r[7] for r in rows], dtype=np.float64),
        }
    )
    return set_order_types(events, empty_trades())


def _write_bitstamp_csv(path: Path, rows: list[dict], *, with_sequence: bool) -> Path:
    """Write a minimal Bitstamp orders CSV, optionally with a ``sequence`` col."""
    cols = [
        "id",
        "timestamp",
        "exchange_timestamp",
        "price",
        "volume",
        "action",
        "direction",
    ]
    if with_sequence:
        cols.append("sequence")
    pd.DataFrame(rows)[cols].to_csv(path, index=False)
    return path


# Arrival-ordered Bitstamp events; venue sequence 102 -> 104 skips 103.
_BITSTAMP_ROWS = [
    {
        "id": 1,
        "timestamp": 1000,
        "exchange_timestamp": 1000,
        "price": 99.0,
        "volume": 2.0,
        "action": "created",
        "direction": "bid",
        "sequence": 101,
    },
    {
        "id": 2,
        "timestamp": 1000,
        "exchange_timestamp": 1000,
        "price": 101.0,
        "volume": 2.0,
        "action": "created",
        "direction": "ask",
        "sequence": 102,
    },
    {
        "id": 1,
        "timestamp": 2000,
        "exchange_timestamp": 2000,
        "price": 99.0,
        "volume": 0.0,
        "action": "deleted",
        "direction": "bid",
        "sequence": 104,
    },
]


# ---------------------------------------------------------------------------
# detect_sequence_gaps — pure helper
# ---------------------------------------------------------------------------


class TestDetectSequenceGaps:
    def test_clean_stream_is_not_flagged(self):
        report = detect_sequence_gaps(_framed([1, 2, 3, 4, 5]))
        assert isinstance(report, SequenceGapReport)
        assert report.clean
        assert report.has_sequence
        assert report.n_sequenced == 5
        assert report.n_missing == 0
        assert report.n_out_of_order == 0
        assert report.first_break_seq is None

    def test_dropped_message_is_flagged(self):
        # 3 -> 5 skips exactly one number.
        report = detect_sequence_gaps(_framed([1, 2, 3, 5, 6]))
        assert not report.clean
        assert report.n_missing == 1
        assert report.n_out_of_order == 0
        assert report.max_gap == 1
        assert report.first_break_seq == 3

    def test_multi_message_gap_counts_each_missing(self):
        # 2 -> 6 skips 3, 4, 5.
        report = detect_sequence_gaps(_framed([1, 2, 6, 7]))
        assert report.n_missing == 3
        assert report.max_gap == 3

    def test_reordered_message_is_flagged(self):
        # 5 arrives before 4 (a reorder in arrival order).
        report = detect_sequence_gaps(_framed([1, 2, 3, 5, 4, 6]))
        assert not report.clean
        assert report.n_out_of_order >= 1

    def test_consecutive_duplicates_collapse_to_one_update(self):
        # One book update emits several rows that share its sequence.
        report = detect_sequence_gaps(_framed([1, 1, 1, 2, 2, 3]))
        assert report.clean
        assert report.n_sequenced == 6
        assert report.n_updates == 3

    def test_non_consecutive_repeat_is_out_of_order(self):
        report = detect_sequence_gaps(_framed([1, 2, 1, 3]))
        assert not report.clean
        assert report.n_out_of_order >= 1

    def test_missing_column_is_trivially_clean(self):
        report = detect_sequence_gaps(pd.DataFrame({"price": [1.0, 2.0]}))
        assert report.clean
        assert not report.has_sequence
        assert report.n_sequenced == 0

    def test_empty_frame_is_clean(self):
        report = detect_sequence_gaps(_framed([]))
        assert report.clean
        assert not report.has_sequence

    def test_all_na_sequence_is_clean(self):
        report = detect_sequence_gaps(_framed([pd.NA, pd.NA, pd.NA]))
        assert report.clean
        assert not report.has_sequence

    def test_grouping_isolates_channels(self):
        # Two channels, each clean on its own, interleaved in arrival order.
        frame = _framed(
            [1, 10, 2, 11, 3, 12],
            venues=["A", "B", "A", "B", "A", "B"],
        )
        # Per-venue: A=[1,2,3], B=[10,11,12] — both clean.
        assert detect_sequence_gaps(frame).clean
        # Ignoring the channel, the merged stream looks badly broken.
        assert not detect_sequence_gaps(frame, group_cols=()).clean

    def test_unordered_frame_is_sorted_by_ingest_seq(self):
        # A physically shuffled frame is still read in ingest order.
        frame = _framed([1, 2, 3, 4]).sample(frac=1.0, random_state=0)
        assert detect_sequence_gaps(frame).clean

    def test_to_dict_is_json_serialisable(self):
        import json

        report = detect_sequence_gaps(_framed([1, 2, 4]))
        payload = json.loads(json.dumps(report.to_dict()))
        assert payload["n_missing"] == 1


# ---------------------------------------------------------------------------
# Loader integration — columns present / absent / opt-in
# ---------------------------------------------------------------------------


class TestBitstampSequenceColumns:
    def test_carries_sequence_and_ingest_seq(self, tmp_path):
        csv = _write_bitstamp_csv(
            tmp_path / "orders.csv", _BITSTAMP_ROWS, with_sequence=True
        )
        events = BitstampLoader(PipelineConfig(track_sequence=True)).load(csv)

        assert SEQUENCE_COLUMN in events.columns
        assert INGEST_SEQ_COLUMN in events.columns
        assert events[SEQUENCE_COLUMN].dtype == "Int64"
        assert events[INGEST_SEQ_COLUMN].dtype == np.dtype("int64")

        # The venue gap (skipped 103) is detected in arrival order despite the
        # loader sorting rows by id.
        report = detect_sequence_gaps(events)
        assert report.n_missing == 1
        assert not report.clean

    def test_without_sequence_column_still_loads(self, tmp_path):
        csv = _write_bitstamp_csv(
            tmp_path / "orders.csv", _BITSTAMP_ROWS, with_sequence=False
        )
        events = BitstampLoader(PipelineConfig(track_sequence=True)).load(csv)

        # No venue sequence in the file: the local counter is still attached.
        assert INGEST_SEQ_COLUMN in events.columns
        assert SEQUENCE_COLUMN not in events.columns
        assert detect_sequence_gaps(events).clean

    def test_default_config_adds_no_columns(self, tmp_path):
        # Off by default: on a source with no venue sequence (like the bundled
        # sample) the loader adds nothing, so existing outputs are unchanged.
        csv = _write_bitstamp_csv(
            tmp_path / "orders.csv", _BITSTAMP_ROWS, with_sequence=False
        )
        events = BitstampLoader().load(csv)
        assert SEQUENCE_COLUMN not in events.columns
        assert INGEST_SEQ_COLUMN not in events.columns

    def test_off_does_not_normalise_a_raw_sequence_column(self, tmp_path):
        # A ``sequence`` column already in the file is passed through as-is (not
        # normalised to Int64) and the local counter is not added.
        csv = _write_bitstamp_csv(
            tmp_path / "orders.csv", _BITSTAMP_ROWS, with_sequence=True
        )
        events = BitstampLoader().load(csv)
        assert INGEST_SEQ_COLUMN not in events.columns
        assert events[SEQUENCE_COLUMN].dtype != "Int64"


class TestLobsterSequenceColumns:
    def _write_messages(self, path: Path) -> Path:
        # LOBSTER message rows: time, event_type, id, size, price, direction.
        rows = [
            (34200.0, 1, 100, 10, 2459800, 1),
            (34200.5, 1, 101, 5, 2460000, -1),
            (34201.0, 3, 100, 10, 2459800, 1),
        ]
        pd.DataFrame(rows).to_csv(path, header=False, index=False)
        return path

    def test_has_ingest_seq_but_no_venue_sequence(self, tmp_path):
        msg = self._write_messages(tmp_path / "AAPL_message.csv")
        loader = LobsterLoader(
            PipelineConfig(track_sequence=True), trading_date="2024-01-02"
        )
        events = loader.load(msg)

        # LOBSTER numbers nothing, so only the local counter appears.
        assert INGEST_SEQ_COLUMN in events.columns
        assert events[INGEST_SEQ_COLUMN].dtype == np.dtype("int64")
        assert SEQUENCE_COLUMN not in events.columns
        assert detect_sequence_gaps(events).clean

    def test_default_config_adds_no_columns(self, tmp_path):
        msg = self._write_messages(tmp_path / "AAPL_message.csv")
        loader = LobsterLoader(trading_date="2024-01-02")
        events = loader.load(msg)
        assert INGEST_SEQ_COLUMN not in events.columns


# ---------------------------------------------------------------------------
# data_quality_summary integration
# ---------------------------------------------------------------------------


class TestDataQualitySummary:
    def _classified_with_sequence(self, seqs) -> pd.DataFrame:
        events = _classified(
            [
                (1, 1, 0.0, 99.0, 2.0, "bid", "created", 0.0),
                (2, 2, 1.0, 101.0, 2.0, "ask", "created", 0.0),
                (3, 3, 2.0, 98.0, 1.0, "bid", "created", 0.0),
                (4, 4, 3.0, 102.0, 1.0, "ask", "created", 0.0),
            ]
        )
        events[SEQUENCE_COLUMN] = pd.array(seqs, dtype="Int64")
        events[INGEST_SEQ_COLUMN] = np.arange(len(events), dtype="int64")
        return events

    def test_surfaces_sequence_gaps(self):
        # Sequence 3 -> 5 skips one message.
        events = self._classified_with_sequence([1, 2, 3, 5])
        summary = data_quality_summary(
            events, empty_trades(), feed_type=FeedType.DIFF_FEED
        )
        assert summary.events_with_sequence == 4
        assert summary.sequence_gaps == 1
        assert summary.sequence_out_of_order == 0
        # Surfaced in the rendered report and the JSON dict.
        assert "venue sequence" in summary.render()
        assert summary.to_dict()["sequence_gaps"] == 1

    def test_clean_sequence_reports_zero(self):
        events = self._classified_with_sequence([1, 2, 3, 4])
        summary = data_quality_summary(events, empty_trades())
        assert summary.sequence_gaps == 0
        assert summary.sequence_out_of_order == 0
        assert summary.events_with_sequence == 4

    def test_no_sequence_columns_report_zero(self):
        events = _classified(
            [
                (1, 1, 0.0, 99.0, 2.0, "bid", "created", 0.0),
                (2, 2, 1.0, 101.0, 2.0, "ask", "created", 0.0),
            ]
        )
        summary = data_quality_summary(events, empty_trades())
        assert summary.events_with_sequence == 0
        assert summary.sequence_gaps == 0
        # The report still renders the (zeroed) line.
        assert "venue sequence" in summary.render()
