"""Tests for the bundled, gzip-compressed sample (roadmap 8.3a)."""

from __future__ import annotations

import pandas as pd

from ob_analytics import sample_csv_path, sample_data_dir


def test_sample_is_bundled_compressed() -> None:
    p = sample_csv_path()
    assert p.exists()
    assert p.name == "orders.csv.gz"
    assert p.parent == sample_data_dir()
    assert (sample_data_dir() / "trades.csv").exists()


def test_sample_reads_as_gzip() -> None:
    # pandas decompresses .gz transparently; a cheap head read proves it.
    head = pd.read_csv(sample_csv_path(), nrows=5)
    assert len(head) == 5
