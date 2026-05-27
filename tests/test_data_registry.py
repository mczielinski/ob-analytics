"""Coverage for the writer registry in ob_analytics.data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ob_analytics.data import (
    _WRITERS,
    list_writers,
    load_data,
    register_writer,
    save_data,
)


# ---------------------------------------------------------------------------
# A stub writer for tests
# ---------------------------------------------------------------------------


class _StubWriter:
    """Minimal DataWriter — records what it was asked to write."""

    written: list[tuple[Path, list[str]]] = []

    def write(self, data: dict, dest: str | Path, **kwargs) -> Path:
        p = Path(dest)
        self.written.append((p, sorted(data.keys())))
        return p


@pytest.fixture(autouse=True)
def _cleanup_registry():
    """Snapshot and restore the global writer registry around each test."""
    before = dict(_WRITERS)
    yield
    _WRITERS.clear()
    _WRITERS.update(before)
    _StubWriter.written.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWriterRegistry:
    def test_builtin_writers_present(self):
        names = list_writers()
        # Plan 2 registers both bitstamp and lobster as writer factories.
        assert "bitstamp" in names
        assert "lobster" in names

    def test_register_new_writer(self, tmp_path):
        register_writer("stub", lambda config, ctx: _StubWriter())
        assert "stub" in list_writers()

        df = pd.DataFrame({"a": [1, 2, 3]})
        save_data({"events": df}, tmp_path / "out", fmt="stub")
        assert _StubWriter.written
        written_path, keys = _StubWriter.written[0]
        assert written_path == tmp_path / "out"
        assert keys == ["events"]

    def test_unknown_format_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported format"):
            save_data({"events": pd.DataFrame()}, tmp_path / "x", fmt="no-such-fmt")

    def test_parquet_default(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        save_data({"events": df, "trades": df}, tmp_path, fmt="parquet")
        assert (tmp_path / "events.parquet").exists()
        assert (tmp_path / "trades.parquet").exists()

    def test_round_trip_parquet(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        save_data({"events": df}, tmp_path, fmt="parquet")
        loaded = load_data(tmp_path)
        assert "events" in loaded
        pd.testing.assert_frame_equal(loaded["events"], df)


class TestSaveDataExplicitWriter:
    def test_explicit_writer_overrides_fmt(self, tmp_path):
        register_writer("stub", lambda config, ctx: _StubWriter())
        stub2 = _StubWriter()
        save_data(
            {"events": pd.DataFrame()},
            tmp_path / "x",
            writer=stub2,
            fmt="parquet",  # ignored because writer= is set
        )
        # stub2 got the call (the registered "stub" did not)
        assert (tmp_path / "x", ["events"]) in _StubWriter.written
