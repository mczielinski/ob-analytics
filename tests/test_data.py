"""Tests for ob_analytics.data (save/load, zombie detection)."""


from pathlib import Path

import pandas as pd
import pytest

from ob_analytics.data import load_data, save_data


class TestSaveLoadParquet:
    def test_round_trip(self, tmp_path: Path):
        data = {
            "events": pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]}),
            "trades": pd.DataFrame({"x": ["foo", "bar"]}),
        }
        save_data(data, tmp_path / "output")
        loaded = load_data(tmp_path / "output")
        assert set(loaded.keys()) == {"events", "trades"}
        pd.testing.assert_frame_equal(loaded["events"], data["events"])
        pd.testing.assert_frame_equal(loaded["trades"], data["trades"])

    def test_load_from_empty_dir_raises(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="No .parquet files"):
            load_data(empty)


class TestSaveLoadPickle:
    def test_pickle_round_trip(self, tmp_path: Path):
        data = {"df": pd.DataFrame({"col": [1, 2, 3]})}
        pkl_path = tmp_path / "data.pkl"
        save_data(data, pkl_path, fmt="pickle")
        loaded = load_data(pkl_path)
        pd.testing.assert_frame_equal(loaded["df"], data["df"])


class TestSaveLoadEdgeCases:
    def test_unsupported_format_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Unsupported format"):
            save_data({}, tmp_path / "out", fmt="csv")

    def test_load_unsupported_extension_raises(self, tmp_path: Path):
        f = tmp_path / "data.json"
        f.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported format"):
            load_data(f)
