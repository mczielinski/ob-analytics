"""Tests for ob_analytics.data (save/load)."""

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


def _assert_same_result(back, result) -> None:
    assert back.config == result.config
    assert back.level is result.level
    for name, frame in result._frames().items():
        # Parquet gives a missing value in an object column back as None, not
        # pd.NA; that is how load_data has always read it.
        loaded, frame = getattr(back, name).copy(), frame.copy()
        objects = [c for c in frame.columns if frame[c].dtype == object]
        for df in (frame, loaded):
            df[objects] = df[objects].astype(object).where(df[objects].notna(), None)
        pd.testing.assert_frame_equal(loaded, frame, check_categorical=False)


class TestLoadResult:
    """``load_result`` gives back the result ``save_data`` was given."""

    def test_a_lobster_run_keeps_its_units(
        self, tmp_path: Path, lobster_day_dir, lobster_day_pipeline
    ):
        from ob_analytics.data import load_result
        from ob_analytics.visualization.gallery import display_result

        result = lobster_day_pipeline.run(lobster_day_dir)
        assert result.config.lot_size == 1.0
        save_data(result, tmp_path / "saved")
        back = load_result(tmp_path / "saved")

        _assert_same_result(back, result)
        assert display_result(back).trades["volume"].tolist() == [40.0, 25.0]
        assert display_result(back).trades["price"].tolist() == [101.0, 100.0]

    def test_a_custom_config_comes_back_whole(
        self, tmp_path: Path, lobster_day_dir, lobster_day_pipeline
    ):
        from dataclasses import replace

        from ob_analytics.config import PipelineConfig
        from ob_analytics.data import load_result

        result = lobster_day_pipeline.run(lobster_day_dir)
        config = PipelineConfig(
            tick_size=0.001,
            price_decimals=3,
            lot_size=1.0,
            volume_decimals=0,
            depth_bps=50,
        )
        result = replace(result, config=config)
        save_data(result, tmp_path / "saved")

        _assert_same_result(load_result(tmp_path / "saved"), result)

    def test_an_l2_run_comes_back_at_l2(self, tmp_path: Path):
        from ob_analytics import Pipeline
        from ob_analytics.data import load_result
        from ob_analytics.datasets import toy_l2_depth, toy_l2_trades
        from ob_analytics.depth_l2 import DepthCsvWriter
        from ob_analytics.protocols import Level

        DepthCsvWriter().write(
            {"depth": toy_l2_depth(), "trades": toy_l2_trades()}, tmp_path / "l2"
        )
        result = Pipeline.from_source("depth_csv").run(tmp_path / "l2")
        save_data(result, tmp_path / "saved")
        back = load_result(tmp_path / "saved")

        assert back.level is Level.L2
        _assert_same_result(back, result)

    def test_a_file_with_only_the_tick_and_lot_takes_decimals_from_them(
        self, tmp_path: Path, lobster_day_dir, lobster_day_pipeline
    ):
        """A windowed run's folder records the tick and lot size only."""
        from ob_analytics.data import load_result

        lobster_day_pipeline.run_windows(
            lobster_day_dir, "2024-01-02 13:30:02", tmp_path / "saved"
        )
        back = load_result(tmp_path / "saved")

        assert back.config.tick_size == 0.01
        assert back.config.price_decimals == 2
        assert back.config.lot_size == 1.0
        assert back.config.volume_decimals == 0

    def test_a_pickle_is_refused(
        self, tmp_path: Path, lobster_day_dir, lobster_day_pipeline
    ):
        """A pickle keeps no config, so its prices would read as ticks."""
        from ob_analytics.data import load_result
        from ob_analytics.exceptions import ConfigError

        result = lobster_day_pipeline.run(lobster_day_dir)
        save_data(result, tmp_path / "run.pkl", fmt="pickle")
        with pytest.raises(ConfigError, match="Parquet"):
            load_result(tmp_path / "run.pkl")

    def test_a_config_this_version_cannot_read_falls_back_to_the_grid(
        self, tmp_path: Path, lobster_day_dir, lobster_day_pipeline
    ):
        """A later version's config, or a damaged one, still loads the run."""
        import pyarrow.parquet as pq

        from ob_analytics.data import load_result
        from ob_analytics.schemas import CONFIG_KEY

        save_data(lobster_day_pipeline.run(lobster_day_dir), tmp_path / "saved")
        for file in (tmp_path / "saved").glob("*.parquet"):
            table = pq.read_table(file)
            metadata = {**table.schema.metadata, CONFIG_KEY: b'{"timestamp_unit": "s"'}
            pq.write_table(table.replace_schema_metadata(metadata), file)

        with pytest.warns(UserWarning, match="config"):
            back = load_result(tmp_path / "saved")
        assert back.config.lot_size == 1.0
        assert back.config.volume_decimals == 0

    def test_a_missing_folder_is_refused(self, tmp_path: Path):
        from ob_analytics.data import load_result
        from ob_analytics.exceptions import ConfigError

        with pytest.raises(ConfigError, match="not a Parquet folder"):
            load_result(tmp_path / "typo")

    def test_integer_tables_with_no_units_are_warned_about(
        self, tmp_path: Path, lobster_day_dir, lobster_day_pipeline
    ):
        """Saved without a config, ticks and lots cannot be read back."""
        from ob_analytics.data import load_result

        result = lobster_day_pipeline.run(lobster_day_dir)
        save_data(result._frames(), tmp_path / "saved")
        with pytest.warns(UserWarning, match="tick_size"):
            load_result(tmp_path / "saved")

    def test_a_folder_missing_a_table_is_refused(self, tmp_path: Path):
        from ob_analytics.data import load_result
        from ob_analytics.exceptions import ConfigError

        save_data({"events": pd.DataFrame({"a": [1]})}, tmp_path / "saved")
        with pytest.raises(ConfigError, match="depth_summary"):
            load_result(tmp_path / "saved")


class TestSaveLoadEdgeCases:
    def test_unsupported_format_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Unsupported format"):
            save_data({}, tmp_path / "out", fmt="csv")

    def test_load_unsupported_extension_raises(self, tmp_path: Path):
        f = tmp_path / "data.json"
        f.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported format"):
            load_data(f)
