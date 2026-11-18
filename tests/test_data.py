"""Tests for ob_analytics.data (save/load)."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from ob_analytics.data import load_data, save_data
from ob_analytics.exceptions import UnknownNameError
from tests._golden import assert_same_result


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

    def test_failed_overwrite_preserves_saved_run(
        self, tmp_path: Path, lobster_day_dir, lobster_day_pipeline
    ):
        from ob_analytics.data import load_result

        result = lobster_day_pipeline.run(lobster_day_dir)
        save_data(result, tmp_path)
        with pytest.raises(pa.ArrowInvalid):
            save_data(
                {
                    "events": pd.DataFrame({"x": [2]}),
                    "trades": pd.DataFrame({"x": [object()]}),
                },
                tmp_path,
            )
        assert_same_result(load_result(tmp_path), result)

    def test_publish_failure_restores_replaced_files(self, tmp_path: Path):
        original = pd.DataFrame({"x": [1]})
        changed = pd.DataFrame({"x": [2]})
        save_data({"first": original}, tmp_path)
        (tmp_path / "blocked.parquet").mkdir()
        with pytest.raises(OSError):
            save_data({"first": changed, "new": changed, "blocked": changed}, tmp_path)
        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_path / "first.parquet"), original
        )
        assert not (tmp_path / "new.parquet").exists()

    def test_failed_rollback_step_keeps_going_and_names_the_backup(
        self, tmp_path: Path, monkeypatch
    ):
        # Issue #524: one step of the rollback fails; the others still run,
        # the first error is the one raised, and its notes name what is left.
        earlier = {name: pd.DataFrame({"x": [i]}) for i, name in enumerate("abc")}
        for name, frame in earlier.items():
            save_data({name: frame}, tmp_path)
        replace = Path.replace

        def fail_twice(self, target):
            if Path(target).name == "c.parquet":
                raise OSError("disk full")
            if self.name == ".b.parquet.bak":
                raise PermissionError("no access")
            return replace(self, target)

        monkeypatch.setattr(Path, "replace", fail_twice)
        changed = pd.DataFrame({"x": [9]})

        with pytest.raises(OSError, match="disk full") as caught:
            save_data({"a": changed, "b": changed, "c": changed}, tmp_path)

        notes = "\n".join(caught.value.__notes__)
        backup = tmp_path / ".b.parquet.bak"
        assert str(tmp_path / "b.parquet") in notes
        assert str(backup) in notes
        assert "no access" in notes
        # Put back despite the failure after it, and the failed file untouched.
        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_path / "a.parquet"), earlier["a"]
        )
        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_path / "c.parquet"), earlier["c"]
        )
        # The backup named in the error survives, holding the earlier file.
        pd.testing.assert_frame_equal(pd.read_parquet(backup), earlier["b"])
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            ".b.parquet.bak",
            "a.parquet",
            "b.parquet",
            "c.parquet",
        ]
        # A later save takes another backup name and leaves this one alone.
        monkeypatch.undo()
        save_data({"b": changed}, tmp_path)
        pd.testing.assert_frame_equal(pd.read_parquet(backup), earlier["b"])

    def test_load_from_empty_dir_raises(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="No .parquet files"):
            load_data(empty)


class TestInstallFiles:
    """Moving a set of files into place, all or none (issue #524)."""

    @staticmethod
    def _staged(tmp_path: Path, names: str) -> list[tuple[Path, Path]]:
        staging = tmp_path / "staging"
        staging.mkdir()
        files = []
        for name in names:
            (staging / name).write_text(f"new {name}")
            files.append((staging / name, tmp_path / name))
        return files

    def test_a_new_file_that_cannot_be_removed_is_named(
        self, tmp_path: Path, monkeypatch
    ):
        from ob_analytics.data import _install_files

        (tmp_path / "a").write_text("old a")
        files = self._staged(tmp_path, "abc")
        replace, unlink = Path.replace, Path.unlink

        def fail_on_c(self, target):
            if Path(target).name == "c":
                raise OSError("disk full")
            return replace(self, target)

        def refuse_b(self, missing_ok=False):
            if self == tmp_path / "b":
                raise PermissionError("no access")
            return unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "replace", fail_on_c)
        monkeypatch.setattr(Path, "unlink", refuse_b)

        with pytest.raises(OSError, match="disk full") as caught:
            _install_files(files)

        assert caught.value.__notes__ == [
            (
                f"Could not remove {tmp_path / 'b'}, which did not exist "
                "before (no access)."
            )
        ]
        assert (tmp_path / "a").read_text() == "old a"
        assert not (tmp_path / ".a.bak").exists()

    def test_a_backup_of_the_failed_file_that_stays_is_named(
        self, tmp_path: Path, monkeypatch
    ):
        from ob_analytics.data import _install_files

        for name in "ab":
            (tmp_path / name).write_text(f"old {name}")
        files = self._staged(tmp_path, "ab")
        replace, unlink = Path.replace, Path.unlink

        def fail_on_b(self, target):
            if Path(target).name == "b":
                raise OSError("disk full")
            return replace(self, target)

        def refuse_b_backup(self, missing_ok=False):
            if self.name == ".b.bak":
                raise PermissionError("no access")
            return unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "replace", fail_on_b)
        monkeypatch.setattr(Path, "unlink", refuse_b_backup)

        with pytest.raises(OSError, match="disk full") as caught:
            _install_files(files)

        assert caught.value.__notes__ == [
            (
                f"Could not remove the backup {tmp_path / '.b.bak'}; "
                f"{tmp_path / 'b'} is intact (no access)."
            )
        ]
        assert (tmp_path / "a").read_text() == "old a"
        assert (tmp_path / "b").read_text() == "old b"

    def test_an_interrupt_is_rolled_back_too(self, tmp_path: Path, monkeypatch):
        from ob_analytics.data import _install_files

        (tmp_path / "a").write_text("old a")
        files = self._staged(tmp_path, "ab")
        replace = Path.replace

        def interrupt_on_b(self, target):
            if Path(target).name == "b":
                raise KeyboardInterrupt
            return replace(self, target)

        monkeypatch.setattr(Path, "replace", interrupt_on_b)

        with pytest.raises(KeyboardInterrupt):
            _install_files(files)

        assert sorted(p.name for p in tmp_path.iterdir()) == ["a", "staging"]
        assert (tmp_path / "a").read_text() == "old a"

    def test_a_backup_left_after_success_warns(self, tmp_path: Path, monkeypatch):
        from ob_analytics.data import _install_files

        for name in "ab":
            (tmp_path / name).write_text(f"old {name}")
        files = self._staged(tmp_path, "ab")
        unlink = Path.unlink

        def refuse_a_backup(self, missing_ok=False):
            if self.name == ".a.bak":
                raise PermissionError("no access")
            return unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", refuse_a_backup)

        with pytest.warns(UserWarning, match=r"\.a\.bak"):
            _install_files(files)

        assert (tmp_path / "a").read_text() == "new a"
        assert (tmp_path / "b").read_text() == "new b"
        # The other backup is still removed.
        assert not (tmp_path / ".b.bak").exists()


class TestLoaderCounts:
    """The loader's counts survive a saved run (issue #503)."""

    def _flagged_run(self):
        from ob_analytics import Pipeline
        from ob_analytics.databento import DatabentoSource
        from tests.test_databento import LIFECYCLE, flag_one_record, mbo_frame

        frame = flag_one_record(mbo_frame(LIFECYCLE), 5)
        with pytest.warns(UserWarning, match="F_MAYBE_BAD_BOOK"):
            return Pipeline(source=DatabentoSource()).run(frame)

    def test_a_saved_run_records_them_in_its_own_metadata(self, tmp_path: Path):
        import json

        import pyarrow.parquet as pq

        from ob_analytics.schemas import LOADER_COUNTS_KEY

        save_data(self._flagged_run(), tmp_path)

        metadata = pq.read_schema(tmp_path / "events.parquet").metadata
        assert json.loads(metadata[LOADER_COUNTS_KEY]) == {"maybe_bad_book_records": 1}
        # Stored once: not again in pandas' own metadata.
        assert b"maybe_bad_book_records" not in metadata[b"pandas"]

    def test_the_audit_of_a_saved_run_makes_the_check(self, tmp_path: Path):
        from ob_analytics.data import load_result

        save_data(self._flagged_run(), tmp_path)
        summary = load_result(tmp_path).audit()

        assert summary.maybe_bad_book_records == 1
        assert [check.name for check in summary.warnings] == ["maybe_bad_book"]

    def test_the_arrow_tables_record_them_too(self):
        import json

        from ob_analytics.schemas import LOADER_COUNTS_KEY

        tables = self._flagged_run().to_arrow()

        counts = json.loads(tables["events"].schema.metadata[LOADER_COUNTS_KEY])
        assert counts == {"maybe_bad_book_records": 1}

    def test_a_file_from_an_older_version_still_loads(self, tmp_path: Path):
        # Older files kept the count only in pandas' own metadata, and most
        # files have no count at all.
        import pyarrow.parquet as pq

        counted = pd.DataFrame({"x": [1]})
        counted.attrs["maybe_bad_book_records"] = 2
        pq.write_table(pa.Table.from_pandas(counted), tmp_path / "counted.parquet")
        pq.write_table(
            pa.Table.from_pandas(pd.DataFrame({"x": [1]})), tmp_path / "plain.parquet"
        )

        with pytest.warns(UserWarning, match="schema version"):
            loaded = load_data(tmp_path)

        assert loaded["counted"].attrs["maybe_bad_book_records"] == 2
        assert "maybe_bad_book_records" not in loaded["plain"].attrs


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

        assert_same_result(back, result)
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

        assert_same_result(load_result(tmp_path / "saved"), result)

    def test_an_l2_run_comes_back_at_l2(self, tmp_path: Path):
        from ob_analytics import Pipeline
        from ob_analytics.data import load_result
        from ob_analytics.datasets import toy_l2_depth, toy_l2_trades
        from ob_analytics.depth_csv import DepthCsvWriter
        from ob_analytics.protocols import Level

        DepthCsvWriter().write(
            {"depth": toy_l2_depth(), "trades": toy_l2_trades()}, tmp_path / "l2"
        )
        result = Pipeline.from_source("depth_csv").run(tmp_path / "l2")
        save_data(result, tmp_path / "saved")
        back = load_result(tmp_path / "saved")

        assert back.level is Level.L2
        assert_same_result(back, result)

    @staticmethod
    def _without_level(folder: Path) -> None:
        """Strip the recorded level, as a folder from an older version has."""
        import pyarrow.parquet as pq

        from ob_analytics.schemas import LEVEL_KEY

        for file in folder.glob("*.parquet"):
            table = pq.read_table(file)
            metadata = dict(table.schema.metadata)
            del metadata[LEVEL_KEY]
            pq.write_table(table.replace_schema_metadata(metadata), file)

    def test_a_per_order_run_with_no_events_comes_back_at_l3(
        self, tmp_path: Path, lobster_day_dir, lobster_day_pipeline
    ):
        """The saved level wins over the empty events table (#510 item 2)."""
        from dataclasses import replace

        from ob_analytics.data import load_result
        from ob_analytics.protocols import Level

        result = lobster_day_pipeline.run(lobster_day_dir)
        assert result.level is Level.L3
        result = replace(result, events=result.events.iloc[0:0])
        save_data(result, tmp_path / "saved")

        assert load_result(tmp_path / "saved").level is Level.L3

    def test_a_folder_without_a_level_takes_it_from_the_events(
        self, tmp_path: Path, lobster_day_dir, lobster_day_pipeline
    ):
        """A folder from an older version still loads, by the old rule."""
        from dataclasses import replace

        from ob_analytics.data import load_result
        from ob_analytics.protocols import Level

        result = lobster_day_pipeline.run(lobster_day_dir)
        save_data(result, tmp_path / "full")
        self._without_level(tmp_path / "full")
        back = load_result(tmp_path / "full")
        assert back.level is Level.L3
        assert_same_result(back, result)

        save_data(replace(result, events=result.events.iloc[0:0]), tmp_path / "empty")
        self._without_level(tmp_path / "empty")
        assert load_result(tmp_path / "empty").level is Level.L2

    def test_a_level_this_version_cannot_read_falls_back_to_the_events(
        self, tmp_path: Path, lobster_day_dir, lobster_day_pipeline
    ):
        import pyarrow.parquet as pq

        from ob_analytics.data import load_result
        from ob_analytics.protocols import Level
        from ob_analytics.schemas import LEVEL_KEY

        save_data(lobster_day_pipeline.run(lobster_day_dir), tmp_path / "saved")
        for file in (tmp_path / "saved").glob("*.parquet"):
            table = pq.read_table(file)
            metadata = {**table.schema.metadata, LEVEL_KEY: b"L9"}
            pq.write_table(table.replace_schema_metadata(metadata), file)

        with pytest.warns(UserWarning, match="level"):
            back = load_result(tmp_path / "saved")
        assert back.level is Level.L3

    def test_a_file_with_only_the_tick_and_lot_takes_decimals_from_them(
        self, tmp_path: Path, lobster_day_dir, lobster_day_pipeline
    ):
        """A windowed run's folder records the tick and lot size only."""
        from ob_analytics.data import load_result

        lobster_day_pipeline.run_windows(
            lobster_day_dir, "2024-01-02 13:30:02", tmp_path / "saved"
        )
        from ob_analytics.protocols import Level

        back = load_result(tmp_path / "saved")

        assert back.level is Level.L3
        assert back.config.tick_size == 0.01
        assert back.config.price_decimals == 2
        assert back.config.lot_size == 1.0
        assert back.config.volume_decimals == 0

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
        save_data(result.tables(), tmp_path / "saved")
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
        with pytest.raises(UnknownNameError, match="Unknown format"):
            save_data({}, tmp_path / "out", fmt="csv")

    def test_load_data_refuses_a_file(self, tmp_path: Path):
        """``load_data`` reads a Parquet folder; a pickle or any file is refused."""
        from ob_analytics.exceptions import ConfigError

        for name in ("data.json", "data.pkl"):
            f = tmp_path / name
            f.write_text("{}")
            with pytest.raises(ConfigError, match="not a Parquet folder"):
                load_data(f)
