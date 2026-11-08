"""Coverage for the writer registry in ob_analytics.data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import pytest

from ob_analytics.config import PipelineConfig
from ob_analytics.data import (
    WRITERS,
    list_writers,
    load_data,
    register_writer,
    save_data,
)
from ob_analytics.exceptions import ConfigError
from ob_analytics.protocols import WriteTarget
from ob_analytics.schemas import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_KEY,
    TICK_SIZE_KEY,
    decode_tick_sizes,
)

# ---------------------------------------------------------------------------
# A stub writer for tests
# ---------------------------------------------------------------------------


class _StubWriter:
    """Minimal DataWriter — records what it was asked to write."""

    written: ClassVar[list[tuple[Path, list[str]]]] = []
    target = WriteTarget.FOLDER

    def write(self, data: dict, dest: str | Path, **kwargs) -> Path:
        p = Path(dest)
        self.written.append((p, sorted(data.keys())))
        return p


@pytest.fixture(autouse=True)
def _cleanup_registry():
    """Snapshot and restore the global writer registry around each test.

    ``Registry`` exposes no public clear/restore (by design — it is a
    write-once-at-import surface), so we snapshot its backing map.
    """
    before = dict(WRITERS._items)
    yield
    WRITERS._items.clear()
    WRITERS._items.update(before)
    _StubWriter.written.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWriterRegistry:
    def test_venue_writers_come_from_the_source_not_the_generic_registry(self):
        # Issue #137 folds venue writers onto their source's create_writer, so
        # the generic writer registry does not list them a second time.
        assert "bitstamp" not in WRITERS
        assert "lobster" not in WRITERS

    def test_list_writers_lists_every_format_save_data_accepts(self):
        # Issue #409: the source writers are formats too, and the live-only
        # sources, which write nothing, are not.
        assert list_writers() == [
            "bitstamp",
            "databento",
            "depth_csv",
            "hftbacktest",
            "lobster",
            "nautilus",
            "parquet",
        ]

    def test_the_unknown_format_error_offers_only_list_writers(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported format") as info:
            save_data({"events": pd.DataFrame()}, tmp_path / "x", fmt="nope")
        offered = str(info.value).split("Available: ")[1].split(", ")
        assert offered == list_writers()
        assert "ccxt" not in offered
        assert "cryptofeed" not in offered

    def test_source_name_resolves_to_its_writer(self):
        # save_data(fmt="<venue>") reaches the writer through the source.
        from ob_analytics.bitstamp import BitstampWriter
        from ob_analytics.data import _named_writer

        assert isinstance(_named_writer("bitstamp", None, None), BitstampWriter)

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


class TestBuiltInFormatsAreRegisteredWriters:
    """Issue #216: the built-in parquet format resolves through ``WRITERS``.

    It was a hardcoded branch in ``save_data``, so it was the only
    extension point in the library a user could not replace or extend.
    """

    def test_a_registered_parquet_writer_replaces_the_built_in_one(self, tmp_path):
        register_writer("parquet", lambda config, ctx: _StubWriter())

        save_data({"events": pd.DataFrame({"a": [1]})}, tmp_path / "out", fmt="parquet")

        assert _StubWriter.written == [(tmp_path / "out", ["events"])]
        assert not (tmp_path / "out" / "events.parquet").exists()

    def test_a_writer_can_ask_for_canonical_arrow_tables(self, tmp_path):
        # A writer that targets Arrow (#113's Nautilus export is the first)
        # must not have to rebuild the canonical metadata by hand.
        seen: dict[str, Any] = {}

        class _ArrowWantingWriter:
            target = WriteTarget.FOLDER

            def write(self, data, dest, **kwargs):
                seen["frames"] = dict(data)
                seen["tables"] = data.arrow()
                return Path(dest)

        register_writer("arrow-wanting", lambda config, ctx: _ArrowWantingWriter())

        save_data(
            {"events": pd.DataFrame({"price": [1, 2]})},
            tmp_path / "out",
            fmt="arrow-wanting",
            config=PipelineConfig(tick_size=0.05),
        )

        # Still a plain mapping of pandas frames, so existing writers are unaffected.
        assert isinstance(seen["frames"]["events"], pd.DataFrame)

        tables = seen["tables"]
        metadata = tables["events"].schema.metadata
        assert metadata[SCHEMA_VERSION_KEY] == SCHEMA_VERSION.encode()
        assert decode_tick_sizes(metadata[TICK_SIZE_KEY]) == {"default": 0.05}

    def test_pickle_is_not_a_format(self, tmp_path):
        # Issue #409: a pickle runs code when it is loaded, so it is gone.
        assert "pickle" not in list_writers()
        with pytest.raises(ValueError, match="Unsupported format"):
            save_data({"events": pd.DataFrame()}, tmp_path / "out.pkl", fmt="pickle")


class _FileStubWriter(_StubWriter):
    target = WriteTarget.FILE


class TestSaveDataChecksThePath:
    """Issue #409: each writer declares a folder or a file, and save_data checks."""

    @pytest.mark.parametrize("name", ["run.parquet", "orders.csv", "run.pkl"])
    def test_a_folder_format_refuses_a_file_name(self, tmp_path, name):
        with pytest.raises(ConfigError, match="'parquet' writes a folder"):
            save_data({"events": pd.DataFrame({"a": [1]})}, tmp_path / name)
        assert not (tmp_path / name).exists()

    def test_a_folder_format_refuses_an_existing_file(self, tmp_path):
        existing = tmp_path / "taken"
        existing.write_text("")
        with pytest.raises(ConfigError, match="an existing file"):
            save_data({"events": pd.DataFrame({"a": [1]})}, existing)

    def test_a_folder_name_with_a_dot_is_a_folder(self, tmp_path):
        save_data({"events": pd.DataFrame({"a": [1]})}, tmp_path / "btc_2024.05.01")
        assert (tmp_path / "btc_2024.05.01" / "events.parquet").is_file()

    def test_a_file_format_refuses_an_existing_folder(self, tmp_path):
        register_writer("filestub", lambda config, ctx: _FileStubWriter())
        with pytest.raises(ConfigError, match="'filestub' writes one file"):
            save_data({"events": pd.DataFrame()}, tmp_path, fmt="filestub")
        assert not _StubWriter.written

    def test_a_file_format_refuses_a_path_with_no_extension(self, tmp_path):
        register_writer("filestub", lambda config, ctx: _FileStubWriter())
        with pytest.raises(ConfigError, match="no file extension"):
            save_data({"events": pd.DataFrame()}, tmp_path / "out", fmt="filestub")

    def test_a_writer_with_no_target_is_refused(self, tmp_path):
        class _NoTarget:
            def write(self, data, dest, **kwargs):
                return Path(dest)

        with pytest.raises(ConfigError, match="_NoTarget declares no target"):
            save_data(
                {"events": pd.DataFrame()},
                tmp_path / "x",
                writer=_NoTarget(),  # ty: ignore[invalid-argument-type] -- the broken writer under test
            )

    def test_a_passed_writer_is_checked_too(self, tmp_path):
        with pytest.raises(ConfigError, match="_FileStubWriter writes one file"):
            save_data({"events": pd.DataFrame()}, tmp_path, writer=_FileStubWriter())

    @pytest.mark.parametrize(
        ("fmt", "target"),
        [
            ("parquet", WriteTarget.FOLDER),
            ("depth_csv", WriteTarget.FOLDER),
            ("bitstamp", WriteTarget.FOLDER),
            ("databento", WriteTarget.FILE),
            ("hftbacktest", WriteTarget.FILE),
            ("nautilus", WriteTarget.FILE),
        ],
    )
    def test_each_built_in_format_declares_its_target(self, fmt, target):
        from ob_analytics.data import _named_writer

        assert _named_writer(fmt, None, None).target is target

    def test_the_lobster_writer_writes_a_folder(self):
        from ob_analytics.lobster import LobsterWriter

        assert LobsterWriter.target is WriteTarget.FOLDER
