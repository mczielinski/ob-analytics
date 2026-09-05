"""Coverage for the writer registry in ob_analytics.data."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

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
        # the generic writer registry no longer double-lists them.
        names = list_writers()
        assert "bitstamp" not in names
        assert "lobster" not in names

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
    """Issue #216: parquet and pickle resolve through ``WRITERS`` like the rest.

    They were hardcoded branches in ``save_data``, so they were the only
    extension point in the library a user could not replace or extend.
    """

    def test_a_registered_parquet_writer_replaces_the_built_in_one(self, tmp_path):
        register_writer("parquet", lambda config, ctx: _StubWriter())

        save_data({"events": pd.DataFrame({"a": [1]})}, tmp_path / "out", fmt="parquet")

        assert _StubWriter.written == [(tmp_path / "out", ["events"])]
        assert not (tmp_path / "out" / "events.parquet").exists()

    def test_a_registered_pickle_writer_replaces_the_built_in_one(self, tmp_path):
        register_writer("pickle", lambda config, ctx: _StubWriter())

        save_data(
            {"events": pd.DataFrame({"a": [1]})}, tmp_path / "out.pkl", fmt="pickle"
        )

        assert _StubWriter.written == [(tmp_path / "out.pkl", ["events"])]
        assert not (tmp_path / "out.pkl").exists()

    def test_a_writer_can_ask_for_canonical_arrow_tables(self, tmp_path):
        # A writer that targets Arrow (#113's Nautilus export is the first)
        # must not have to rebuild the canonical metadata by hand.
        seen: dict[str, object] = {}

        class _ArrowWantingWriter:
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

    def test_a_saved_pickle_holds_a_plain_dict(self, tmp_path):
        # The payload handed to a writer is a dict subclass (#216).  Pickling it
        # whole would write ob-analytics' own class into the file, so the file
        # would only load where that class exists.
        pkl = tmp_path / "out.pkl"
        save_data({"events": pd.DataFrame({"a": [1]})}, pkl, fmt="pickle")

        assert type(load_data(pkl)) is dict
        assert b"OutputTables" not in pkl.read_bytes()
