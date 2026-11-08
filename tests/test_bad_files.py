"""Every reader refuses a malformed file with a package error.

A reader handed a bad file must say so, not return a frame built from it.
Two bugs of that kind reached users before: an empty price read as -9.2e18
ticks, and a segment named ``..`` in a capture's ``manifest.json`` led
outside the capture's folder.  The property tests in
:mod:`test_properties_schema` cover the engine; these cover the readers.

Each test starts from a small valid file, which the reader reads, and breaks
it in one way that leaves no doubt the file is wrong:

* a required cell is empty, or holds text no reader can take for a number,
  a time, a side or an action;
* a required column is missing;
* the file is not text at all;
* the file ends part way through a row (or, for DBN, a record).

The reader must raise :class:`~ob_analytics.exceptions.ObAnalyticsError`,
the base of every package error, so ``except ObAnalyticsError`` is enough to
catch a bad file.
"""

from __future__ import annotations

import json
import shutil
import string
import tempfile
import warnings
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from ob_analytics.bitstamp import BitstampLoader, BitstampTradeReader
from ob_analytics.data import load_data, save_data
from ob_analytics.datasets import toy_l2_depth, toy_l2_trades
from ob_analytics.depth_l2 import L2DepthLoader, L2TradeReader
from ob_analytics.exceptions import ObAnalyticsError
from ob_analytics.live._manifest import (
    CAPTURE_FORMAT,
    CAPTURE_FORMAT_VERSION,
    MANIFEST_NAME,
    read_manifest,
)
from ob_analytics.lobster import LobsterLoader, LobsterSource
from ob_analytics.pipeline import Pipeline
from ob_analytics.protocols import RunContext
from tests.conftest import LOBSTER_DAY_DATE, write_lobster_day, write_tiny_bitstamp

_SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

#: Text that is never a number, a time, a side or an action: it starts with a
#: character none of those can start with, and has no comma to shift a column.
_junk = st.builds(
    lambda tail: "?" + tail, st.text(alphabet=string.ascii_letters, max_size=6)
)
_bad_cell = st.one_of(st.just(""), _junk)


@contextmanager
def _folder() -> Generator[Path]:
    """A fresh folder for one Hypothesis example."""
    with tempfile.TemporaryDirectory() as name:
        yield Path(name)


def _refuses(read: Callable[[Path], Any], folder: Path) -> None:
    """Assert *read* raises a package error on *folder*, and returns nothing."""
    with warnings.catch_warnings():
        # A reader may warn on its way to raising; the raise is what counts.
        warnings.simplefilter("ignore")
        with pytest.raises(ObAnalyticsError):
            read(folder)


# ── CSV readers ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class CsvReader:
    """A reader of one CSV file, and a valid file for it.

    ``required`` are the columns (by name, or by position for a file with no
    header) every row must fill with a valid value.  Of those, ``ids`` are
    order ids, which may be text (some venues use UUIDs), so only an empty
    one is wrong.  ``optional`` columns may be empty, but not hold junk.
    """

    name: str
    write_valid: Callable[[Path], object]
    file: str
    read: Callable[[Path], Any]
    required: tuple[str | int, ...]
    ids: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    header: bool = True


def _bitstamp_trades(folder: Path) -> pd.DataFrame:
    events = BitstampLoader().load(folder / "orders.csv")
    return BitstampTradeReader().load(events, folder)


def _toy_l2(folder: Path) -> None:
    save_data(
        {"depth": toy_l2_depth(), "trades": toy_l2_trades()}, folder, fmt="depth_csv"
    )


_LOBSTER_STEM = f"TEST_{LOBSTER_DAY_DATE}_34200000_57600000"
_LOBSTER_MESSAGES = f"{_LOBSTER_STEM}_message_1.csv"
_LOBSTER_ORDERBOOK = f"{_LOBSTER_STEM}_orderbook_1.csv"

CSV_READERS = [
    CsvReader(
        "bitstamp orders",
        write_tiny_bitstamp,
        "orders.csv",
        lambda folder: BitstampLoader().load(folder / "orders.csv"),
        required=("id", "timestamp", "price", "volume", "action", "direction"),
        ids=("id",),
        optional=("exchange_timestamp",),
    ),
    CsvReader(
        "bitstamp trades",
        write_tiny_bitstamp,
        "trades.csv",
        _bitstamp_trades,
        required=("timestamp", "price", "amount"),
        optional=("side",),
    ),
    CsvReader(
        "lobster messages",
        write_lobster_day,
        _LOBSTER_MESSAGES,
        lambda folder: LobsterLoader(trading_date=LOBSTER_DAY_DATE).load(folder),
        # time, event type, order id, size, price, direction
        required=(0, 1, 2, 3, 4, 5),
        header=False,
    ),
    CsvReader(
        "lobster orderbook",
        write_lobster_day,
        _LOBSTER_ORDERBOOK,
        # The orderbook file is read when the run builds its depth.
        lambda folder: Pipeline(
            source=LobsterSource(), ctx=RunContext(trading_date=LOBSTER_DAY_DATE)
        ).run(folder),
        # ask price, ask size, bid price, bid size
        required=(0, 1, 2, 3),
        header=False,
    ),
    CsvReader(
        "depth_csv depth",
        _toy_l2,
        "depth.csv",
        lambda folder: L2DepthLoader().load(folder),
        required=("timestamp", "side", "price", "volume"),
    ),
    CsvReader(
        "depth_csv trades",
        _toy_l2,
        "trades.csv",
        lambda folder: L2TradeReader().load(pd.DataFrame(), folder),
        required=("timestamp", "price", "amount"),
        optional=("side",),
    ),
]


@pytest.fixture(scope="module", params=CSV_READERS, ids=[r.name for r in CSV_READERS])
def csv_reader(request, tmp_path_factory) -> tuple[CsvReader, Path]:
    """A reader and a folder holding a valid file for it, which it reads."""
    reader: CsvReader = request.param
    valid = tmp_path_factory.mktemp(reader.name.replace(" ", "_"))
    reader.write_valid(valid)
    reader.read(valid)
    return reader, valid


@contextmanager
def _broken(valid: Path, file: str, text: str | bytes) -> Generator[Path]:
    """A copy of *valid* whose *file* holds *text* instead."""
    with _folder() as folder:
        shutil.copytree(valid, folder, dirs_exist_ok=True)
        if isinstance(text, bytes):
            (folder / file).write_bytes(text)
        else:
            (folder / file).write_text(text)
        yield folder


def _rows(valid: Path, reader: CsvReader) -> tuple[list[str], list[list[str]]]:
    """The header (empty when there is none) and the data rows, as cells."""
    lines = (valid / reader.file).read_text().splitlines()
    header = lines[0].split(",") if reader.header else []
    return header, [line.split(",") for line in lines[len(header) > 0 :]]


def _position(header: list[str], column: str | int) -> int:
    return column if isinstance(column, int) else header.index(column)


def _text(header: list[str], rows: list[list[str]]) -> str:
    lines = ([header] if header else []) + rows
    return "".join(",".join(cells) + "\n" for cells in lines)


@_SETTINGS
@given(data=st.data())
def test_a_bad_cell_is_refused(csv_reader, data) -> None:
    reader, valid = csv_reader
    header, rows = _rows(valid, reader)
    row = data.draw(st.integers(0, len(rows) - 1), label="row")
    columns = reader.required + reader.optional
    column = data.draw(st.sampled_from(columns), label="column")
    if column in reader.ids:
        value = ""
    elif column in reader.optional:
        value = data.draw(_junk, label="value")
    else:
        value = data.draw(_bad_cell, label="value")
    rows[row][_position(header, column)] = value
    with _broken(valid, reader.file, _text(header, rows)) as folder:
        _refuses(reader.read, folder)


@_SETTINGS
@given(data=st.data())
def test_a_missing_column_is_refused(csv_reader, data) -> None:
    reader, valid = csv_reader
    header, rows = _rows(valid, reader)
    at = _position(header, data.draw(st.sampled_from(reader.required)))
    header = [c for i, c in enumerate(header) if i != at]
    rows = [[c for i, c in enumerate(cells) if i != at] for cells in rows]
    with _broken(valid, reader.file, _text(header, rows)) as folder:
        _refuses(reader.read, folder)


@_SETTINGS
@given(noise=st.binary(min_size=1, max_size=64))
def test_a_file_that_is_not_text_is_refused(csv_reader, noise: bytes) -> None:
    reader, valid = csv_reader
    # 0xff never appears in UTF-8, so the file cannot be read as text.
    with _broken(valid, reader.file, b"\xff" + noise) as folder:
        _refuses(reader.read, folder)


@_SETTINGS
@given(data=st.data())
def test_a_file_cut_inside_a_row_is_refused(csv_reader, data) -> None:
    reader, valid = csv_reader
    header, rows = _rows(valid, reader)
    row = data.draw(st.integers(0, len(rows) - 1), label="row")
    # Keep at least one cell, and lose a required one.  A row that loses only
    # optional cells reads the same as one whose optional cells are empty,
    # which is a valid row.
    last = max(_position(header, column) for column in reader.required)
    keep = data.draw(st.integers(1, min(last, len(rows[row]) - 1)), label="cells kept")
    text = _text(header, rows[:row]) + ",".join(rows[row][:keep])
    with _broken(valid, reader.file, text) as folder:
        _refuses(reader.read, folder)


# ── DBN ───────────────────────────────────────────────────────────────


#: The size of one DBN MBO record, in bytes.
_MBO_RECORD = 56


@pytest.fixture(scope="module")
def dbn_file(tmp_path_factory) -> tuple[bytes, int]:
    """A valid DBN file's bytes, and where its first record starts."""
    pytest.importorskip("databento", reason="databento is an optional extra")
    from ob_analytics.databento import DatabentoSource, read_mbo_frame
    from ob_analytics.pipeline import Pipeline
    from tests._mbo import mbo_frame

    run = Pipeline(source=DatabentoSource()).run(mbo_frame())
    path = tmp_path_factory.mktemp("dbn") / "valid.dbn"
    save_data(run, path, fmt="databento")
    _dbn_read(path.parent, path.name)  # the reader takes the valid file
    raw = path.read_bytes()
    # The records follow a metadata header of any length.
    return raw, len(raw) - _MBO_RECORD * len(read_mbo_frame(path))


def _dbn_read(folder: Path, name: str = "bad.dbn") -> pd.DataFrame:
    from ob_analytics.databento import DatabentoLoader

    return DatabentoLoader().load(folder / name)


@_SETTINGS
@given(data=st.data())
def test_a_dbn_file_cut_inside_a_record_is_refused(dbn_file, data) -> None:
    raw, start = dbn_file
    cut = data.draw(
        st.integers(1, len(raw) - 1).filter(
            lambda n: n < start or (n - start) % _MBO_RECORD
        ),
        label="bytes kept",
    )
    with _folder() as folder:
        (folder / "bad.dbn").write_bytes(raw[:cut])
        _refuses(_dbn_read, folder)


@_SETTINGS
@given(noise=st.binary(min_size=0, max_size=64))
def test_a_dbn_file_that_is_not_dbn_is_refused(dbn_file, noise: bytes) -> None:
    # dbn_file is asked for only so the test skips without the databento extra.
    with _folder() as folder:
        (folder / "bad.dbn").write_bytes(b"\xff" + noise)
        _refuses(_dbn_read, folder)


# ── Parquet ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def parquet_folder(tmp_path_factory) -> Path:
    """A folder ``save_data`` wrote, which ``load_data`` reads."""
    folder = tmp_path_factory.mktemp("parquet")
    save_data({"depth": toy_l2_depth(), "trades": toy_l2_trades()}, folder)
    load_data(folder)
    return folder


@_SETTINGS
@given(data=st.data())
def test_a_cut_parquet_file_is_refused(parquet_folder, data) -> None:
    raw = (parquet_folder / "depth.parquet").read_bytes()
    cut = data.draw(st.integers(0, len(raw) - 1), label="bytes kept")
    with _broken(parquet_folder, "depth.parquet", raw[:cut]) as folder:
        _refuses(load_data, folder)


# ── manifest.json ─────────────────────────────────────────────────────


def _manifest() -> dict[str, Any]:
    """A valid manifest of one closed segment."""
    return {
        "format": CAPTURE_FORMAT,
        "version": CAPTURE_FORMAT_VERSION,
        "source": "bitstamp",
        "pair": "btcusd",
        "level": "L3",
        "started": "2026-01-01T00:00:00+00:00",
        "segments": [
            {
                "name": "seg-0001",
                "started": "2026-01-01T00:00:00+00:00",
                "ended": "2026-01-01T00:10:00+00:00",
                "end_reason": "finished",
                "n_book_events": 10,
            }
        ],
        "gaps": [],
    }


def test_the_valid_manifest_is_read(tmp_path) -> None:
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(_manifest()))
    manifest = read_manifest(tmp_path)
    assert manifest is not None
    assert [s.name for s in manifest.segments] == ["seg-0001"]


#: A segment name that would lead a reader out of the capture's folder, or
#: is no directory name at all.
_unsafe_names = st.one_of(
    st.sampled_from(["", ".", "..", "../x", "a/b", "/abs", "a\\b", "C:x", "a\0b"]),
    st.builds("{}/{}".format, _junk, _junk),
)

#: ``(path, bad value)`` edits to the valid manifest.  A path names a key, or
#: a key of the first segment (``("segments", 0, key)``).
_bad_edits = st.one_of(
    st.tuples(st.just(("segments", 0, "name")), _unsafe_names),
    st.tuples(
        st.sampled_from(
            [("started",), ("segments", 0, "started"), ("segments", 0, "ended")]
        ),
        _junk,
    ),
    st.tuples(st.just(("segments", 0, "end_reason")), _junk),
    st.tuples(
        st.just(("segments", 0, "n_book_events")),
        st.one_of(_junk, st.lists(st.integers(), max_size=2)),
    ),
    st.tuples(
        st.sampled_from([("source",), ("pair",), ("level",)]),
        st.one_of(st.none(), st.integers(), st.lists(st.integers(), max_size=2)),
    ),
    st.tuples(
        st.sampled_from([("segments",), ("gaps",)]),
        st.one_of(
            _junk, st.integers(), st.dictionaries(_junk, st.integers(), max_size=2)
        ),
    ),
)


@_SETTINGS
@given(edit=_bad_edits)
def test_a_bad_manifest_value_is_refused(edit) -> None:
    path, value = edit
    manifest = _manifest()
    target: Any = manifest
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with _folder() as folder:
        (folder / MANIFEST_NAME).write_text(json.dumps(manifest))
        _refuses(read_manifest, folder)


@_SETTINGS
@given(key=st.sampled_from(["format", "version", "source", "pair", "level", "started"]))
def test_a_manifest_missing_a_key_is_refused(key: str) -> None:
    manifest = _manifest()
    del manifest[key]
    with _folder() as folder:
        (folder / MANIFEST_NAME).write_text(json.dumps(manifest))
        _refuses(read_manifest, folder)


@_SETTINGS
@given(
    text=st.one_of(
        st.builds(lambda s: "{" + s, st.text(max_size=20)),
        st.just("[]"),
        st.just("null"),
        st.just('"manifest"'),
    )
)
def test_a_manifest_that_is_not_a_json_object_is_refused(text: str) -> None:
    with _folder() as folder:
        (folder / MANIFEST_NAME).write_text(text)
        _refuses(read_manifest, folder)
