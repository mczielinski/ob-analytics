"""Read a saved run in parts, and aggregate it without loading it whole.

:func:`~ob_analytics.data.load_data` reads every table of a saved run into
memory.  The functions here read the same folder one batch of rows at a time,
so the memory they use is set by the batch size and the size of the result, not
by the size of the file.  A file larger than the machine's memory can be read
and aggregated this way.

Each function reads one table of the folder that
:func:`~ob_analytics.data.save_data` or
:meth:`~ob_analytics.pipeline.Pipeline.run_windows` writes, through
``pyarrow.dataset``.  The columns you name and the time range you give are
passed down to the Parquet reader, so it reads only those columns, and skips
each row group whose times are all outside the range.

* :func:`scan_table` gives the rows as pandas frames, one batch at a time.
* :func:`read_table` gives the selected rows as one frame.
* :func:`aggregate_by_time` reduces a table to one row per fixed time bucket.
* :func:`summarize_depth_file` builds a new ``depth_summary`` from a saved
  ``depth`` table, for example with other ``depth_bps`` and ``depth_bins``.

Each frame comes out as :func:`~ob_analytics.data.load_data` gives it: the
same dtypes, and the tick size and lot size on ``attrs``.  The numbers are the
same as an in-memory run gives on the same rows.

Usage::

    from ob_analytics.scan import aggregate_by_time, read_table

    # The best bid and ask at the end of each minute, read in parts.
    quotes = aggregate_by_time(
        "out/", "depth_summary", "1min",
        {"best_bid_price": "last", "best_ask_price": "last"},
    )

    # One hour of the depth summary, and only the columns asked for.
    hour = read_table(
        "out/", "depth_summary",
        columns=["timestamp", "best_bid_price", "best_ask_price"],
        start="2015-05-01 01:00", end="2015-05-01 02:00",
    )
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from ob_analytics._utils import window_start
from ob_analytics._windows import ParquetAppender, _as_utc
from ob_analytics.config import PipelineConfig
from ob_analytics.data import _canonical_frame, _checked_version, _saved_config
from ob_analytics.depth import DepthSummaryEngine
from ob_analytics.exceptions import ConfigError
from ob_analytics.protocols import FeedDeclarations, Level
from ob_analytics.schemas import (
    DECLARATIONS_KEY,
    LEVEL_KEY,
    TIMESTAMP_COLUMN,
    UNIT_ATTRS,
    time_order_keys,
)

__all__ = [
    "AGGREGATIONS",
    "BATCH_ROWS",
    "aggregate_by_time",
    "read_table",
    "scan_table",
    "summarize_depth_file",
]

#: The default number of rows in one batch.  A ``depth_summary`` batch of this
#: size takes about 50 MB as a pandas frame.
BATCH_ROWS = 131_072

#: The reductions :func:`aggregate_by_time` can apply to a column.  Each one
#: gives the same number whether it runs over a whole bucket at once or over
#: the parts of a bucket that different batches hold.  Like pandas, every one
#: but ``count`` skips empty values.
AGGREGATIONS = ("first", "last", "min", "max", "sum", "count", "mean")

# The partial results each reduction keeps per batch, and how two partial
# results of one bucket combine.  A mean keeps a sum and a count.
_PARTS: dict[str, tuple[str, ...]] = {
    "first": ("first",),
    "last": ("last",),
    "min": ("min",),
    "max": ("max",),
    "sum": ("sum",),
    "count": ("count",),
    "mean": ("sum", "count"),
}
_COMBINE = {"first": "first", "last": "last", "min": "min", "max": "max",
            "sum": "sum", "count": "sum"}  # fmt: skip

# Combine the partial results after this many batches, so they stay about the
# size of the result however many batches the file has.
_COMBINE_EVERY = 64


# ── Reading ───────────────────────────────────────────────────────────

# Parquet read with pre-buffering off.  Pre-buffering fetches the bytes of the
# coming row groups ahead of the reader and keeps them, so its memory grew with
# the part of the file read so far: about 0.7 GiB after 20M rows of a depth
# summary, against about 0.25 GiB, level, without it.  The files are local, so
# fetching ahead saves nothing.
_PARQUET = ds.ParquetFileFormat(
    default_fragment_scan_options=ds.ParquetFragmentScanOptions(pre_buffer=False)
)


class _SavedTable:
    """One table of a saved run: its file, its metadata and its version."""

    def __init__(self, folder: str | Path, table: str) -> None:
        self.folder = Path(folder)
        self.name = table
        self.path = self.folder / f"{table}.parquet"
        if not self.folder.is_dir():
            raise ConfigError(
                f"{self.folder} is not a folder. The scan functions read the "
                "folder save_data or run_windows writes."
            )
        if not self.path.is_file():
            found = sorted(p.stem for p in self.folder.glob("*.parquet"))
            raise ConfigError(
                f"{self.folder} has no {table!r} table; it has {found or 'none'}."
            )
        try:
            self.schema = pq.read_schema(self.path)
        except pa.ArrowException as exc:
            raise ConfigError(
                f"{self.path} is not a Parquet file this build can read: {exc}"
            ) from exc
        self.metadata: dict[bytes, bytes] = dict(self.schema.metadata or {})
        # The user's call: check_schema_version, _checked_version, this
        # constructor, the public function, then the user.
        self.version = _checked_version(self.metadata, self.path, stacklevel=5)

    def batches(
        self,
        columns: Sequence[str] | None,
        start: Any,
        end: Any,
        filter: ds.Expression | None,
        batch_rows: int,
    ) -> Iterator[pd.DataFrame]:
        """The selected rows as canonical frames, in the file's order.

        The arguments are checked here, at the call, and the rows are read
        only as the iterator is consumed.
        """
        if batch_rows < 1:
            raise ConfigError(f"batch_rows must be at least 1; got {batch_rows}.")
        selected, expression = self._selection(columns, start, end, filter)
        return self._read(selected, expression, batch_rows)

    def whole(
        self,
        columns: Sequence[str] | None,
        start: Any,
        end: Any,
        filter: ds.Expression | None,
    ) -> pd.DataFrame:
        """The selected rows as one canonical frame, in the file's order."""
        selected, expression = self._selection(columns, start, end, filter)
        table = (
            ds.dataset(self.path, format=_PARQUET)
            .to_table(columns=selected, filter=expression)
            .replace_schema_metadata(self.metadata)
        )
        # The only reference, so each column's Arrow memory goes as soon as it
        # is converted and the result is never held twice.
        return _canonical_frame(
            table, self.name, self.metadata, self.version, release=True
        )

    def _selection(
        self,
        columns: Sequence[str] | None,
        start: Any,
        end: Any,
        filter: ds.Expression | None,
    ) -> tuple[list[str] | None, ds.Expression | None]:
        """The checked columns and the filter the reader is given."""
        if columns is not None:
            missing = [c for c in columns if c not in self.schema.names]
            if missing:
                raise ConfigError(
                    f"The {self.name!r} table has no column {missing}; it has "
                    f"{self.schema.names}."
                )
        expression = self._time_filter(start, end)
        if filter is not None:
            expression = filter if expression is None else expression & filter
        return (list(columns) if columns is not None else None), expression

    def _read(
        self,
        columns: list[str] | None,
        expression: ds.Expression | None,
        batch_rows: int,
    ) -> Iterator[pd.DataFrame]:
        dataset = ds.dataset(self.path, format=_PARQUET)
        # The scanner gives the batches in the file's order, so a rolling
        # computation over them sees the rows as a single read would.  With
        # threads, or with its default read-ahead, the scanner decodes row
        # groups before they are asked for, as fast as it can, so a consumer
        # slower than the reader (any aggregation) lets them pile up: on a
        # 5M-row depth summary Arrow held about 1.3 GiB with threads and about
        # 0.3 GiB without, at the same speed.  See _PARQUET for the rest.
        for batch in dataset.to_batches(
            columns=columns,
            filter=expression,
            batch_size=batch_rows,
            batch_readahead=0,
            fragment_readahead=0,
            use_threads=False,
        ):
            if batch.num_rows:
                yield self._frame(batch)

    def empty(self, columns: Sequence[str] | None) -> pd.DataFrame:
        """A canonical frame with the selected columns and no rows."""
        schema = self.schema
        if columns is not None:
            schema = pa.schema([schema.field(c) for c in columns], schema.metadata)
        return _canonical_frame(
            schema.empty_table(), self.name, self.metadata, self.version
        )

    def _frame(self, batch: pa.RecordBatch) -> pd.DataFrame:
        # The file's pandas metadata restores what Arrow alone does not, such
        # as an ordered category.
        table = pa.Table.from_batches([batch]).replace_schema_metadata(self.metadata)
        return _canonical_frame(table, self.name, self.metadata, self.version)

    def _time_filter(self, start: Any, end: Any) -> ds.Expression | None:
        if start is None and end is None:
            return None
        if TIMESTAMP_COLUMN not in self.schema.names:
            raise ConfigError(
                f"The {self.name!r} table has no {TIMESTAMP_COLUMN!r} column, "
                "so it cannot be read by time."
            )
        kind = self.schema.field(TIMESTAMP_COLUMN).type
        field = ds.field(TIMESTAMP_COLUMN)
        expression = None
        if start is not None:
            expression = field >= _time_scalar(start, kind)
        if end is not None:
            before_end = field < _time_scalar(end, kind)
            expression = before_end if expression is None else expression & before_end
        return expression


def _time_scalar(value: Any, kind: pa.DataType) -> pa.Scalar:
    """*value* as a scalar of the file's time type; no zone is read as UTC."""
    ts = _as_utc(value, "scan")
    if isinstance(kind, pa.TimestampType) and kind.tz is None:
        # A file written before times carried a zone holds them as UTC.
        ts = ts.tz_localize(None)
    return pa.scalar(ts, type=kind)


def scan_table(
    folder: str | Path,
    table: str,
    *,
    columns: Sequence[str] | None = None,
    start: Any = None,
    end: Any = None,
    filter: ds.Expression | None = None,
    batch_rows: int = BATCH_ROWS,
) -> Iterator[pd.DataFrame]:
    """Read one table of a saved run as pandas frames, one batch at a time.

    Only the named columns are read, and only the row groups that can hold a
    row in the time range.  The batches come in the file's row order, and
    together they hold the same rows :func:`read_table` gives.

    Parameters
    ----------
    folder : str or Path
        A folder written by :func:`~ob_analytics.data.save_data` or
        :meth:`~ob_analytics.pipeline.Pipeline.run_windows`.
    table : str
        The table to read: ``"events"``, ``"trades"``, ``"depth"`` or
        ``"depth_summary"``, or any other ``<table>.parquet`` in the folder.
    columns : sequence of str, optional
        The columns to read, in this order.  ``None`` reads every column.
    start, end : str, datetime or pandas.Timestamp, optional
        Keep the rows with ``start <= timestamp < end``.  A time with no zone
        is read as UTC.  ``None`` leaves that side open.
    filter : pyarrow.dataset.Expression, optional
        A further row filter, for example
        ``pyarrow.dataset.field("direction") == "bid"``.  It is passed to the
        reader together with the time range.
    batch_rows : int, optional
        The most rows in one frame.  The memory the read uses grows with it.

    Returns
    -------
    iterator of pandas.DataFrame
        Each frame has the dtypes :func:`~ob_analytics.data.load_data` gives
        and the tick size and lot size on ``attrs``.  A batch with no rows
        left after the filter is not given.

    Raises
    ------
    ConfigError
        If *folder* is not a folder or has no such table, the file is not
        Parquet this build can read or has an unsupported schema version, a
        named column is missing, or *start* or *end* is not a time.  These are
        raised by the call, before the first frame is read.
    """
    return _SavedTable(folder, table).batches(columns, start, end, filter, batch_rows)


def read_table(
    folder: str | Path,
    table: str,
    *,
    columns: Sequence[str] | None = None,
    start: Any = None,
    end: Any = None,
    filter: ds.Expression | None = None,
) -> pd.DataFrame:
    """Read the selected rows and columns of one table of a saved run.

    The selection is passed to the Parquet reader, as in :func:`scan_table`,
    so a time range or a few columns of a large file read only that part of
    it.  The result is held in memory, so it must fit.

    Parameters
    ----------
    folder, table, columns, start, end, filter
        As for :func:`scan_table`.

    Returns
    -------
    pandas.DataFrame
        The rows in the file's order, with a fresh index, the dtypes
        :func:`~ob_analytics.data.load_data` gives and the tick size and lot
        size on ``attrs``.

    Raises
    ------
    ConfigError
        As for :func:`scan_table`.
    """
    return _SavedTable(folder, table).whole(columns, start, end, filter)


# ── Aggregating ───────────────────────────────────────────────────────


def aggregate_by_time(
    folder: str | Path,
    table: str,
    every: str | pd.Timedelta,
    how: str | Mapping[str, str] = "last",
    *,
    columns: Sequence[str] | None = None,
    start: Any = None,
    end: Any = None,
    filter: ds.Expression | None = None,
    batch_rows: int = BATCH_ROWS,
) -> pd.DataFrame:
    """Reduce one table of a saved run to one row per time bucket, in parts.

    The table is read one batch at a time.  Each batch is reduced per bucket,
    and the partial results of a bucket that two batches share are combined,
    so the result is the same as reducing the whole table at once:
    ``frame.groupby(bucket).agg(how)``.  The memory used is set by
    *batch_rows* and the number of buckets, not by the size of the file.

    Buckets are fixed spans of time counted from the Unix epoch, as everywhere
    else in the package, so a ``"1min"`` bucket starts on the minute.

    Parameters
    ----------
    folder, table
        As for :func:`scan_table`.
    every : str or pandas.Timedelta
        The bucket length, a fixed duration such as ``"1s"``, ``"1min"`` or
        ``"1h"``.
    how : str or mapping of str to str, optional
        The reduction, one of :data:`AGGREGATIONS`.  A string applies to every
        column in *columns*.  A mapping gives one reduction per column, and
        then *columns* must be ``None``.  ``"first"`` and ``"last"`` follow the
        file's row order.  Default ``"last"``: the value at the end of each
        bucket.
    columns : sequence of str, optional
        The columns to reduce when *how* is a string.  ``None`` takes every
        numeric column except ``event_id``.
    start, end, filter, batch_rows
        As for :func:`scan_table`.

    Returns
    -------
    pandas.DataFrame
        One row per bucket that holds a row, in time order: ``timestamp`` (the
        start of the bucket) and one column per reduced column, with the same
        name.  A bucket with no rows is not in the result.  Prices and sizes
        stay in the file's units (integer ticks and lots in a current file);
        the tick size and lot size are on ``attrs``.

    Raises
    ------
    ConfigError
        If *every* is not a fixed duration, a reduction is not one of
        :data:`AGGREGATIONS`, both a mapping and *columns* are given, a column
        is missing, or as for :func:`scan_table`.
    """
    saved = _SavedTable(folder, table)
    if TIMESTAMP_COLUMN not in saved.schema.names:
        raise ConfigError(
            f"The {table!r} table has no {TIMESTAMP_COLUMN!r} column, so it "
            "cannot be cut into time buckets."
        )
    plan = _reductions(saved, how, columns)
    # Check the bucket length before reading anything.
    window_start(pd.Series([], dtype="datetime64[ns, UTC]"), every, "aggregate_by_time")
    read = [TIMESTAMP_COLUMN, *plan]
    empty = saved.empty(read)
    units = {k: v for k, v in empty.attrs.items() if k in UNIT_ATTRS}
    pending: list[pd.DataFrame] = []
    for frame in saved.batches(read, start, end, filter, batch_rows):
        bucket = window_start(frame[TIMESTAMP_COLUMN], every, "aggregate_by_time")
        pending.append(_partial(frame, bucket, plan))
        if len(pending) >= _COMBINE_EVERY:
            pending = [_combine(pending, plan)]
    if not pending:
        result = empty.astype({c: _empty_dtype(empty[c], plan[c]) for c in plan})
        result.attrs = units
        return result
    combined = _combine(pending, plan)
    # Back from nanoseconds to the timestamp column's own type.
    starts = pd.to_datetime(combined.index, unit="ns", utc=True)
    zone = empty[TIMESTAMP_COLUMN].dt.tz
    starts = starts.tz_localize(None) if zone is None else starts.tz_convert(zone)
    combined.index = starts
    result = pd.DataFrame(index=combined.index)
    for column, reduction in plan.items():
        if reduction == "mean":
            result[column] = combined[(column, "sum")] / combined[(column, "count")]
        else:
            result[column] = combined[(column, reduction)]
    result.index.name = TIMESTAMP_COLUMN
    result = result.reset_index()
    result.attrs = units
    return result


def _reductions(
    saved: _SavedTable,
    how: str | Mapping[str, str],
    columns: Sequence[str] | None,
) -> dict[str, str]:
    """The reduction for each column, checked against the table."""
    if isinstance(how, str):
        if columns is None:
            columns = [
                f.name
                for f in saved.schema
                if f.name not in (TIMESTAMP_COLUMN, "event_id")
                and (pa.types.is_integer(f.type) or pa.types.is_floating(f.type))
            ]
        plan = dict.fromkeys(columns, how)
    else:
        if columns is not None:
            raise ConfigError(
                "aggregate_by_time takes columns only with a single reduction; "
                "a mapping names its columns itself."
            )
        plan = dict(how)
    unknown = sorted({r for r in plan.values() if r not in AGGREGATIONS})
    if unknown:
        raise ConfigError(
            f"aggregate_by_time cannot reduce by {unknown}; use one of "
            f"{list(AGGREGATIONS)}."
        )
    missing = [c for c in plan if c not in saved.schema.names]
    if missing:
        raise ConfigError(
            f"The {saved.name!r} table has no column {missing}; it has "
            f"{saved.schema.names}."
        )
    if TIMESTAMP_COLUMN in plan:
        raise ConfigError(
            f"aggregate_by_time cuts the buckets on {TIMESTAMP_COLUMN!r}, so it "
            "cannot reduce that column too."
        )
    if not plan:
        raise ConfigError(f"The {saved.name!r} table has no column to reduce.")
    return plan


def _partial(
    frame: pd.DataFrame, bucket: pd.Series, plan: dict[str, str]
) -> pd.DataFrame:
    """One batch's partial result per bucket, a ``(column, part)`` per column.

    One grouped reduction per part, over every column that needs it.
    """
    # The bucket goes in as a column, not as an array: a zoned time series
    # turned into an array is an array of objects, which groups far slower.
    grouped = frame.assign(_bucket=pd.DatetimeIndex(bucket).asi8).groupby(
        "_bucket", sort=False
    )
    pieces = []
    for part in dict.fromkeys(p for r in plan.values() for p in _PARTS[r]):
        needing = [c for c, r in plan.items() if part in _PARTS[r]]
        piece = getattr(grouped[needing], part)()
        piece.columns = pd.MultiIndex.from_tuples([(c, part) for c in needing])
        pieces.append(piece)
    return pd.concat(pieces, axis=1)


def _combine(pending: list[pd.DataFrame], plan: dict[str, str]) -> pd.DataFrame:
    """Combine partial results, one row per bucket, in time order.

    The partial results are in the file's order, so the first and last of a
    bucket stay the first and last row of the file that falls in it.
    """
    stacked = pd.concat(pending)
    return stacked.groupby(level=0, sort=True).agg(
        {key: _COMBINE[key[1]] for key in stacked.columns}
    )


def _empty_dtype(column: pd.Series, reduction: str) -> Any:
    """The dtype a reduced column would have, for a result with no rows."""
    if reduction == "count":
        return "int64"
    if reduction == "mean":
        return "float64"
    return column.dtype


# ── Rebuilding the depth summary ──────────────────────────────────────


def summarize_depth_file(
    folder: str | Path,
    dest: str | Path,
    *,
    depth_bps: int | None = None,
    depth_bins: int | None = None,
    batch_rows: int = BATCH_ROWS,
) -> Path:
    """Build the depth summary from a saved ``depth`` table, in parts.

    The ``depth`` table is read one batch at a time and replayed through one
    :class:`~ob_analytics.depth.DepthSummaryEngine`, which keeps the book
    from one batch to the next.  Each batch's summary rows are written as
    they are made.  The result is the same as
    :func:`~ob_analytics.depth.summarize_depth` gives on the whole table,
    row for row, and the memory used is set by *batch_rows* and the size of
    the book, not by the size of the file.

    Parameters
    ----------
    folder : str or Path
        A folder written by :func:`~ob_analytics.data.save_data` or
        :meth:`~ob_analytics.pipeline.Pipeline.run_windows` that holds a
        ``depth`` table.
    dest : str or Path
        The folder to write ``depth_summary.parquet`` to; it is created when
        missing.  It may be *folder*, which replaces the run's own summary.
        The file is moved into place only once the whole table is done, so a
        run that stops part-way leaves *dest* as it was.
    depth_bps, depth_bins : int, optional
        The width of each depth bin in basis points, and the number of bins.
        ``None`` keeps the value the run recorded, or the
        :class:`~ob_analytics.config.PipelineConfig` default when it recorded
        none.
    batch_rows : int, optional
        The most depth rows replayed at a time.

    Returns
    -------
    Path
        The path of the ``depth_summary.parquet`` written.

    Raises
    ------
    ConfigError
        If the folder has no readable ``depth`` table, the table has no rows,
        its rows are not in time order, or as for :func:`scan_table`.
    """
    saved = _SavedTable(folder, "depth")
    # A folder with no recorded config falls back to the depth file's tick and
    # lot size, which an empty frame read from it carries.
    config = _saved_config(saved.folder, {"depth": saved.empty(None)})
    changes = {
        key: value
        for key, value in (("depth_bps", depth_bps), ("depth_bins", depth_bins))
        if value is not None
    }
    if changes:
        config = PipelineConfig(**{**config.model_dump(), **changes})
    engine = DepthSummaryEngine(config)
    previous: pd.DataFrame | None = None
    with ParquetAppender(
        dest,
        config,
        declarations=_saved_declarations(saved),
        level=_saved_level(saved),
        context="summarize_depth_file",
        part="batch",
    ) as out:
        for frame in saved.batches(None, None, None, None, batch_rows):
            keys = time_order_keys(frame)
            # Sorted the way the engine sorts, empty values included, so the
            # first and last rows are the ones it replays first and last.
            ordered = frame[keys].sort_values(keys, kind="stable")
            if previous is not None and not _in_order(previous, ordered.iloc[:1]):
                raise ConfigError(
                    f"{saved.path} is not in time order: a row at "
                    f"{ordered[TIMESTAMP_COLUMN].iloc[0]} follows one at "
                    f"{previous[TIMESTAMP_COLUMN].iloc[0]}. summarize_depth_file "
                    "replays the table in parts, so it needs the rows sorted "
                    "by time, as a run writes them."
                )
            previous = ordered.iloc[-1:]
            out.write("depth_summary", engine.compute(frame))
        if previous is None:
            raise ConfigError(f"{saved.path} has no rows to summarize.")
        out.finish(["depth_summary"])
    return Path(dest) / "depth_summary.parquet"


def _in_order(before: pd.DataFrame, after: pd.DataFrame) -> bool:
    """Whether the one-row *before* sorts no later than the one-row *after*.

    Sorted as the engine sorts, so an empty ``sequence`` compares the way it
    does there instead of raising.
    """
    pair = pd.concat([before, after], ignore_index=True)
    keys = list(pair.columns)
    return pair.sort_values(keys, kind="stable").index[0] == 0


def _saved_declarations(saved: _SavedTable) -> FeedDeclarations | None:
    raw = saved.metadata.get(DECLARATIONS_KEY)
    if raw is None:
        return None
    try:
        return FeedDeclarations.from_dict(json.loads(raw.decode()))
    except (ValueError, TypeError, AttributeError):
        # The summary does not depend on them; leave them out of the new file.
        return None


def _saved_level(saved: _SavedTable) -> Level | None:
    raw = saved.metadata.get(LEVEL_KEY)
    if raw is None:
        return None
    try:
        return Level(raw.decode())
    except (ValueError, UnicodeDecodeError):
        return None
