"""Data I/O: Parquet serialization and writer registry."""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ob_analytics._registry import Registry, load_plugins
from ob_analytics.config import PipelineConfig, instrument_fields
from ob_analytics.exceptions import ConfigError
from ob_analytics.protocols import DataWriter, FeedDeclarations, WriteTarget
from ob_analytics.schemas import (
    CONFIG_KEY,
    DECLARATIONS_KEY,
    EVENT_DTYPES,
    LOT_SIZE_KEY,
    SCHEMA_VERSION,
    SCHEMA_VERSION_KEY,
    TICK_SIZE_KEY,
    TRADE_DTYPES,
    UNIT_ATTRS,
    cast_columns,
    check_schema_version,
    decode_lot_sizes,
    decode_tick_sizes,
    encode_lot_sizes,
    encode_tick_sizes,
    lot_sizes_from_config,
    put_units,
    tick_sizes_from_config,
)

if TYPE_CHECKING:
    from ob_analytics.pipeline import PipelineResult
    from ob_analytics.protocols import RunContext


# ── Writer registry ───────────────────────────────────────────────────

WriterFactory = Callable[[Any, "RunContext"], DataWriter]

WRITERS: Registry[str, WriterFactory] = Registry("writer")

#: The entry-point group a third-party package advertises a writer under, e.g.
#: ``[project.entry-points."ob_analytics.writers"]`` with ``arrow_ipc =
#: my_pkg.arrow:make_arrow_writer``.
ENTRY_POINT_GROUP = "ob_analytics.writers"


def register_writer(name: str, factory: WriterFactory) -> None:
    """Register a writer *factory* under *name* for use with
    ``save_data(fmt=name, ctx=...)``.

    The factory is called as ``factory(config, ctx)`` and must return a
    :class:`DataWriter`. This is what lets format-specific writers
    (e.g. :class:`~ob_analytics.lobster.LobsterWriter`, which needs
    ``trading_date``) participate in the registry — they pull required
    parameters from the :class:`~ob_analytics.protocols.RunContext`.
    """
    WRITERS.register(name, factory)


def list_writers() -> list[str]:
    """Return every format :func:`save_data` accepts, sorted.

    The formats are the writers registered via :func:`register_writer` and
    the sources that write their own format (those with a ``create_writer``
    method).  This is the list ``save_data`` checks *fmt* against and names
    in its error.
    """
    from ob_analytics.sources import SOURCES

    sources = [name for name in SOURCES.list() if _writes_a_format(name)]
    return sorted({*WRITERS.list(), *sources})


def _register_entry_point(ep: EntryPoint) -> str:
    register_writer(ep.name, ep.load())
    return ep.name


def load_writer_plugins(*, force: bool = False) -> list[str]:
    """Discover and register writers advertised via entry points.

    Scans the :data:`ENTRY_POINT_GROUP` entry-point group; each entry's value
    is loaded to a writer factory ``(config, ctx) -> DataWriter`` and
    registered under the entry-point name, as :func:`register_writer` would.
    This is what lets a writer live in a separate installable package without
    editing ob-analytics.

    The scan runs once per process unless *force* is set.  A plug-in that
    fails to load is skipped with a warning, so one broken package cannot stop
    the rest from loading.

    Returns
    -------
    list of str
        The names newly registered by this call.
    """
    return load_plugins(ENTRY_POINT_GROUP, _register_entry_point, force=force)


# ── Canonical Parquet I/O (versioned) ─────────────────────────────────


def _write_versioned_parquet(
    df: pd.DataFrame,
    path: Path,
    *,
    tick_sizes: dict[str, float] | None = None,
    lot_sizes: dict[str, float] | None = None,
    config: Any = None,
    declarations: FeedDeclarations | None = None,
) -> None:
    """Write *df* to *path* as Parquet, tagging the version, tick and lot size.

    Goes through pyarrow so the file carries :data:`SCHEMA_VERSION` under
    :data:`SCHEMA_VERSION_KEY` in its key-value metadata, alongside the pandas
    metadata that preserves dtypes on read.  When *tick_sizes* is given (a
    ``{instrument_key: tick_size}`` map) it is written under
    :data:`TICK_SIZE_KEY` so a reader can recover the float price from the
    integer ticks.  *lot_sizes* does the same for the integer sizes under
    :data:`LOT_SIZE_KEY`, *config* records the whole run configuration
    under :data:`CONFIG_KEY`, and *declarations* the run's declarations under
    :data:`DECLARATIONS_KEY`.  The index is dropped,
    matching the previous ``df.to_parquet(..., index=False)`` behaviour.
    """
    pq.write_table(
        _to_arrow_table(
            df,
            tick_sizes=tick_sizes,
            lot_sizes=lot_sizes,
            config=config,
            declarations=declarations,
        ),
        path,
    )


class OutputTables(dict[str, pd.DataFrame]):
    """A run's output tables, as pandas, that a writer can also ask for in Arrow.

    ``DataWriter.write`` takes a mapping of pandas frames, and this **is** that
    mapping — every existing writer treats it as the dict it is, and the
    protocol's annotation stays honest.  What it adds is :meth:`arrow`, for a
    writer whose target is columnar: Parquet here, and the Nautilus catalogue.

    Without it such a writer would call ``pa.Table.from_pandas`` itself and
    silently drop the schema version and tick size that make the output
    canonical, because the function that attaches them is private.  The frame
    type therefore stays out of the protocol: a writer asks for the shape it
    wants rather than the pipeline guessing which one every writer needs.

    Parameters
    ----------
    tables : mapping of str to pandas.DataFrame
        The run's tables, keyed by name.
    tick_sizes : dict of str to float, optional
        Tick sizes to record in :meth:`arrow`'s metadata.  ``None`` when the
        caller declared no config, which writes no tick metadata rather than a
        default one.
    lot_sizes : dict of str to float, optional
        Lot sizes to record, in the same way.
    config : PipelineConfig, optional
        The run's configuration, recorded whole so
        :func:`load_result` can rebuild the run.
    declarations : FeedDeclarations, optional
        The run's declarations, recorded so the rebuilt run is audited
        against them.  ``None`` when the tables are not a run's.
    """

    def __init__(
        self,
        tables: dict[str, pd.DataFrame],
        *,
        tick_sizes: dict[str, float] | None = None,
        lot_sizes: dict[str, float] | None = None,
        config: PipelineConfig | None = None,
        declarations: FeedDeclarations | None = None,
    ) -> None:
        super().__init__(tables)
        self._tick_sizes = tick_sizes
        self._lot_sizes = lot_sizes
        self._config = config
        self.declarations = declarations

    def arrow(self) -> dict[str, pa.Table]:
        """Return the same tables as canonical Arrow, keyed the same way.

        Each table carries the schema version and, when the run declared one,
        the tick size, the lot size, the configuration and the
        :attr:`declarations` — the same key-value
        metadata a canonical Parquet file carries, so a writer building one is
        no worse off than :class:`ParquetWriter`.
        """
        return {
            name: _to_arrow_table(
                df,
                tick_sizes=self._tick_sizes,
                lot_sizes=self._lot_sizes,
                config=self._config,
                declarations=self.declarations,
            )
            for name, df in self.items()
        }


class ParquetWriter:
    """Write a run's frames as one canonical Parquet file per key.

    The library's default output format, and a registered writer like any other
    rather than a branch inside :func:`save_data`, so a user can register their
    own under ``"parquet"`` and replace it.  It writes a folder.

    Satisfies the :class:`~ob_analytics.protocols.DataWriter` protocol.
    """

    target = WriteTarget.FOLDER

    def __init__(self, config: Any = None) -> None:
        self._tick_sizes = tick_sizes_from_config(config)
        self._lot_sizes = lot_sizes_from_config(config)
        self._config = config

    def write(
        self,
        data: dict[str, pd.DataFrame],
        dest: str | Path,
        **kwargs: Any,
    ) -> Path:
        """Write each frame in *data* to ``<dest>/<key>.parquet``.

        *dest* is a directory and is created when missing.  Each file carries
        the schema version and, when the writer was given the run's config,
        the tick size, the lot size and the config itself, so
        :func:`load_data` can check the first and restore prices and sizes,
        and :func:`load_result` can rebuild the run.  When *data* is a run's
        :class:`OutputTables`, its declarations are recorded too.
        """
        p = Path(dest)
        p.mkdir(parents=True, exist_ok=True)
        declarations = data.declarations if isinstance(data, OutputTables) else None
        for name, df in data.items():
            _write_versioned_parquet(
                df,
                p / f"{name}.parquet",
                tick_sizes=self._tick_sizes,
                lot_sizes=self._lot_sizes,
                config=self._config,
                declarations=declarations,
            )
        return p


def _to_arrow_table(
    df: pd.DataFrame,
    *,
    tick_sizes: dict[str, float] | None = None,
    lot_sizes: dict[str, float] | None = None,
    config: Any = None,
    declarations: FeedDeclarations | None = None,
) -> pa.Table:
    """Convert *df* to an Arrow table tagged with the canonical metadata.

    The table carries :data:`SCHEMA_VERSION` under :data:`SCHEMA_VERSION_KEY`
    and, when *tick_sizes* is given, the ``{instrument_key: tick_size}`` map
    under :data:`TICK_SIZE_KEY` — the same key-value metadata a canonical
    Parquet file carries, so a reader handed a table from memory is no worse off
    than one reading a file.  The pandas index is dropped.

    Parameters
    ----------
    df : pandas.DataFrame
        A canonical pipeline frame.
    tick_sizes : dict of str to float, optional
        Tick sizes to record, keyed by instrument.  Omitted
        metadata means a reader sees the integer prices as-is.
    lot_sizes : dict of str to float, optional
        Lot sizes to record, keyed by instrument.  Omitted
        metadata means a reader sees the integer sizes as-is.
    config : PipelineConfig, optional
        The run's configuration, recorded whole under :data:`CONFIG_KEY`.
        Anything else is not recorded.
    declarations : FeedDeclarations, optional
        The run's declarations, recorded under :data:`DECLARATIONS_KEY`.

    Returns
    -------
    pyarrow.Table
        *df* as Arrow, with the metadata attached.  The tick and lot size on
        *df*'s ``attrs`` are not written a second time with pandas' own
        metadata: the key-value metadata above is where they are stored.
    """
    if any(key in df.attrs for key in UNIT_ATTRS):
        df = df.copy(deep=False)
        df.attrs = {k: v for k, v in df.attrs.items() if k not in UNIT_ATTRS}
    table = pa.Table.from_pandas(df, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[SCHEMA_VERSION_KEY] = SCHEMA_VERSION.encode()
    if tick_sizes is not None:
        metadata[TICK_SIZE_KEY] = encode_tick_sizes(tick_sizes)
    if lot_sizes is not None:
        metadata[LOT_SIZE_KEY] = encode_lot_sizes(lot_sizes)
    if isinstance(config, PipelineConfig):
        metadata[CONFIG_KEY] = config.model_dump_json().encode()
    if declarations is not None:
        metadata[DECLARATIONS_KEY] = json.dumps(declarations.to_dict()).encode()
    return table.replace_schema_metadata(metadata)


#: Tables whose dtypes ``load_data`` restores when the file is in the current
#: schema version.  Parquet keeps most dtypes, but an ``object`` column of ints
#: (the trade ``maker`` and ``taker`` order ids, LOBSTER's event
#: ``raw_event_type``) comes back ``int64``, so a reloaded table would not match
#: one a loader made.  Trades cast through their whole map.  Events cast only
#: their ``object`` columns: the order ``id`` may be text, and the rest already
#: come back as they were saved.
_RESTORED_DTYPES: dict[str, dict[str, str | pd.CategoricalDtype]] = {
    "events": {col: t for col, t in EVENT_DTYPES.items() if t == "object"},
    "trades": TRADE_DTYPES,
}


def _read_versioned_parquet(path: Path) -> pd.DataFrame:
    """Read a canonical Parquet *path*, checking its schema version first.

    Raises :class:`~ob_analytics.exceptions.ConfigError` on an unsupported
    version; a file with no version key loads as legacy data with a warning
    (see :func:`ob_analytics.schemas.check_schema_version`).  The tick size
    stored under :data:`TICK_SIZE_KEY` is surfaced on the returned frame's
    ``attrs``: ``df.attrs["tick_sizes"]`` holds the full instrument map and
    ``df.attrs["tick_size"]`` the resolved default, so ``price * tick_size``
    recovers the quote currency.  An older file that stored float
    quote-currency prices has neither.
    """
    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    raw = metadata.get(SCHEMA_VERSION_KEY)
    version = raw.decode() if raw is not None else None
    check_schema_version(version, source=path.name)
    df = table.to_pandas()
    # An older version stored float prices and sizes; casting them to integer
    # ticks and lots would change the numbers, so only a current file is cast.
    if version == SCHEMA_VERSION and path.stem in _RESTORED_DTYPES:
        df = cast_columns(df, _RESTORED_DTYPES[path.stem])
    put_units(
        df,
        tick_sizes=decode_tick_sizes(metadata.get(TICK_SIZE_KEY)),
        lot_sizes=decode_lot_sizes(metadata.get(LOT_SIZE_KEY)),
    )
    return df


def load_data(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load pre-processed pipeline data from a Parquet folder.

    Parameters
    ----------
    path : str or Path
        A folder written by :func:`save_data` in its default Parquet format.
        Each ``.parquet`` file inside is loaded as a DataFrame keyed by its
        stem (``events.parquet`` → ``"events"``).

    Returns
    -------
    dict of str to pandas.DataFrame

    Raises
    ------
    ConfigError
        If *path* is not a folder, or a Parquet file declares a schema version
        this build does not support.  A file written with no version key
        (legacy data, or the bundled sample) loads with a warning — see
        :func:`ob_analytics.schemas.check_schema_version`.
    FileNotFoundError
        If the folder holds no Parquet file.
    """
    p = Path(path)
    if not p.is_dir():
        raise ConfigError(
            f"{p} is not a Parquet folder. load_data and load_result read the "
            "folder save_data writes in its default format; to read another "
            "format, run it through its source with Pipeline(source=...)."
        )
    result = {}
    for parquet_path in sorted(p.glob("*.parquet")):
        result[parquet_path.stem] = _read_versioned_parquet(parquet_path)
    if not result:
        raise FileNotFoundError(f"No .parquet files found in {p}")
    return result


def save_data(
    lob_data: PipelineResult | dict[str, pd.DataFrame],
    path: str | Path,
    *,
    fmt: str = "parquet",
    writer: DataWriter | None = None,
    config: Any = None,
    ctx: Any = None,
    **write_kwargs: Any,
) -> None:
    """Save pipeline data to disk.

    Parameters
    ----------
    lob_data : PipelineResult or dict of str to pandas.DataFrame
        A pipeline result, or the DataFrames to save (keys become file
        stems).  A result is saved as its four tables, with its own config
        unless *config* is given, and with its declarations.
        :func:`load_result` reads a Parquet folder back as the same result.
    path : str or Path
        A folder for this package's own formats (``"parquet"``,
        ``"depth_csv"``, ``"bitstamp"``, ``"lobster"``), and a file for an
        export to another tool (``"hftbacktest"``, ``"nautilus"``,
        ``"databento"``).  Each writer declares which it takes as its
        :attr:`~ob_analytics.protocols.DataWriter.target`.  A folder is
        created when missing.
    fmt : str
        The format, one of :func:`list_writers`.  The default, ``"parquet"``,
        writes one file per key and tags each with :data:`SCHEMA_VERSION` in
        its metadata (checked by :func:`load_data`).  A **source name** (e.g.
        ``"bitstamp"``, ``"lobster"``) round-trips through that source's own
        writer (its ``create_writer`` method, so no separate writer
        registration is needed); a source that needs run state — LOBSTER's
        ``trading_date`` — reads it from *ctx*.  A generic, source-independent
        writer registered via :func:`register_writer` is also resolved by name,
        and wins over a source of the same name.
    writer : DataWriter, optional
        A pre-constructed writer instance.  When provided, *fmt* is
        ignored and the writer is used directly.
    config, ctx
        Forwarded to a registered writer factory when ``fmt`` names one.
        The ``"parquet"`` writer records *config* in each file, so the files
        keep the run's tick size, lot size and display precision.
        ``ctx`` defaults to an empty
        :class:`~ob_analytics.protocols.RunContext`.
    **write_kwargs
        Extra keyword arguments forwarded to ``writer.write()``.

    Raises
    ------
    ValueError
        If *fmt* is not one of :func:`list_writers`.
    ConfigError
        If *path* is a file where the writer takes a folder, or a folder
        where it takes a file.  The message says which the writer expected.
        Also if the writer declares no ``target``.
    """
    from ob_analytics.pipeline import PipelineResult

    p = Path(path)
    declarations = None
    if isinstance(lob_data, PipelineResult):
        if config is None:
            config = lob_data.config
        declarations = lob_data.declarations
        lob_data = lob_data.tables()
    # Every writer is handed the same payload: a mapping of pandas frames that
    # can also produce canonical Arrow (#216).  It is a dict, so a writer that
    # ignores the extra sees exactly what it saw before.
    tables = OutputTables(
        lob_data,
        tick_sizes=tick_sizes_from_config(config),
        lot_sizes=lot_sizes_from_config(config),
        config=config if isinstance(config, PipelineConfig) else None,
        declarations=declarations,
    )

    if writer is None:
        writer = _named_writer(fmt, config, ctx)
        label = repr(fmt)
    else:
        label = type(writer).__name__
    _check_target(p, writer, label)
    writer.write(tables, p, **write_kwargs)


#: Suffixes that mark a path as a file, so a folder format refuses it before it
#: makes a folder called ``run.parquet``.  A folder name may hold a dot (a date
#: such as ``btc_2024.05.01``), so only these suffixes count.
_FILE_SUFFIXES = frozenset(
    {".csv", ".dbn", ".gz", ".json", ".npz", ".parquet", ".pickle", ".pkl", ".zst"}
)


def _check_target(path: Path, writer: DataWriter, label: str) -> None:
    """Refuse *path* when it is not the folder or file *writer* takes.

    *label* names the format in the message: the ``fmt`` given, or the
    writer's class when the caller passed a writer.  A writer that declares
    no ``target`` is refused too.
    """
    declared = getattr(writer, "target", None)
    try:
        target = WriteTarget(declared)
    except ValueError:
        raise ConfigError(
            f"{label} declares no target: give the writer class "
            "target = WriteTarget.FOLDER or target = WriteTarget.FILE, so "
            "save_data can check the path it is given."
        ) from None
    if target is WriteTarget.FOLDER:
        if path.exists() and not path.is_dir():
            found = "an existing file"
        elif path.suffix.lower() in _FILE_SUFFIXES:
            found = f"a file name (it ends in {path.suffix})"
        else:
            return
        raise ConfigError(
            f"{label} writes a folder, and {path} is {found}. Pass the path of "
            "a folder to write into; it is created when missing."
        )
    if path.is_dir():
        found = "an existing folder"
    elif not path.suffix:
        found = "a folder name (it has no file extension)"
    else:
        return
    raise ConfigError(
        f"{label} writes one file, and {path} is {found}. Pass the path of the "
        "file to write, with its extension."
    )


#: The tables a saved run holds, in the order :class:`PipelineResult` lists them.
_RESULT_TABLES: tuple[str, ...] = ("events", "trades", "depth", "depth_summary")


def load_result(path: str | Path, *, source: Any = None) -> PipelineResult:
    """Load a saved run back into a :class:`~ob_analytics.pipeline.PipelineResult`.

    The inverse of ``save_data(result, path)``: the four tables come back
    as they were saved, and so does the run's
    :class:`~ob_analytics.config.PipelineConfig`, which each Parquet file
    records.  A result rebuilt this way shows the same prices and sizes as
    the run did, for example in
    :func:`~ob_analytics.visualization.display_result` or a gallery.

    A folder that records only the tick and lot size (written by
    :meth:`~ob_analytics.pipeline.Pipeline.run_windows`, or by an older
    version) gets those two, with the display precision that shows one step
    of each.  A folder with neither (prices and sizes stored as floats) gets
    a tick size of 1, so its prices read as they are.

    The resolution comes from the tables: an
    :attr:`~ob_analytics.protocols.Level.L2` run has an empty ``events``
    table, and an L3 run never does.

    The run's declarations (:attr:`PipelineResult.declarations`) come back
    too.  A folder written by an older version does not record them; it gets
    those of the capture record ``ob-analytics process`` keeps beside the
    tables, over those of *source*
    (:func:`~ob_analytics.capture_record.run_declarations`).

    Parameters
    ----------
    path : str or Path
        A Parquet folder written by :func:`save_data` or by
        ``ob-analytics process``.
    source : Source, optional
        The source whose declarations to use when the folder records none.
        ``None`` leaves them undeclared.  Recorded declarations always win,
        since they describe the data the run read.

    Returns
    -------
    PipelineResult
        The saved run.

    Raises
    ------
    ConfigError
        If *path* is not a folder (it does not exist, or it is a file), if
        the folder lacks one of the four tables, or if a Parquet file
        declares a schema version this build does not support.
    FileNotFoundError
        If the folder holds no Parquet file at all.
    """
    from ob_analytics.pipeline import PipelineResult
    from ob_analytics.protocols import Level

    p = Path(path)
    data = load_data(p)
    missing = [name for name in _RESULT_TABLES if name not in data]
    if missing:
        raise ConfigError(
            f"{p} is not a saved run: it has no {', '.join(missing)} table. "
            f"A saved run holds {', '.join(_RESULT_TABLES)}."
        )
    return PipelineResult(
        events=data["events"],
        trades=data["trades"],
        depth=data["depth"],
        depth_summary=data["depth_summary"],
        config=_saved_config(p, data),
        level=Level.L2 if data["events"].empty else Level.L3,
        declarations=_saved_declarations(p, source),
    )


def _saved_declarations(path: Path, source: Any) -> FeedDeclarations:
    """The declarations a saved run recorded, or its capture's and *source*'s."""
    from ob_analytics.capture_record import run_declarations

    for name in _RESULT_TABLES:
        file = path / f"{name}.parquet"
        raw = (pq.read_schema(file).metadata or {}).get(DECLARATIONS_KEY)
        if raw is None:
            continue
        try:
            return FeedDeclarations.from_dict(json.loads(raw.decode()))
        except (ValueError, TypeError, AttributeError) as exc:
            warnings.warn(
                f"{file}: the saved declarations cannot be read by this "
                f"version ({exc}); using the capture record's and the "
                "source's instead.",
                UserWarning,
                stacklevel=3,
            )
            break
    return run_declarations(path, source)


def _saved_config(path: Path, data: dict[str, pd.DataFrame]) -> PipelineConfig:
    """The config a saved run recorded, or the one its tick and lot size give."""
    for name in _RESULT_TABLES:
        file = path / f"{name}.parquet"
        if not file.is_file():
            continue
        raw = (pq.read_schema(file).metadata or {}).get(CONFIG_KEY)
        if raw is None:
            continue
        try:
            recorded = json.loads(raw.decode())
            # Only the fields this version knows, so a file from a later
            # version still loads.
            known = {
                k: v for k, v in recorded.items() if k in PipelineConfig.model_fields
            }
            return PipelineConfig(**known)
        except (ValueError, TypeError, AttributeError) as exc:
            warnings.warn(
                f"{file}: the saved config cannot be read by this version "
                f"({exc}); using the file's tick size and lot size instead.",
                UserWarning,
                stacklevel=3,
            )
            break
    tick_size, lot_size = (
        next((df.attrs[key] for df in data.values() if key in df.attrs), None)
        for key in ("tick_size", "lot_size")
    )
    fields = instrument_fields(tick_size=tick_size, lot_size=lot_size)
    _warn_missing_units(path, data, tick_size=tick_size, lot_size=lot_size)
    if tick_size is None:
        # A file from before integer prices has no tick size and float prices:
        # a tick of 1 shows them as they are.
        fields["tick_size"] = 1.0
    return PipelineConfig(**fields)


def _warn_missing_units(
    path: Path,
    data: dict[str, pd.DataFrame],
    *,
    tick_size: float | None,
    lot_size: float | None,
) -> None:
    """Warn when integer prices or sizes were saved with no tick or lot size.

    Such a folder was written by ``save_data`` without a config.  Its prices
    are ticks and its sizes lots of a step nothing records, so they cannot be
    turned back into the quote currency and the base asset.
    """
    missing = [
        name
        for name, step, column in (
            ("tick_size", tick_size, "price"),
            ("lot_size", lot_size, "volume"),
        )
        if step is None
        and any(
            column in df.columns and pd.api.types.is_integer_dtype(df[column])
            for df in data.values()
        )
    ]
    if missing:
        warnings.warn(
            f"{path} holds whole-number prices or sizes but records no "
            f"{' or '.join(missing)}, so they cannot be shown in the "
            "instrument's own units. Save the run with save_data(result, path) "
            "so each file records its config.",
            UserWarning,
            stacklevel=4,
        )


def _named_writer(fmt: str, config: Any, ctx: Any) -> DataWriter:
    """Resolve *fmt* to a writer: a registered generic writer, or a source's own.

    A generic writer registered via :func:`register_writer` wins; otherwise a
    source name resolves to that source's ``create_writer`` (so a venue writer
    lives on the source, not in a parallel registry).

    Raises
    ------
    ValueError
        If *fmt* names neither, listing the formats :func:`list_writers` gives.
    """
    from ob_analytics.protocols import RunContext, SupportsWriter

    # *config* is passed through as given, ``None`` included: every writer
    # defaults for itself, and a writer that records what the caller declared
    # must be able to tell "no config" from a default one.  Substituting a
    # ``PipelineConfig()`` here would tag a file with its default tick size
    # (#155) that the caller never asked for.  *ctx* is different: an empty
    # ``RunContext`` is an absence, not an invented value.
    rctx = ctx if ctx is not None else RunContext()

    if fmt in WRITERS:
        return WRITERS.get(fmt)(config, rctx)
    if _writes_a_format(fmt):
        from ob_analytics.sources import SOURCES

        source = SOURCES.get(fmt)()
        if isinstance(source, SupportsWriter):
            return source.create_writer(config, rctx)
    raise ValueError(
        f"Unsupported format: {fmt!r}. Available: {', '.join(list_writers())}"
    )


def _writes_a_format(name: str) -> bool:
    """Whether the source called *name* writes a format of its own.

    It does when its class has a ``create_writer`` method.  This is the one
    test of whether a source name is a format, so :func:`list_writers` and
    :func:`save_data` cannot disagree.
    """
    from ob_analytics.sources import SOURCES

    return name in SOURCES and callable(
        getattr(SOURCES.get(name), "create_writer", None)
    )


# Built-in output formats self-register, the way sources and metrics do, so
# ``save_data(fmt=...)`` has one resolution path and no special cases (#216).
register_writer("parquet", lambda config, ctx: ParquetWriter(config))
