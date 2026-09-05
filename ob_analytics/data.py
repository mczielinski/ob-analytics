"""Data I/O: Parquet serialization and writer registry."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from ob_analytics._registry import Registry
from ob_analytics.protocols import DataWriter
from ob_analytics.schemas import (
    _DEFAULT_TICK_KEY,
    SCHEMA_VERSION,
    SCHEMA_VERSION_KEY,
    TICK_SIZE_KEY,
    check_schema_version,
    decode_tick_sizes,
    encode_tick_sizes,
    resolve_tick_size,
)

if TYPE_CHECKING:
    from ob_analytics.protocols import RunContext


# ── Writer registry ───────────────────────────────────────────────────

WriterFactory = Callable[[Any, "RunContext"], DataWriter]

WRITERS: Registry[str, WriterFactory] = Registry("writer")


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
    """Return a sorted list of registered writer names."""
    return WRITERS.list()


# ── Canonical Parquet I/O (versioned) ─────────────────────────────────


def _tick_sizes_from_config(config: Any) -> dict[str, float] | None:
    """Build the tick-size metadata map from a run's *config* (issue #155).

    Returns ``{"default": config.tick_size}`` when *config* carries a
    ``tick_size``, so ``save_data(config=...)`` tags each Parquet file with the
    tick that scales its integer prices back to the quote currency.  ``None``
    when no config is supplied — the file is then written without tick metadata,
    and a reader falls back to showing ticks as-is.
    """
    if config is None:
        return None
    tick_size = getattr(config, "tick_size", None)
    if tick_size is None:
        return None
    return {_DEFAULT_TICK_KEY: float(tick_size)}


def _write_versioned_parquet(
    df: pd.DataFrame,
    path: Path,
    *,
    tick_sizes: dict[str, float] | None = None,
) -> None:
    """Write *df* to *path* as Parquet, tagging the schema version and tick size.

    Goes through pyarrow so the file carries :data:`SCHEMA_VERSION` under
    :data:`SCHEMA_VERSION_KEY` in its key-value metadata, alongside the pandas
    metadata that preserves dtypes on read.  When *tick_sizes* is given (a
    ``{instrument_key: tick_size}`` map) it is written under
    :data:`TICK_SIZE_KEY` so a reader can recover the float price from the
    integer ticks (issue #155).  The index is dropped, matching the previous
    ``df.to_parquet(..., index=False)`` behaviour.
    """
    pq.write_table(_to_arrow_table(df, tick_sizes=tick_sizes), path)


class OutputTables(dict[str, pd.DataFrame]):
    """A run's output tables, as pandas, that a writer can also ask for in Arrow.

    Issue #216.  ``DataWriter.write`` takes a mapping of pandas frames, and this
    **is** that mapping — every existing writer treats it as the dict it is, and
    the protocol's annotation stays honest.  What it adds is :meth:`arrow`, for a
    writer whose target is columnar: Parquet here, and the Nautilus catalogue in
    #113.

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
        Tick sizes to record in :meth:`arrow`'s metadata (issue #155).  ``None``
        when the caller declared no config, which writes no tick metadata rather
        than a default one.
    """

    def __init__(
        self,
        tables: dict[str, pd.DataFrame],
        *,
        tick_sizes: dict[str, float] | None = None,
    ) -> None:
        super().__init__(tables)
        self._tick_sizes = tick_sizes

    def arrow(self) -> dict[str, pa.Table]:
        """Return the same tables as canonical Arrow, keyed the same way.

        Each table carries the schema version and, when the run declared one,
        the tick size — the same key-value metadata a canonical Parquet file
        carries, so a writer building one is no worse off than
        :class:`ParquetWriter`.
        """
        return {
            name: _to_arrow_table(df, tick_sizes=self._tick_sizes)
            for name, df in self.items()
        }


class ParquetWriter:
    """Write a run's frames as one canonical Parquet file per key.

    The library's default output format, and a registered writer like any
    other (issue #216) rather than a branch inside :func:`save_data`, so a user
    can register their own under ``"parquet"`` and replace it.

    Satisfies the :class:`~ob_analytics.protocols.DataWriter` protocol.
    """

    def __init__(self, config: Any = None) -> None:
        self._tick_sizes = _tick_sizes_from_config(config)

    def write(
        self,
        data: dict[str, pd.DataFrame],
        dest: str | Path,
        **kwargs: Any,
    ) -> Path:
        """Write each frame in *data* to ``<dest>/<key>.parquet``.

        *dest* is a directory and is created when missing.  Each file carries
        the schema version and, when the run's config named one, the tick size
        (issue #155), so :func:`load_data` can check the first and restore
        prices with the second.
        """
        p = Path(dest)
        p.mkdir(parents=True, exist_ok=True)
        for name, df in data.items():
            _write_versioned_parquet(
                df, p / f"{name}.parquet", tick_sizes=self._tick_sizes
            )
        return p


class PickleWriter:
    """Write a run's frames as one pickle file.

    Kept for backward compatibility and warned about on every call: a pickle
    executes code on load, so it is unsafe for data you did not write.  A
    registered writer like any other (issue #216).

    Satisfies the :class:`~ob_analytics.protocols.DataWriter` protocol.
    """

    def write(
        self,
        data: dict[str, pd.DataFrame],
        dest: str | Path,
        **kwargs: Any,
    ) -> Path:
        """Write *data* whole to *dest* with :func:`pandas.to_pickle`."""
        logger.warning(
            "Saving as pickle. Consider using fmt='parquet' for "
            "portability and security."
        )
        p = Path(dest)
        # ``dict(data)``, not *data*: the payload is a dict subclass (#216) and
        # pickling it whole would write ob-analytics' own class into the file,
        # so the file would only load where that class exists.
        pd.to_pickle(dict(data), p)  # type: ignore
        return p


def _to_arrow_table(
    df: pd.DataFrame,
    *,
    tick_sizes: dict[str, float] | None = None,
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
        Tick sizes to record, keyed by instrument (issue #155).  Omitted
        metadata means a reader sees the integer prices as-is.

    Returns
    -------
    pyarrow.Table
        *df* as Arrow, with the metadata attached.
    """
    table = pa.Table.from_pandas(df, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[SCHEMA_VERSION_KEY] = SCHEMA_VERSION.encode()
    if tick_sizes is not None:
        metadata[TICK_SIZE_KEY] = encode_tick_sizes(tick_sizes)
    return table.replace_schema_metadata(metadata)


def _read_versioned_parquet(path: Path) -> pd.DataFrame:
    """Read a canonical Parquet *path*, checking its schema version first.

    Raises :class:`~ob_analytics.exceptions.ConfigError` on an unsupported
    version; a file with no version key loads as legacy data with a warning
    (see :func:`ob_analytics.schemas.check_schema_version`).  The tick size
    stored under :data:`TICK_SIZE_KEY` (issue #155) is surfaced on the returned
    frame's ``attrs``: ``df.attrs["tick_sizes"]`` holds the full instrument map
    and ``df.attrs["tick_size"]`` the resolved default, so ``price * tick_size``
    recovers the quote currency.  A legacy (pre-#155) file has neither.
    """
    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    raw = metadata.get(SCHEMA_VERSION_KEY)
    version = raw.decode() if raw is not None else None
    check_schema_version(version, source=path.name)
    df = table.to_pandas()
    tick_sizes = decode_tick_sizes(metadata.get(TICK_SIZE_KEY))
    if tick_sizes is not None:
        df.attrs["tick_sizes"] = tick_sizes
        resolved = resolve_tick_size(tick_sizes)
        if resolved is not None:
            df.attrs["tick_size"] = resolved
    return df


def load_data(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load pre-processed pipeline data from a Parquet directory or pickle file.

    Parameters
    ----------
    path : str or Path
        If *path* is a directory, each ``.parquet`` file inside is loaded
        as a DataFrame keyed by its stem (``events.parquet`` → ``"events"``).
        If *path* is a single file with a ``.pkl`` / ``.pickle`` extension,
        it is loaded via :func:`pandas.read_pickle` for backward
        compatibility (**not recommended** for untrusted data).

    Returns
    -------
    dict of str to pandas.DataFrame

    Raises
    ------
    ConfigError
        If a Parquet file declares a schema version this build does not
        support.  A file written with no version key (legacy data, or the
        bundled sample) loads with a warning — see
        :func:`ob_analytics.schemas.check_schema_version`.
    """
    p = Path(path)
    if p.is_dir():
        result = {}
        for parquet_path in sorted(p.glob("*.parquet")):
            result[parquet_path.stem] = _read_versioned_parquet(parquet_path)
        if not result:
            raise FileNotFoundError(f"No .parquet files found in {p}")
        return result
    if p.suffix in (".pkl", ".pickle"):
        logger.warning(
            "Loading from pickle ({}). Pickle is insecure for untrusted "
            "data; prefer Parquet via save_data().",
            p,
        )
        return pd.read_pickle(p)
    raise ValueError(
        f"Unsupported format: {p.suffix}. Use a Parquet directory or .pkl file."
    )


def save_data(
    lob_data: dict[str, pd.DataFrame],
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
    lob_data : dict of str to pandas.DataFrame
        The DataFrames to save (keys become file stems).
    path : str or Path
        Destination directory (Parquet) or file (pickle).
    fmt : str
        Serialisation format.  Built-in values are ``"parquet"``
        (default) and ``"pickle"``.  The ``"parquet"`` path writes one
        file per key and tags each with :data:`SCHEMA_VERSION` in its
        metadata (checked by :func:`load_data`).  A **source name** (e.g.
        ``"bitstamp"``, ``"lobster"``) round-trips through that source's own
        writer (its ``create_writer`` capability, so no separate writer
        registration is needed); a source that needs run state — LOBSTER's
        ``trading_date`` — reads it from *ctx*.  A generic, source-independent
        writer registered via :func:`register_writer` is also resolved by name.
    writer : DataWriter, optional
        A pre-constructed writer instance.  When provided, *fmt* is
        ignored and the writer is used directly.  This is the preferred
        path when saving from a :class:`Pipeline` that already holds a
        configured writer.
    config, ctx
        Forwarded to a registered writer factory when ``fmt`` names one.
        ``ctx`` defaults to an empty
        :class:`~ob_analytics.protocols.RunContext`.
    **write_kwargs
        Extra keyword arguments forwarded to ``writer.write()``.
    """
    p = Path(path)
    # Every writer is handed the same payload: a mapping of pandas frames that
    # can also produce canonical Arrow (#216).  It is a dict, so a writer that
    # ignores the extra sees exactly what it saw before.
    tables = OutputTables(lob_data, tick_sizes=_tick_sizes_from_config(config))

    if writer is not None:
        writer.write(tables, p, **write_kwargs)
        return

    resolved = _named_writer(fmt, config, ctx)
    if resolved is not None:
        resolved.write(tables, p, **write_kwargs)
        return

    from ob_analytics.sources import SOURCES

    available = [*WRITERS.list(), *SOURCES.list()]
    raise ValueError(f"Unsupported format: {fmt!r}. Available: {', '.join(available)}")


def _named_writer(fmt: str, config: Any, ctx: Any) -> DataWriter | None:
    """Resolve *fmt* to a writer: a registered generic writer, or a source's own.

    A generic writer registered via :func:`register_writer` wins; otherwise a
    source name resolves to that source's ``create_writer`` (so a venue writer
    lives on the source, not in a parallel registry).  Returns ``None`` when
    *fmt* names neither.
    """
    from ob_analytics.protocols import RunContext
    from ob_analytics.sources import SOURCES

    # *config* is passed through as given, ``None`` included: every writer
    # defaults for itself, and a writer that records what the caller declared
    # must be able to tell "no config" from a default one.  Substituting a
    # ``PipelineConfig()`` here would tag a file with its default tick size
    # (#155) that the caller never asked for.  *ctx* is different: an empty
    # ``RunContext`` is an absence, not an invented value.
    cfg = config
    rctx = ctx if ctx is not None else RunContext()

    if fmt in WRITERS:
        return WRITERS.get(fmt)(cfg, rctx)
    if fmt in SOURCES:
        make_writer = getattr(SOURCES.get(fmt)(), "create_writer", None)
        if make_writer is not None:
            return make_writer(cfg, rctx)
    return None


# Built-in output formats self-register, the way sources and metrics do, so
# ``save_data(fmt=...)`` has one resolution path and no special cases (#216).
register_writer("parquet", lambda config, ctx: ParquetWriter(config))
register_writer("pickle", lambda config, ctx: PickleWriter())
