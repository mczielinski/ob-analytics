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
    table = pa.Table.from_pandas(df, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[SCHEMA_VERSION_KEY] = SCHEMA_VERSION.encode()
    if tick_sizes is not None:
        metadata[TICK_SIZE_KEY] = encode_tick_sizes(tick_sizes)
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path)


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
        metadata (checked by :func:`load_data`).  Additional formats (e.g.
        ``"bitstamp"``, ``"lobster"``) are available when the
        corresponding writer factory has been registered via
        :func:`register_writer`.
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

    if writer is not None:
        writer.write(lob_data, p, **write_kwargs)
        return

    if fmt in WRITERS:
        from ob_analytics.config import PipelineConfig
        from ob_analytics.protocols import RunContext

        cfg = config if config is not None else PipelineConfig()
        rctx = ctx if ctx is not None else RunContext()
        w = WRITERS.get(fmt)(cfg, rctx)
        w.write(lob_data, p, **write_kwargs)
        return

    if fmt == "parquet":
        tick_sizes = _tick_sizes_from_config(config)
        p.mkdir(parents=True, exist_ok=True)
        for name, df in lob_data.items():
            _write_versioned_parquet(df, p / f"{name}.parquet", tick_sizes=tick_sizes)
    elif fmt == "pickle":
        logger.warning(
            "Saving as pickle. Consider using fmt='parquet' for "
            "portability and security."
        )
        pd.to_pickle(lob_data, p)  # type: ignore
    else:
        available = ["parquet", "pickle"] + WRITERS.list()
        raise ValueError(
            f"Unsupported format: {fmt!r}. Available: {', '.join(available)}"
        )
