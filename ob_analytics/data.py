"""Data I/O: Parquet serialization and writer registry."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from loguru import logger

from ob_analytics.protocols import DataWriter
from ob_analytics._registry import Registry

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
    """
    p = Path(path)
    if p.is_dir():
        result = {}
        for pq in sorted(p.glob("*.parquet")):
            result[pq.stem] = pd.read_parquet(pq)
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
        (default) and ``"pickle"``.  Additional formats (e.g.
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
        p.mkdir(parents=True, exist_ok=True)
        for name, df in lob_data.items():
            df.to_parquet(p / f"{name}.parquet", index=False)
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
