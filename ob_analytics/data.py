"""Data I/O: Parquet serialization and writer registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from loguru import logger

from ob_analytics.protocols import DataWriter


# ── Writer registry ───────────────────────────────────────────────────

_WRITERS: dict[str, type] = {}


def register_writer(name: str, writer_cls: type) -> None:
    """Register a :class:`DataWriter` implementation under *name* for use
    with ``save_data(fmt=name)``.
    """
    _WRITERS[name.lower()] = writer_cls


def list_writers() -> list[str]:
    """Return a sorted list of registered writer names."""
    return sorted(_WRITERS)


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
        corresponding writer has been registered via
        :func:`register_writer`.
    writer : DataWriter, optional
        A pre-constructed writer instance.  When provided, *fmt* is
        ignored and the writer is used directly.  This is the preferred
        path when saving from a :class:`Pipeline` that already holds a
        configured writer.
    **write_kwargs
        Extra keyword arguments forwarded to ``writer.write()``.
    """
    p = Path(path)

    if writer is not None:
        writer.write(lob_data, p, **write_kwargs)
        return

    if fmt == "lobster":
        raise ValueError(
            "LOBSTER write requires a configured writer. "
            "Use LobsterFormat(trading_date=...).create_writer(config) "
            "or pass writer= directly to save_data()."
        )

    if fmt in _WRITERS:
        w = _WRITERS[fmt]()
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
        available = ["parquet", "pickle"] + sorted(_WRITERS)
        raise ValueError(
            f"Unsupported format: {fmt!r}. Available: {', '.join(available)}"
        )
