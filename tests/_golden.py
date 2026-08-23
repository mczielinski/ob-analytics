"""Shared helpers for the golden-output correctness gates (issue #143).

A golden-output test pins the *exact numbers* a fixed input produces, so any
change that shifts a value fails loudly instead of slipping through a quick
check. :func:`df_fingerprint` turns a DataFrame into a stable content hash used
by those gates; :func:`book_fingerprint` does the same for an
:func:`~ob_analytics.analytics.order_book` snapshot (its two side frames).

The hash is deliberately dtype- and value-exact. Numeric and datetime columns
hash their raw IEEE-754 / int64 value buffers (native byte order), which is
stable across pandas / pyarrow versions and ~200x faster than a ``to_csv`` dump,
unlike CSV float formatting; object-like columns (categoricals, mixed id
columns) hash a string join. Two runs match only when every column name, every
dtype, and every value matches.

This module is a helper, not a test module: its name has no ``test_`` prefix so
pytest does not collect it.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd


def df_fingerprint(df: pd.DataFrame | None) -> str:
    """Return a stable content hash of *df* (column names + dtypes + values).

    Parameters
    ----------
    df : pandas.DataFrame or None
        The frame to fingerprint. ``None`` hashes to the literal ``"None"`` so a
        missing frame is distinguishable from an empty one.

    Returns
    -------
    str
        A hex SHA-256 digest that changes if any column name, dtype, or value
        changes.
    """
    if df is None:
        return "None"
    h = hashlib.sha256()
    h.update("|".join(map(str, df.columns)).encode())
    h.update("|".join(map(str, df.dtypes)).encode())
    for col in df.columns:
        arr = df[col].to_numpy()
        if arr.dtype == object:
            h.update("\x1f".join(map(str, arr.tolist())).encode())
        else:
            h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def book_fingerprint(book: dict[str, Any]) -> str:
    """Return a stable content hash of an ``order_book`` snapshot.

    Combines the ``bids`` and ``asks`` frame fingerprints, so one value locks
    both reconstructed sides of the book.

    Parameters
    ----------
    book : dict
        An :func:`~ob_analytics.analytics.order_book` result, carrying ``bids``
        and ``asks`` DataFrames.

    Returns
    -------
    str
        ``"<bids-digest>:<asks-digest>"``.
    """
    return f"{df_fingerprint(book['bids'])}:{df_fingerprint(book['asks'])}"
