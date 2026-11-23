"""The compiled loops behind the depth replay (#138).

The depth summary's own tests (``test_depth_engine.py``) check what the loops
compute.  These check how the module compiles them.
"""

from __future__ import annotations

from typing import Any

import pytest

from ob_analytics import _depth_replay


def test_compiles_without_the_disk_cache_when_it_cannot_write_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # numba raises RuntimeError at decoration when no cache folder can be
    # written, as in a read-only install with a read-only home folder.
    def njit(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("cache"):
            raise RuntimeError("cannot cache function: no locator available")
        (func,) = args
        return func

    monkeypatch.setattr(_depth_replay, "njit", njit)

    def add_one(x: int) -> int:
        return x + 1

    assert _depth_replay._compiled(add_one)(1) == 2
