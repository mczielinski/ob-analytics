"""The matplotlib queue-position face draws one line collection per fate.

The face used to call ``ax.plot`` once per order, which took about 25 s on the
bundled sample (#539).  These tests pin that the batched drawing keeps every
order's steps-post line, its colour and its end marker.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pytest
from matplotlib import collections
from matplotlib.cbook import pts_to_poststep

from ob_analytics.visualization._matplotlib import (
    _steps_post_paths,
    mpl_queue_position_per_order,
)
from ob_analytics.visualization._theme import DEFAULT_THEME

T0 = pd.Timestamp("2024-01-01 10:00", tz="UTC")


def _frame(spec: list[tuple[int, list[int]]]) -> pd.DataFrame:
    """Rows for each ``(id, ranks)``, interleaved by time across ids."""
    rows = [
        (oid, T0 + pd.Timedelta(seconds=3 * i + oid), rank)
        for oid, ranks in spec
        for i, rank in enumerate(ranks)
    ]
    df = pd.DataFrame(rows, columns=["id", "timestamp", "rank"])
    return df.sort_values("timestamp", kind="stable", ignore_index=True)


@pytest.fixture
def data() -> dict:
    return {
        # Order 9 has one row: no line, but it still gets an end marker.
        "filled": _frame([(1, [5, 4, 3, 1]), (2, [3, 2]), (9, [7])]),
        "cancelled": _frame([(3, [6, 6, 5]), (5, [8, 7, 7, 4, 2])]),
        "resting": _frame([(6, [4, 3])]),
        "max_rank": 9,
        "show_markers": True,
    }


def _expected_paths(side: pd.DataFrame) -> list[np.ndarray]:
    """What one ``ax.plot(..., drawstyle="steps-post")`` per order draws."""
    paths = []
    for _, g in side.groupby("id", sort=False):
        x, y = pts_to_poststep(
            mdates.date2num(g["timestamp"]), g["rank"].to_numpy(dtype=float)
        )
        paths.append(np.column_stack((x, y)))
    return paths


def test_one_collection_per_fate_with_per_order_steps(data: dict) -> None:
    ax = mpl_queue_position_per_order(data).axes[0]
    lcs = [c for c in ax.collections if isinstance(c, collections.LineCollection)]
    assert ax.get_lines() == []  # no per-order Line2D any more

    pal = DEFAULT_THEME.palette
    fates = [
        (data["filled"], pal.filled),
        (data["cancelled"], pal.cancelled),
        (data["resting"], pal.partial),
    ]
    assert len(lcs) == len(fates)
    for lc, (side, color) in zip(lcs, fates):
        got = [np.asarray(p.vertices) for p in lc.get_paths()]
        want = _expected_paths(side)
        assert len(got) == len(want)
        for g, w in zip(got, want):
            np.testing.assert_allclose(g, w)
        np.testing.assert_allclose(
            np.asarray(lc.get_edgecolor()), [mcolors.to_rgba(color, alpha=0.5)]
        )


def test_end_markers_sit_on_each_order_last_rank(data: dict) -> None:
    ax = mpl_queue_position_per_order(data).axes[0]
    markers = [
        c for c in ax.collections if not isinstance(c, collections.LineCollection)
    ]
    # Filled (×) and cancelled (○); still-resting orders have no end marker.
    assert len(markers) == 2
    for pc, side in zip(markers, (data["filled"], data["cancelled"])):
        ends = side.groupby("id", sort=False).tail(1)
        np.testing.assert_allclose(
            np.asarray(pc.get_offsets()),
            np.column_stack(
                (mdates.date2num(ends["timestamp"]), ends["rank"].to_numpy(float))
            ),
        )


def test_data_limits_match_per_order_plot(data: dict) -> None:
    # A one-row order draws nothing, but ax.plot still counted it in the
    # axes' data limits, so the x-range must not shrink without it.
    ax = mpl_queue_position_per_order(data).axes[0]
    x = mdates.date2num(
        pd.concat([data["filled"], data["cancelled"], data["resting"]])["timestamp"]
    )
    np.testing.assert_allclose(ax.dataLim.intervalx, [x.min(), x.max()])


def test_steps_post_paths_empty() -> None:
    empty = np.array([])
    assert _steps_post_paths(empty, empty, empty) == []
