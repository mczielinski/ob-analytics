"""The matplotlib queue-position face draws like one line per order.

The face draws one line collection per fate instead of one line per order
(#539).  ``tests/test_plot_result.py`` checks the step paths.  These tests
check what a collection does differently from a line by default: caps and
joins, end markers, and the axes' data limits.
"""

from __future__ import annotations

from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pytest
from matplotlib import collections
from matplotlib.lines import Line2D

from ob_analytics.visualization._matplotlib import (
    _theme_rc,
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


def _line_collections(ax) -> list[collections.LineCollection]:
    return [c for c in ax.collections if isinstance(c, collections.LineCollection)]


def test_lines_use_line_caps_and_joins(data: dict) -> None:
    # A collection defaults to butt caps; ax.plot drew the theme's line caps.
    with mpl.rc_context(cast(Any, _theme_rc(DEFAULT_THEME))):
        line = Line2D([0, 1], [0, 1])
    cap, join = line.get_solid_capstyle(), line.get_solid_joinstyle()
    ax = mpl_queue_position_per_order(data).axes[0]
    lcs = _line_collections(ax)
    assert len(lcs) == 3
    for lc in lcs:
        assert lc.get_capstyle() == cap
        assert lc.get_joinstyle() == join


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
