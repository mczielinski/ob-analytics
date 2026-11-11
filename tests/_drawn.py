"""What a figure drew, for tests of the plot renderers.

A renderer test that only checks the type of the figure passes when the
renderer draws nothing. :func:`drawn` counts the data points each kind of mark
holds, so a test can say what must be on the figure.
"""

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


def drawn(fig: Any) -> Counter[str]:
    """Count the data points *fig* draws, by kind of mark.

    Works on a matplotlib ``Figure``, a plotly ``Figure`` and a bokeh model
    (a figure or a layout of figures). The keys are the backend's own names
    for its marks, in lower case: ``line``, ``pathcollection``, ``quadmesh``,
    ``rectangle`` and ``image`` for matplotlib; the trace type (``heatmap``,
    ``bar``), with its mode when it has one (``scatter markers``, ``scattergl
    lines``), and ``shape`` for plotly; the glyph class
    (``scatter``, ``quad``, ``image``, ``segment``) for bokeh. A figure that
    shows only text, such as a "no data" figure, draws nothing: the counter is
    empty.

    Text, axes, legends and colour bars are not counted.
    """
    from matplotlib.figure import Figure as MplFigure

    if isinstance(fig, MplFigure):
        return _matplotlib(fig)
    module = type(fig).__module__
    if module.startswith("plotly"):
        return _plotly(fig)
    if module.startswith("bokeh"):
        return _bokeh(fig)
    raise TypeError(f"drawn() does not know a {type(fig).__name__}")


def _add(counts: Counter[str], kind: str, points: int) -> None:
    if points:
        counts[kind] += points


def _matplotlib(fig: Any) -> Counter[str]:
    from matplotlib.collections import QuadMesh

    counts: Counter[str] = Counter()
    for ax in fig.axes:
        if getattr(ax, "_colorbar", None) is not None:
            continue
        for line in ax.lines:
            _add(counts, "line", len(line.get_xdata()))
        for collection in ax.collections:
            kind = type(collection).__name__.lower()
            if isinstance(collection, QuadMesh):
                array = collection.get_array()
                _add(counts, kind, 0 if array is None else np.ma.count(array))
            elif len(collection.get_offsets()) > 1 or kind == "pathcollection":
                _add(counts, kind, len(collection.get_offsets()))
            else:
                _add(counts, kind, len(collection.get_paths()))
        for patch in ax.patches:
            _add(counts, type(patch).__name__.lower(), 1)
        for image in ax.images:
            _add(counts, "image", np.asarray(image.get_array()).size)
    return counts


def _plotly(fig: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    for trace in fig.data:
        values = next(
            (
                getattr(trace, name)
                for name in ("z", "x", "y", "values")
                if getattr(trace, name, None) is not None
            ),
            (),
        )
        # A gap (None or NaN) is not a point: plotly draws many stems as one
        # line trace with a gap after each stem.
        points = sum(not pd.isna(v) for v in np.asarray(values, dtype=object).ravel())
        mode = getattr(trace, "mode", None)
        _add(counts, f"{trace.type} {mode}" if mode else trace.type, points)
    _add(counts, "shape", len(fig.layout.shapes))
    return counts


def _bokeh(fig: Any) -> Counter[str]:
    from bokeh.models import GlyphRenderer

    counts: Counter[str] = Counter()
    for renderer in fig.select({"type": GlyphRenderer}):
        kind = type(renderer.glyph).__name__.lower()
        data = renderer.data_source.data
        if "image" in data:
            points = sum(np.asarray(a).size for a in data["image"])
        else:
            points = max((len(column) for column in data.values()), default=0)
        _add(counts, kind, points)
    return counts
