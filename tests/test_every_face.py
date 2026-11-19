"""Draw every face the concept table declares, on the bundled sample.

The other tests that loop over the concept table check names and titles, and
the per-plot tests draw one face each.  This module draws each coordinate
``(concept, level, backend)`` of :data:`~ob_analytics.visualization._concepts.PLOTS`
from one run of the bundled sample, so a face added to the table is drawn here
without a new test.  Each figure must pass three checks:

* it has marks (:func:`tests._drawn.drawn`), or its data says why it has
  none (:data:`_REASON_KEYS`) and the figure shows that text;
* its time and price axes lie inside the run: a time axis inside the run's
  span, give or take a tenth of its length; a price axis inside the traded
  prices, give or take their range once more;
* each size axis is in the instrument's units: what it draws is no larger than
  the sizes the renderer was given, which are in the instrument's units
  (:func:`~ob_analytics.visualization.gallery.display_result`).

An axis is told by its title: ``Time``; a title with the word ``price``; a
title that names a size, volume or liquidity.  An untitled axis is not
checked.  The range and unit checks read matplotlib and plotly figures; a bokeh
figure computes its ranges in the browser, so it is checked for marks only.

A face the gallery does not build on the sample (it needs LOBSTER data, or it
is drawn by ``plot()`` only) is bound in :data:`_EXTRA_SPECS`.  A face that
fails a check is listed in :data:`KNOWN_FAILURES` as a strict xfail, so the
marker has to go when the face is fixed.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pytest

from ob_analytics.bars import bars
from ob_analytics.pipeline import PipelineResult
from ob_analytics.protocols import Level
from ob_analytics.visualization import _concepts, _data, plot
from ob_analytics.visualization.gallery import (
    PlotSpec,
    bars_card,
    build_gallery_model,
    display_result,
    trading_halts_card,
)
from tests._drawn import drawn

Coordinate = tuple[str, Level | None, str]

# Faces that fail the range check, each with the reason.  Each is a strict
# xfail that expects :class:`OutOfRange` only: when the face is fixed the test
# passes and the strict marker fails the run, and when it fails another check
# (no marks, a size) the failure is reported.  Either way the entry must be
# looked at.  None fails today.
KNOWN_FAILURES: dict[Coordinate, str] = {}

#: Payload entries in which a face says why it draws no marks: the one value
#: of a measure with one window (``headline``), or too few trades to draw
#: (``too_few``).  Empty when the face draws.
_REASON_KEYS = ("headline", "too_few")


def _hidden_executions(display: PipelineResult) -> PlotSpec:
    # The sample has no hidden executions; the face still draws the trades.
    return PlotSpec(
        "hidden_executions",
        "Hidden Executions",
        "hidden_executions",
        _data.prepare_hidden_executions_data,
        {"events": display.events, "trades": display.trades},
    )


def _time_series(display: PipelineResult) -> PlotSpec:
    trades = display.trades
    return PlotSpec(
        "time_series",
        "Trade price",
        "time_series",
        _data.prepare_time_series_data,
        {
            "timestamp": trades["timestamp"],
            "series": trades["price"],
            "y_label": "Trade price",
        },
    )


#: How to bind the faces the gallery does not build on the bundled sample,
#: from the run in display units.
_EXTRA_SPECS: dict[str, Callable[[PipelineResult], PlotSpec]] = {
    "hidden_executions": _hidden_executions,
    "time_series": _time_series,
    "bars": lambda d: bars_card(bars(d.trades, "time")),
    "trading_halts": lambda d: trading_halts_card(
        d.trades, pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns, UTC]")})
    ),
}


class OutOfRange(AssertionError):
    """A time or price axis reaches outside the run."""


@dataclass
class Faces:
    """The bundled sample's run and how each face binds its data."""

    display: PipelineResult
    specs: dict[tuple[str, Level | None], PlotSpec]

    def payload(self, key: str, level: Level | None) -> dict:
        """The face's prepared data, made afresh so no backend sees another's."""
        spec = self.specs.get((key, level))
        if spec is None:
            pytest.fail(
                f"{key} at {level} is not built on the bundled sample: "
                "bind it in _EXTRA_SPECS"
            )
        return spec.prepare(**spec.prep_kwargs)


@pytest.fixture(scope="module")
def faces(bundled_sample_result: PipelineResult) -> Faces:
    model = build_gallery_model(bundled_sample_result)
    specs: dict[tuple[str, Level | None], PlotSpec] = {
        (concept.key, level): spec
        for concept in model.concepts
        for level, spec in concept.variants.items()
    }
    specs.update({(spec.plot_name, None): spec for spec in model.analytics})
    display = display_result(bundled_sample_result)
    levels = {(key, level) for key, level, _ in _concepts.PLOTS}
    for key, level in levels - specs.keys():
        if key in _EXTRA_SPECS:
            specs[key, level] = _EXTRA_SPECS[key](display)
    return Faces(display, specs)


# ── Reading the axes of a figure ──────────────────────────────────────

_PRICE = re.compile(r"\bprice\b", re.IGNORECASE)
_SIZE = re.compile(r"size|volume|liquidity", re.IGNORECASE)
# ``placed`` is the order outcome's size at placement.
_PAYLOAD_SIZE = re.compile(r"vol|size|liquidity|placed", re.IGNORECASE)


@dataclass
class Axis:
    """One axis of a figure: its title, what it shows and what it draws."""

    title: str
    view: tuple[Any, Any]  # the range shown
    marks: tuple[Any, Any] | None  # the range of the data drawn on it

    @property
    def is_time(self) -> bool:
        return self.title.strip().lower() == "time"

    @property
    def is_price(self) -> bool:
        return bool(_PRICE.search(self.title))

    @property
    def is_size(self) -> bool:
        return bool(_SIZE.search(self.title)) and not self.is_price


def _axes(fig: Any) -> list[Axis]:
    from matplotlib.figure import Figure

    if isinstance(fig, Figure):
        return list(_matplotlib_axes(fig))
    return list(_plotly_axes(fig))


def _as_times(values: tuple[float, float]) -> tuple[pd.Timestamp, pd.Timestamp]:
    lo, hi = (pd.Timestamp(mdates.num2date(v)) for v in values)
    return lo, hi


def _matplotlib_axes(fig: Any) -> Iterator[Axis]:
    for ax in fig.axes:
        if getattr(ax, "_colorbar", None) is not None:
            continue
        data = ax.dataLim
        for title, view, marks in (
            (ax.get_xlabel(), ax.get_xlim(), (data.x0, data.x1)),
            (ax.get_ylabel(), ax.get_ylim(), (data.y0, data.y1)),
        ):
            if not title:
                continue
            view = (min(view), max(view))
            marks = (min(marks), max(marks)) if np.isfinite(marks).all() else None
            axis = Axis(title, view, marks)
            if axis.is_time:
                axis.view = _as_times(view)
                axis.marks = None if marks is None else _as_times(marks)
            yield axis


def _plotly_values(trace: Any, letter: str) -> list[Any]:
    """The values *trace* draws along its *letter* (``x`` or ``y``) axis."""
    if trace.type == "candlestick" and letter == "y":
        return [v for name in ("low", "high") for v in getattr(trace, name)]
    values = getattr(trace, letter, None)
    if values is None:
        return []
    values = list(np.asarray(values, dtype=object).ravel())
    base = getattr(trace, "base", None)
    along = "x" if getattr(trace, "orientation", None) == "h" else "y"
    if trace.type == "bar" and base is not None and letter == along:
        bases = np.broadcast_to(np.asarray(base, dtype=object), (len(values),))
        values = [*values, *(b + v for b, v in zip(bases, values))]
    return [v for v in values if v is not None and not pd.isna(v)]


def _utc(value: Any) -> pd.Timestamp:
    # Plotly reads a time without a zone as UTC.
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp


def _extent(values: list[Any], time: bool) -> tuple[Any, Any] | None:
    """The range of *values*, or ``None`` when there are none or they are labels."""
    if not values:
        return None
    if time:
        stamps = pd.to_datetime(pd.Series(values), utc=True)
        return stamps.min(), stamps.max()
    try:
        numbers = np.asarray(values, dtype=float)
    except (TypeError, ValueError):  # a category axis
        return None
    return float(numbers.min()), float(numbers.max())


def _plotly_axes(fig: Any) -> Iterator[Axis]:
    layout = fig.layout
    for name in layout.to_plotly_json():
        if not name.startswith(("xaxis", "yaxis")):
            continue
        axis_layout = layout[name]
        title = axis_layout.title.text or ""
        if not title:
            continue
        letter = name[0]
        ref = letter + name.removeprefix(f"{letter}axis")
        values = [
            v
            for trace in fig.data
            if (getattr(trace, f"{letter}axis", None) or letter) == ref
            for v in _plotly_values(trace, letter)
        ]
        time = title.strip().lower() == "time"
        marks = _extent(values, time)
        view = marks
        if axis_layout.range is not None:
            lo, hi = axis_layout.range
            if time:
                view = (_utc(lo), _utc(hi))
            elif axis_layout.type == "log":
                view = (10.0**lo, 10.0**hi)
            else:
                view = (min(lo, hi), max(lo, hi))
        if view is not None:
            yield Axis(title, view, marks)


def _payload_size_max(payload: Any, named: bool = False) -> float:
    """The largest size, by absolute value, in a renderer's payload.

    A size is a column, series or entry whose name matches
    :data:`_PAYLOAD_SIZE`.  Returns 0 when the payload holds none.
    """
    best = 0.0

    def numbers(values: Any) -> float:
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            return 0.0
        array = array[np.isfinite(array)]
        return float(np.abs(array).max()) if array.size else 0.0

    if isinstance(payload, pd.DataFrame):
        for column in payload.columns:
            if _PAYLOAD_SIZE.search(str(column)):
                best = max(best, numbers(payload[column]))
    elif isinstance(payload, pd.Series):
        if named or _PAYLOAD_SIZE.search(str(payload.name)):
            best = numbers(payload)
    elif isinstance(payload, Mapping):
        for name, value in payload.items():
            best = max(
                best, _payload_size_max(value, bool(_PAYLOAD_SIZE.search(str(name))))
            )
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            best = max(best, _payload_size_max(value, named))
    elif named and isinstance(payload, np.ndarray):
        best = numbers(payload)
    return best


def _figure_text(fig: Any) -> str:
    """Every piece of text a matplotlib, plotly or bokeh figure shows, joined."""
    from matplotlib.figure import Figure
    from matplotlib.text import Text

    if isinstance(fig, Figure):
        return " ".join(t.get_text() for t in fig.findobj(Text))
    if hasattr(fig, "layout"):  # plotly
        parts = [fig.layout.title.text or ""]
        parts += [a.text or "" for a in fig.layout.annotations]
        # Plotly breaks a line with <br>.
        return " ".join(parts).replace("<br>", "\n")
    from bokeh.models import Label, Title

    models = [*fig.select({"type": Label}), *fig.select({"type": Title})]
    return " ".join(m.text or "" for m in models)


# ── The test ──────────────────────────────────────────────────────────


def _coordinates() -> list[Any]:
    params = []
    for coordinate in _concepts.PLOTS:
        key, level, backend = coordinate
        reason = KNOWN_FAILURES.get(coordinate)
        marks = (
            [pytest.mark.xfail(strict=True, raises=OutOfRange, reason=reason)]
            if reason
            else []
        )
        name = f"{key}-{level.value if level else 'none'}-{backend}"
        params.append(pytest.param(coordinate, id=name, marks=marks))
    return params


def test_known_failures_are_declared_faces() -> None:
    assert set(KNOWN_FAILURES) <= set(_concepts.PLOTS)


@pytest.mark.parametrize("coordinate", _coordinates())
def test_face_draws_inside_the_data(faces: Faces, coordinate: Coordinate) -> None:
    key, level, backend = coordinate
    if backend != "matplotlib":
        pytest.importorskip(backend)
    payload = faces.payload(key, level)
    fig = plot(key, level, backend=backend, **payload)

    if not drawn(fig):
        reason = next((payload[k] for k in _REASON_KEYS if payload.get(k)), "")
        assert reason, f"{key} at {level} drew no marks on {backend}, and no reason"
        assert reason in _figure_text(fig), f"{reason!r} is not on the figure"
        return
    if backend == "bokeh":
        return

    run = faces.display
    clock = run.events if not run.events.empty else run.depth
    start, end = clock["timestamp"].min(), clock["timestamp"].max()
    # Bars and fixed windows start on a whole minute, a little before the run.
    pad = (end - start) * 0.1
    low, high = run.trades["price"].min(), run.trades["price"].max()
    # The book faces reach past the traded prices into the resting book.
    spread = high - low
    sizes = _payload_size_max(payload)

    for axis in _axes(fig):
        if axis.is_time:
            lo, hi = axis.view
            if not (start - pad <= lo and hi <= end + pad):
                raise OutOfRange(
                    f"{axis.title!r} axis shows {lo} to {hi}; "
                    f"the run is {start} to {end}"
                )
        elif axis.is_price:
            lo, hi = axis.view
            if not (low - spread <= lo and hi <= high + spread):
                raise OutOfRange(
                    f"{axis.title!r} axis shows {lo} to {hi}; "
                    f"trades are {low} to {high}"
                )
        elif axis.is_size and axis.marks is not None:
            assert sizes > 0, f"{axis.title!r} axis, but the payload holds no size"
            drawn_max = max(abs(axis.marks[0]), abs(axis.marks[1]))
            assert drawn_max <= sizes * 1.5, (
                f"{axis.title!r} axis draws up to {drawn_max}; "
                f"the largest size given is {sizes}"
            )


# ── Every backend reads the same payload ──────────────────────────────


class _ReadLog(dict[str, Any]):
    """A payload that records each key a renderer reads from it."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        super().__init__(payload)
        self.read: set[str] = set()

    def __getitem__(self, key: Any, /) -> Any:
        self.read.add(str(key))
        return super().__getitem__(key)

    def get(self, key: Any, default: Any = None, /) -> Any:
        self.read.add(str(key))
        return super().get(key, default)

    def __contains__(self, key: object, /) -> bool:
        self.read.add(str(key))
        return super().__contains__(key)


def _faces_on_backends() -> list[Any]:
    """Each face drawn by more than one backend, with its backends."""
    backends: dict[tuple[str, Level | None], list[str]] = {}
    for key, level, backend in _concepts.PLOTS:
        backends.setdefault((key, level), []).append(backend)
    return [
        pytest.param(key, level, names, id=f"{key}-{level.value if level else 'none'}")
        for (key, level), names in backends.items()
        if len(names) > 1
    ]


@pytest.mark.parametrize(("key", "level", "backends"), _faces_on_backends())
def test_backends_read_the_same_payload(
    faces: Faces, key: str, level: Level | None, backends: list[str]
) -> None:
    # A renderer only draws what the payload holds (ARCHITECTURE.md).  When
    # one backend reads a key another ignores, the two draw different
    # pictures from one payload, or one computes what the other is given.
    import importlib.util

    import matplotlib.pyplot as plt

    from ob_analytics.visualization._renderers import RENDERERS, _load_backend

    # Compare the backends that are installed; skip only when fewer than two.
    backends = [b for b in backends if b == "matplotlib" or importlib.util.find_spec(b)]
    if len(backends) < 2:
        pytest.skip(f"{key} at {level}: fewer than two backends installed")
    read: dict[str, set[str]] = {}
    for backend in backends:
        _load_backend(backend)
        payload = _ReadLog(faces.payload(key, level))
        RENDERERS.get((key, level, backend))(payload)
        plt.close("all")
        read[backend] = payload.read
    every = set().union(*read.values())
    missing = {b: sorted(every - keys) for b, keys in read.items() if every - keys}
    assert not missing, f"{key} at {level}: keys a backend does not read: {missing}"
