"""One error type for one kind of mistake.

A bad value raises ``ConfigError`` (also a ``ValueError``); an unknown name
raises ``UnknownNameError`` (also a ``KeyError``).  Both are
``ObAnalyticsError``, so one ``except`` clause catches every mistake the
package reports.
"""

from __future__ import annotations

import ast
import pathlib
import pickle
from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

import ob_analytics as oa
from ob_analytics.exceptions import ConfigError, ObAnalyticsError, UnknownNameError
from ob_analytics.protocols import Level
from ob_analytics.queue import queue_positions
from ob_analytics.visualization import PlotTheme, plot
from ob_analytics.visualization._renderers import RendererRegistry

_PACKAGE = pathlib.Path(oa.__file__).parent


def _empty_trades() -> pd.DataFrame:
    return oa.toy_trades().iloc[:0]


# Each public entry point, called with one mistake, and the type it must raise.
_CASES: list[tuple[str, Callable[[], Any], type[ObAnalyticsError]]] = [
    ("get_source", lambda: oa.get_source("bitstmp"), UnknownNameError),
    ("get_metric", lambda: oa.get_metric("nope"), UnknownNameError),
    ("get_feature", lambda: oa.get_feature("nope"), UnknownNameError),
    ("get_bar_rule", lambda: oa.get_bar_rule("nope"), UnknownNameError),
    (
        "Pipeline.from_source",
        lambda: oa.Pipeline.from_source("nope"),
        UnknownNameError,
    ),
    (
        "save_data",
        lambda: oa.save_data({}, "unused", fmt="nope"),
        UnknownNameError,
    ),
    ("plot concept", lambda: plot("nope"), UnknownNameError),
    ("plot backend", lambda: plot("trade_tape", backend="nope"), UnknownNameError),
    ("PlotTheme style", lambda: PlotTheme(style="nope"), UnknownNameError),
    (
        "bars",
        lambda: oa.bars(oa.toy_trades(), "nope", threshold=1),
        UnknownNameError,
    ),
    (
        "features",
        lambda: oa.features(oa.toy_trades(), include=["nope"]),
        UnknownNameError,
    ),
    (
        "compute_kyle_lambda, empty trades",
        lambda: oa.compute_kyle_lambda(_empty_trades()),
        ConfigError,
    ),
    (
        "compute_kyle_lambda, bad ci_level",
        lambda: oa.compute_kyle_lambda(oa.toy_trades(), ci_level=2.0),
        ConfigError,
    ),
    (
        "vpin_bucket_volume",
        lambda: oa.vpin_bucket_volume(oa.toy_trades(), buckets_per_day=0),
        ConfigError,
    ),
    (
        "queue_positions",
        lambda: queue_positions(oa.toy_events(), levels="some"),
        ConfigError,
    ),
    (
        "PipelineConfig.price_multiplier",
        lambda: oa.PipelineConfig(tick_size=0.3).price_multiplier,
        ConfigError,
    ),
]


@pytest.mark.parametrize(
    ("call", "expected"),
    [pytest.param(call, expected, id=name) for name, call, expected in _CASES],
)
def test_every_entry_point_raises_a_package_error(
    call: Callable[[], Any], expected: type[ObAnalyticsError]
) -> None:
    with pytest.raises(ObAnalyticsError) as info:
        call()
    assert type(info.value) is expected


def test_config_error_is_a_value_error() -> None:
    assert issubclass(ConfigError, ValueError)
    assert issubclass(ConfigError, ObAnalyticsError)


def test_unknown_name_error_is_a_key_error() -> None:
    assert issubclass(UnknownNameError, KeyError)
    assert issubclass(UnknownNameError, ObAnalyticsError)


def test_unmapped_message_error_is_a_package_error() -> None:
    from ob_analytics.live import UnmappedMessageError

    assert issubclass(UnmappedMessageError, ObAnalyticsError)
    assert issubclass(UnmappedMessageError, ValueError)


class TestUnknownNameMessage:
    def test_names_close_matches(self) -> None:
        err = UnknownNameError("source", "bitstmp", ["bitstamp", "lobster"])
        assert str(err) == "Unknown source 'bitstmp'. Did you mean 'bitstamp'?"

    def test_match_ignores_case_and_keeps_the_known_spelling(self) -> None:
        err = UnknownNameError("style", "DARKGRID", ["darkgrid", "white"])
        assert str(err) == "Unknown style 'DARKGRID'. Did you mean 'darkgrid'?"

    def test_lists_a_short_registry_when_nothing_is_close(self) -> None:
        err = UnknownNameError("view", "zzz", ["l2", "l3"])
        assert str(err) == "Unknown view 'zzz'. Known: 'l2', 'l3'."

    def test_does_not_dump_a_long_registry(self) -> None:
        known = [f"venue_{i}" for i in range(100)]
        err = UnknownNameError("exchange", "zzz", known)
        assert str(err) == "Unknown exchange 'zzz'."

    def test_says_when_nothing_is_registered(self) -> None:
        err = UnknownNameError("widget", "a", [])
        assert str(err) == "Unknown widget 'a'. No widget is registered."

    def test_str_is_not_quoted_like_a_dict_key(self) -> None:
        err = UnknownNameError("widget", "a", ["b"])
        assert not str(err).startswith("'")

    def test_survives_pickling(self) -> None:
        err = UnknownNameError("source", "bitstmp", ["bitstamp"], hint="See X.")
        again = pickle.loads(pickle.dumps(err))
        assert type(again) is UnknownNameError
        assert str(again) == str(err)
        assert again.known == ["bitstamp"]


class TestRendererMissing:
    """A plot a backend does not draw names the backends that do draw it."""

    @pytest.fixture
    def registry(self) -> RendererRegistry:
        reg = RendererRegistry("renderer")
        for backend in ("matplotlib", "plotly"):
            reg.register(("depth_map", Level.L2, backend), lambda data: None)
        reg.register(("trade_tape", None, "matplotlib"), lambda data: None)
        for i in range(30):
            reg.register((f"other_{i}", None, "matplotlib"), lambda data: None)
        return reg

    def test_a_backend_that_does_not_draw_it(self, registry) -> None:
        with pytest.raises(UnknownNameError) as info:
            registry.get(("depth_map", Level.L2, "bokeh"))
        assert str(info.value) == (
            "Unknown 'bokeh' plot 'depth_map'. "
            "Backends that draw it: 'matplotlib', 'plotly'."
        )

    def test_a_level_the_backend_does_not_draw(self, registry) -> None:
        with pytest.raises(UnknownNameError) as info:
            registry.get(("depth_map", Level.L3, "matplotlib"))
        assert str(info.value) == (
            "Unknown L3 'matplotlib' plot 'depth_map'. On 'matplotlib' it is at L2."
        )

    def test_a_level_less_plot_asked_for_at_a_level(self, registry) -> None:
        with pytest.raises(UnknownNameError, match="On 'matplotlib' it is level-less"):
            registry.get(("trade_tape", Level.L2, "matplotlib"))

    def test_an_unknown_concept_names_close_matches_only(self, registry) -> None:
        with pytest.raises(UnknownNameError) as info:
            registry.get(("trade_tap", None, "matplotlib"))
        assert str(info.value) == (
            "Unknown plot concept 'trade_tap'. Did you mean 'trade_tape'?"
        )


# ---------------------------------------------------------------------------
# No plain ValueError or KeyError for a mistake the package reports
# ---------------------------------------------------------------------------

# Raises that report the package's own broken state, not a caller's mistake,
# by file and the function that raises.
_INTERNAL_RAISES = {
    # A circular reference, raised as json.dumps raises it.
    ("live/_runner.py", "_encode", "ValueError"),
    # A message helper asked to name no levels.
    ("visualization/_renderers.py", "_kind_text", "ValueError"),
}

# The base class says nothing about the kind of mistake, so it is never raised.
_PLAIN = ("ValueError", "KeyError", "ObAnalyticsError")


def _raises_in(tree: ast.AST) -> list[tuple[str, ast.Raise]]:
    """Each ``raise`` in *tree*, with the name of the function it is in."""
    found = []

    def visit(node: ast.AST, function: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, child.name)
            else:
                if isinstance(child, ast.Raise):
                    found.append((function, child))
                visit(child, function)

    visit(tree, "<module>")
    return found


def _plain_raises() -> list[tuple[str, str, int, str]]:
    found = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        rel = path.relative_to(_PACKAGE).as_posix()
        # The engine imports only numpy and itself, so it raises built-ins;
        # the public functions over it check their arguments first.
        if rel.startswith("engine/"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function, node in _raises_in(tree):
            if node.exc is None:
                continue
            exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            name = getattr(exc, "id", None)
            if name in _PLAIN and (rel, function, name) not in _INTERNAL_RAISES:
                found.append((rel, function, node.lineno, name))
    return found


def test_no_plain_value_key_or_base_error_is_raised() -> None:
    assert _plain_raises() == []


def test_every_allowed_internal_raise_still_exists() -> None:
    """An entry left behind after its raise is gone would hide a new one."""
    present = set()
    for rel, function, name in _INTERNAL_RAISES:
        tree = ast.parse((_PACKAGE / rel).read_text(encoding="utf-8"))
        for fn, node in _raises_in(tree):
            exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if fn == function and getattr(exc, "id", None) == name:
                present.add((rel, function, name))
    assert present == _INTERNAL_RAISES
