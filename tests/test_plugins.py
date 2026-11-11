"""The one entry-point plug-in loader every registry shares.

Each registry discovers plug-ins through ``ob_analytics._registry.load_plugins``
under its own entry-point group.  The "third-party package" is stood up by
monkeypatching ``importlib.metadata.entry_points`` so the tests stay offline
and need no real install.
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import pytest

from ob_analytics import _registry, data, metrics, sources

# ``ob_analytics.bars`` and ``ob_analytics.features`` are also the names of
# functions on the package, so these two modules are imported by their names.
from ob_analytics.bars import (
    ENTRY_POINT_GROUP as BAR_RULES_GROUP,
    get_bar_rule,
    load_bar_rule_plugins,
)
from ob_analytics.features import (
    ENTRY_POINT_GROUP as FEATURES_GROUP,
    get_feature,
    load_feature_plugins,
)
from ob_analytics.protocols import Level
from ob_analytics.visualization import _renderers

# Every test here registers a plug-in; put each registry back afterwards.
pytestmark = pytest.mark.usefixtures("restore_registries")


class FakePlugin:
    """A stand-in for a plug-in class, constructed with no arguments."""

    name = "fake_plugin"


class FakeMetric(FakePlugin):
    """A stand-in metric: register_metric refuses an object without every
    member of the Metric protocol."""

    title = "Fake"
    levels = (Level.L2, Level.L3)
    needs = ()

    def compute(self, result: Any) -> Any:
        return None

    def prepare(self, frame: Any) -> dict[str, Any]:
        return {}


def fake_writer_factory(config: Any, ctx: Any) -> None:
    """A stand-in writer factory."""


@dataclass(frozen=True)
class _Group:
    """One registry's entry-point group, its loader, and how to find a plug-in."""

    group: str
    load: Callable[..., list[str]]
    value: str
    registered: Callable[[], Any]
    expected: Callable[[Any], bool]


_GROUPS = [
    _Group(
        sources.ENTRY_POINT_GROUP,
        sources.load_source_plugins,
        f"{__name__}:FakePlugin",
        lambda: sources.get_source("fake_plugin"),
        lambda found: found is FakePlugin,
    ),
    _Group(
        metrics.ENTRY_POINT_GROUP,
        metrics.load_metric_plugins,
        f"{__name__}:FakeMetric",
        lambda: metrics.get_metric("fake_plugin"),
        lambda found: isinstance(found, FakeMetric),
    ),
    _Group(
        BAR_RULES_GROUP,
        load_bar_rule_plugins,
        f"{__name__}:FakePlugin",
        lambda: get_bar_rule("fake_plugin"),
        lambda found: isinstance(found, FakePlugin),
    ),
    _Group(
        FEATURES_GROUP,
        load_feature_plugins,
        f"{__name__}:FakePlugin",
        lambda: get_feature("fake_plugin"),
        lambda found: isinstance(found, FakePlugin),
    ),
    _Group(
        data.ENTRY_POINT_GROUP,
        data.load_writer_plugins,
        f"{__name__}:fake_writer_factory",
        lambda: data.WRITERS.get("fake_plugin"),
        lambda found: found is fake_writer_factory,
    ),
    _Group(
        _renderers.ENTRY_POINT_GROUP,
        _renderers.load_plot_backend_plugins,
        __name__,
        lambda: _renderers._BACKEND_MODULES["fake_plugin"],
        lambda found: found == __name__,
    ),
]


def _advertise(monkeypatch, group: str, *eps: EntryPoint) -> None:
    """Make *eps* the only entry points advertised under *group*."""

    def _entry_points(*, group: str | None = None) -> list[EntryPoint]:
        return [ep for ep in eps if ep.group == group]

    monkeypatch.setattr(_registry, "entry_points", _entry_points)


@pytest.mark.parametrize("spec", _GROUPS, ids=lambda spec: spec.group)
def test_each_registry_loads_a_plugin_from_its_group(monkeypatch, spec: _Group):
    _advertise(
        monkeypatch, spec.group, EntryPoint("fake_plugin", spec.value, spec.group)
    )

    assert spec.load(force=True) == ["fake_plugin"]
    assert spec.expected(spec.registered())


def test_every_group_is_declared_in_pyproject():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text())["project"]["entry-points"]

    assert {spec.group for spec in _GROUPS} <= set(declared)


def test_a_scanned_group_is_not_scanned_again_without_force(monkeypatch):
    group = sources.ENTRY_POINT_GROUP
    _advertise(
        monkeypatch, group, EntryPoint("fake_plugin", f"{__name__}:FakePlugin", group)
    )
    monkeypatch.setattr(_registry, "_loaded_groups", {group})

    assert sources.load_source_plugins() == []
    assert "fake_plugin" not in sources.list_sources()
    assert sources.load_source_plugins(force=True) == ["fake_plugin"]


def test_a_broken_plugin_is_skipped_with_a_warning(monkeypatch):
    group = sources.ENTRY_POINT_GROUP
    _advertise(
        monkeypatch,
        group,
        EntryPoint("broken", "no_such_module_for_ob_analytics:Nope", group),
        EntryPoint("fake_plugin", f"{__name__}:FakePlugin", group),
    )

    with pytest.warns(UserWarning, match="Skipped the plug-in 'broken'"):
        registered = sources.load_source_plugins(force=True)

    # The good one still loads.
    assert registered == ["fake_plugin"]
    assert "broken" not in sources.list_sources()
