"""Check that a core-only install of the package works.

Run it with the Python of an environment that has the wheel installed with
no extras. The ``-I`` flag keeps the checkout off ``sys.path``, so Python
imports the installed package and not the source tree:

    python -I scripts/smoke_wheel.py

It runs the bundled sample through the pipeline, draws one Matplotlib plot,
and checks that asking for a Plotly or Bokeh figure without its extra raises
a ``ConfigError`` that names the extra to install. It exits non-zero on the
first check that fails.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure

import ob_analytics
from ob_analytics import ConfigError, Pipeline, sample_csv_path

# The backends an extra adds, the module each one imports, and the extra's
# name. A core-only install must have none of these modules.
EXTRAS = {
    "plotly": ("plotly", "interactive"),
    "bokeh": ("bokeh", "bokeh"),
}


def _fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def check_install() -> None:
    """The package comes from site-packages, and no extra is installed."""
    location = Path(ob_analytics.__file__).resolve()
    if "site-packages" not in location.parts:
        _fail(f"ob_analytics was imported from {location}, not an installed wheel")
    for module, extra in EXTRAS.values():
        if importlib.util.find_spec(module) is not None:
            _fail(
                f"{module} is installed, so this is not a core-only install ({extra})"
            )
    print(f"ok: ob_analytics {ob_analytics.__version__} from {location.parent}")


def main() -> None:
    check_install()

    result = Pipeline().run(sample_csv_path())
    if result.events.empty or result.trades.empty or result.depth_summary.empty:
        _fail("the sample run returned an empty frame")
    print(f"ok: sample run, {len(result.events)} events, {len(result.trades)} trades")

    figure = result.plot("trade_tape", "L2")
    if not isinstance(figure, Figure):
        _fail(f"the Matplotlib plot returned {type(figure).__name__}, not a Figure")
    print("ok: Matplotlib plot")

    for backend, (_, extra) in EXTRAS.items():
        try:
            result.plot("trade_tape", "L2", backend=backend)
        except ConfigError as exc:
            if f"ob-analytics[{extra}]" not in str(exc):
                _fail(f"the {backend} error does not name the extra: {exc}")
            print(f"ok: {backend} without its extra raises ConfigError")
        else:
            _fail(f"the {backend} plot worked without the {extra} extra")

    print("smoke test passed")


if __name__ == "__main__":
    main()
