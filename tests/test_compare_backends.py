"""Backend-equivalence gate for a compiled depth engine.

A faster engine must give the same numbers as the one it replaces. When the
depth engine gains a compiled backend (numba, then perhaps Rust), it must
reproduce the pure-Python engine's output, to rounding, on the same input.

No compiled backend exists yet, so each test skips with a message saying so.
The comparison bodies are real: the day the backend lands, the only change
needed is for the probe below to find it. The assertions already state what
"same numbers" means.

Activation contract
-------------------
The compiled engine is exposed through a factory,
``ob_analytics.depth.make_engine(backend="compiled")``, that returns an engine
compatible with :class:`~ob_analytics.depth.DepthMetricsEngine`. Adjust the
probe if the final design differs; the intent (one input, both engines, the
same output) does not change.

There is no pandas-vs-Polars gate. The public API takes and returns pandas,
and Polars is not a dependency (see ``adr/0002-dataframe-library.md``). A
Polars user converts at the boundary with ``PipelineResult.to_polars()`` or
reads the Parquet output directly. If analytics code is ever ported to Polars
inside the package, ADR 0002 requires the golden tests (such as
``tests/test_golden_synth.py``) to pass byte for byte. That is the check, so
this module has no Polars test.
"""

from __future__ import annotations

import importlib
from typing import Any

import pandas as pd
import pytest

from ob_analytics.depth import DepthMetricsEngine, price_level_volume
from ob_analytics.synth import SynthConfig, generate_session

# Tight numeric tolerance: a backend swap may reorder floating-point sums, so
# "same numbers" means equal to rounding, not necessarily bit-identical.
_RTOL = 1e-9
_ATOL = 1e-9


def _fixed_session(seed: int = 921, duration: float = 45.0):
    """A seeded L3 session, generated on demand (only when a gate is active).

    Kept out of module scope so a skipped gate costs nothing: the generator
    runs only after the probe has confirmed the backend is present.
    """
    return generate_session(SynthConfig(seed=seed, duration=duration))


# ── Python vs compiled engine (#138) ──────────────────────────────────────


def _require_compiled_depth_engine() -> Any:
    """Return a compiled :class:`DepthMetricsEngine` iff #138 has added one.

    Probes the documented factory ``ob_analytics.depth.make_engine(
    backend="compiled")``. Skips cleanly when the factory is absent (the state
    today) or when it declines the compiled backend on this platform.
    """
    depth_mod = importlib.import_module("ob_analytics.depth")
    factory = getattr(depth_mod, "make_engine", None)
    if factory is None:
        pytest.skip("compiled depth engine (#138) not implemented yet")
    try:
        return factory(backend="compiled")
    except (TypeError, ValueError, NotImplementedError, LookupError) as exc:
        pytest.skip(f"compiled depth engine backend unavailable: {exc}")


class TestPythonVsCompiledEngine:
    """The compiled engine must reproduce the pure-Python engine's output.

    The pure-Python :class:`DepthMetricsEngine` is the reference; the compiled
    engine (numba / Rust) must match it on the same input, across many seeded
    sessions and on a hand-built crossed stream that exercises the crossed-level
    eviction path.
    """

    @pytest.mark.parametrize("seed", [1, 7, 42, 143])
    def test_depth_summary_matches_across_seeds(self, seed: int) -> None:
        engine = _require_compiled_depth_engine()
        depth = price_level_volume(_fixed_session(seed=seed).events)

        reference = DepthMetricsEngine().compute(depth)
        compiled = engine.compute(depth)

        pd.testing.assert_frame_equal(
            reference,
            compiled,
            check_exact=False,
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_crossed_stream_matches(self) -> None:
        # A crossed / locked opposing quote drives the eviction branch — the
        # kind of edge case a backend rewrite is most likely to get subtly
        # wrong, so pin that both engines handle it identically.
        engine = _require_compiled_depth_engine()
        depth = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2026-01-01T00:00:00",
                        "2026-01-01T00:00:01",
                        "2026-01-01T00:00:02",
                        "2026-01-01T00:00:03",
                    ]
                ),
                "price": [100.00, 105.00, 101.00, 100.00],
                "volume": [0.17, 1.0, 1.0, 0.0],
                "direction": pd.Categorical(
                    ["ask", "ask", "bid", "ask"],
                    categories=["bid", "ask"],
                    ordered=True,
                ),
            }
        )

        reference = DepthMetricsEngine().compute(depth)
        compiled = engine.compute(depth)

        pd.testing.assert_frame_equal(
            reference,
            compiled,
            check_exact=False,
            rtol=_RTOL,
            atol=_ATOL,
        )
