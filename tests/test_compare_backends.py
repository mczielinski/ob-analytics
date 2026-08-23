"""Backend-equivalence gates (issue #143), scaffolded ahead of the backends.

Issue #143 asks for compare tests that prove a rewritten backend produces the
*same numbers* as the current one:

* **pandas vs Polars** — when analytics move to Polars behind Narwhals (#104), a
  function handed a Polars frame must return the same values as the pandas path.
* **Python vs compiled** — when the depth / order-book engine gains a compiled
  backend (numba, then maybe Rust) (#138), it must reproduce the pure-Python
  engine's output bit-for-bit (or within a tight tolerance).

Neither backend exists yet, and Polars is deliberately **not** a dependency of
ob-analytics (see ``docs/schema.md``). So each gate is guarded twice: the
optional library must import, and ob-analytics must advertise the backend. Until
both hold, the gate skips with a message naming the issue that will switch it
on. The comparison bodies are real, so the day a backend lands the only change
needed is to make the probe below see it — the assertions already state what
"same numbers" means.

Activation contract
-------------------
When the backends land, satisfy these hooks (adjust the probes here if the final
design differs; the intent — one input, both backends, identical output — does
not change):

* **Polars path (#104)** sets ``ob_analytics.analytics.SUPPORTS_POLARS = True``
  and makes the compared functions Narwhals-native: a pandas frame in returns a
  pandas frame, a Polars frame in returns a Polars frame, and the values agree.
* **Compiled engine (#138)** exposes a factory
  ``ob_analytics.depth.make_engine(backend="compiled")`` returning a
  :class:`~ob_analytics.depth.DepthMetricsEngine`-compatible engine.
"""

from __future__ import annotations

import importlib
from typing import Any

import pandas as pd
import pytest

from ob_analytics import analytics
from ob_analytics.depth import DepthMetricsEngine, depth_metrics, price_level_volume
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


# ── pandas vs Polars (#104) ───────────────────────────────────────────────


def _require_polars_analytics() -> Any:
    """Return the ``polars`` module iff the Polars analytics path (#104) is live.

    Two gates, both required: ``polars`` must import (it is not a dependency),
    and ob-analytics must advertise the Narwhals path via
    ``analytics.SUPPORTS_POLARS``.
    """
    pl = pytest.importorskip("polars", reason="polars is not a dependency (#104)")
    if not getattr(analytics, "SUPPORTS_POLARS", False):
        pytest.skip("Polars analytics backend (#104) not implemented yet")
    return pl


def _as_pandas(frame: Any) -> pd.DataFrame:
    """Return *frame* as pandas, whether it is a pandas or a Polars frame."""
    to_pandas = getattr(frame, "to_pandas", None)
    return to_pandas() if callable(to_pandas) else frame


class TestPandasVsPolars:
    """Analytics must return the same values on a Polars frame as on pandas.

    The comparison is exact to :data:`_RTOL` / :data:`_ATOL`; categorical and
    dtype checks are relaxed because Polars encodes categoricals and integers
    differently from pandas while carrying the same values.
    """

    def test_price_level_volume_matches(self) -> None:
        pl = _require_polars_analytics()
        events = _fixed_session().events

        reference = price_level_volume(events)
        polars_out = price_level_volume(pl.from_pandas(events))

        pd.testing.assert_frame_equal(
            reference.reset_index(drop=True),
            _as_pandas(polars_out).reset_index(drop=True),
            check_dtype=False,
            check_categorical=False,
            check_exact=False,
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_depth_metrics_matches(self) -> None:
        pl = _require_polars_analytics()
        depth = price_level_volume(_fixed_session().events)

        reference = depth_metrics(depth)
        polars_out = depth_metrics(pl.from_pandas(depth))

        pd.testing.assert_frame_equal(
            reference.reset_index(drop=True),
            _as_pandas(polars_out).reset_index(drop=True),
            check_dtype=False,
            check_categorical=False,
            check_exact=False,
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_order_flow_imbalance_matches(self) -> None:
        pl = _require_polars_analytics()
        from ob_analytics.flow_toxicity import order_flow_imbalance

        trades = _fixed_session().trades

        reference = order_flow_imbalance(trades)
        polars_out = order_flow_imbalance(pl.from_pandas(trades))

        pd.testing.assert_frame_equal(
            reference.reset_index(drop=True),
            _as_pandas(polars_out).reset_index(drop=True),
            check_dtype=False,
            check_categorical=False,
            check_exact=False,
            rtol=_RTOL,
            atol=_ATOL,
        )


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
