"""The order-book engine's boundary holds (issue #136).

Issue #136 asks for three things, and this module checks each one:

* the engine is **its own module** with a clear input and output, and imports
  no pandas;
* the layers above it reach it only through that interface, never into its
  internals;
* what crosses the boundary is the shared schema (issue #112) in array form —
  events in, book states, lifecycles, and queue positions out.

The import checks are static: they parse the engine's source rather than
importing it, because ``ob_analytics.engine`` cannot be imported without
``ob_analytics/__init__.py`` running first and pulling pandas in for the rest of
the library.  Parsing answers the question that actually matters — *what does
the engine's own code depend on* — and answers it for every module in the
package, not just the ones a given test happens to exercise.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

from ob_analytics import engine
from ob_analytics.engine import Action, Direction, OrderEvents, Outcome

_ENGINE_DIR = pathlib.Path(engine.__file__).parent
_PACKAGE_DIR = _ENGINE_DIR.parent

# Everything the engine is allowed to import.  numpy is the one third-party
# dependency; the rest is the standard library plus the engine's own modules.
_ALLOWED_ROOTS = frozenset(
    {"__future__", "collections", "dataclasses", "enum", "math", "numpy", "typing"}
)


def _imported_modules(path: pathlib.Path) -> set[str]:
    """Every module name *path* imports, absolute and relative alike."""
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # a relative import stays inside the package
                names.add("ob_analytics.engine")
            elif node.module:
                names.add(node.module)
    return names


def _engine_sources() -> list[pathlib.Path]:
    return sorted(_ENGINE_DIR.glob("*.py"))


class TestEngineIsSelfContained:
    """The engine depends on numpy and itself, and on nothing else."""

    def test_the_package_has_sources_to_check(self) -> None:
        # Guard the gate: a glob that matched nothing would pass vacuously.
        assert len(_engine_sources()) >= 2

    @pytest.mark.parametrize("source", _engine_sources(), ids=lambda p: p.name)
    def test_imports_no_pandas(self, source: pathlib.Path) -> None:
        assert not [
            name
            for name in _imported_modules(source)
            if name == "pandas" or name.startswith("pandas.")
        ], f"{source.name} imports pandas; the engine must stay pandas-free"

    @pytest.mark.parametrize("source", _engine_sources(), ids=lambda p: p.name)
    def test_imports_only_numpy_and_itself(self, source: pathlib.Path) -> None:
        for name in _imported_modules(source):
            root = name.split(".")[0]
            if name.startswith("ob_analytics"):
                assert name.startswith("ob_analytics.engine"), (
                    f"{source.name} imports {name}: the engine must not depend "
                    "on the layers above it"
                )
            else:
                assert root in _ALLOWED_ROOTS, (
                    f"{source.name} imports {name}, which is not on the "
                    f"engine's allowed list {sorted(_ALLOWED_ROOTS)}"
                )


class TestCallersUseTheInterface:
    """The layers above reach the engine only through its public interface."""

    @pytest.mark.parametrize(
        "module",
        ["analytics.py", "queue.py", "_engine_frames.py", "visualization/_data.py"],
    )
    def test_no_reach_into_engine_internals(self, module: str) -> None:
        private = [
            name
            for name in _imported_modules(_PACKAGE_DIR / module)
            if name.startswith("ob_analytics.engine.")
        ]
        assert not private, (
            f"{module} imports {private}: use the ob_analytics.engine "
            "interface, not its private submodules"
        )

    def test_only_the_adapter_builds_engine_input(self) -> None:
        # One place in the library turns DataFrames into engine arrays.  If a
        # second appears, the conversion has started to spread and the
        # boundary will drift.
        builders = [
            path.name
            for path in sorted(_PACKAGE_DIR.rglob("*.py"))
            if _ENGINE_DIR not in path.parents and "OrderEvents(" in path.read_text()
        ]
        assert builders == ["_engine_frames.py"]


# ── The interface itself ──────────────────────────────────────────────


def _stream() -> OrderEvents:
    """Two bids and an ask: one filled, one cancelled, one still resting."""
    second = 1_000_000_000
    return OrderEvents(
        #        A join  B join  A fill  C join  B cancel
        order_id=np.array([1, 2, 1, 3, 2], dtype=np.int64),
        timestamp=np.array([0, 1, 2, 3, 4], dtype=np.int64) * second,
        price=np.array([100, 100, 100, 101, 100], dtype=np.int64),
        volume=np.array([5.0, 3.0, 0.0, 7.0, 3.0], dtype=np.float64),
        direction=np.array(
            [Direction.BID, Direction.BID, Direction.BID, Direction.ASK, Direction.BID],
            dtype=np.int8,
        ),
        action=np.array(
            [
                Action.CREATED,
                Action.CREATED,
                Action.CHANGED,
                Action.CREATED,
                Action.DELETED,
            ],
            np.int8,
        ),
        fill=np.array([0.0, 0.0, 5.0, 0.0, 0.0], dtype=np.float64),
    )


class TestEngineInterface:
    """Arrays in, arrays out — no pandas object crosses the boundary."""

    def test_book_state_reports_rows_and_derived_columns(self) -> None:
        events = _stream()
        book = engine.book_state(events, at=events.timestamp[-1])

        # A filled and a cancelled order have left; only C is still resting.
        assert len(book.bids) == 0
        assert list(book.asks.row) == [3]
        assert events.order_id[book.asks.row].tolist() == [3]
        for array in (book.asks.row, book.asks.liquidity, book.asks.bps):
            assert isinstance(array, np.ndarray)
        assert book.asks.liquidity.tolist() == [7.0]
        assert book.asks.bps.tolist() == [0.0]  # the touch is its own reference

    def test_book_state_is_a_point_in_time(self) -> None:
        events = _stream()
        early = engine.book_state(events, at=events.timestamp[1])
        assert events.order_id[early.bids.row].tolist() == [1, 2]
        assert early.bids.liquidity.tolist() == [5.0, 8.0]

    def test_lifecycles_report_one_row_per_order(self) -> None:
        life = engine.order_lifecycles(_stream())
        assert life.order_id.tolist() == [1, 2, 3]
        assert life.created_row.tolist() == [0, 1, 3]
        assert life.filled_vol.tolist() == [5.0, 0.0, 0.0]
        assert life.outcome.tolist() == [
            Outcome.FILLED,
            Outcome.CANCELLED,
            Outcome.RESTING,
        ]

    def test_queue_positions_report_fifo_rank(self) -> None:
        positions = engine.queue_positions(_stream(), touch_only=False)
        # B joins behind A, then reaches the front once A fills.
        b_rows = positions.rank[np.isin(positions.row, [1, 4])]
        assert b_rows.tolist() == [2, 1]
        assert positions.action[-1] == Action.DELETED

    def test_queue_age_grid_snapshots_the_touch(self) -> None:
        events = _stream()
        grid = engine.queue_age_grid(
            events, side=Direction.BID, at=events.timestamp[[0, 1, 2]]
        )
        assert grid.max_rank == 2
        assert grid.ages.shape == (2, 3)
        assert grid.ages[0, 1] == 1.0  # A is a second old when B joins
        assert np.isnan(grid.ages[1, 0])  # nothing behind A yet

    def test_events_reject_ragged_columns(self) -> None:
        with pytest.raises(ValueError, match="unequal lengths"):
            OrderEvents(
                order_id=np.array([1, 2], dtype=np.int64),
                timestamp=np.array([0], dtype=np.int64),
                price=np.array([100], dtype=np.int64),
                volume=np.array([1.0], dtype=np.float64),
                direction=np.array([Direction.BID], dtype=np.int8),
                action=np.array([Action.CREATED], dtype=np.int8),
            )

    def test_filled_volume_is_compensated(self) -> None:
        # A large placement followed by many small fills is where a plain
        # running total drifts in the last bits.  The engine compensates, so
        # the executed total is the correctly accumulated one -- and the
        # outcome derived from it does not wobble.
        second = 1_000_000_000
        n = 400
        small = np.full(n, 1e-3)
        events = OrderEvents(
            order_id=np.full(n + 1, 1, dtype=np.int64),
            timestamp=np.arange(n + 1, dtype=np.int64) * second,
            price=np.full(n + 1, 100, dtype=np.int64),
            volume=np.r_[1e8, 1e8 - np.cumsum(small)],
            direction=np.full(n + 1, Direction.BID, dtype=np.int8),
            action=np.array([Action.CREATED] + [Action.CHANGED] * n, dtype=np.int8),
            fill=np.r_[0.0, small],
        )
        life = engine.order_lifecycles(events)

        drifting = float(np.sum(small))  # the naive running total
        compensated = 0.0
        carry = 0.0
        for value in small.tolist():  # Kahan, spelled out
            corrected = value - carry
            stepped = compensated + corrected
            carry = (stepped - compensated) - corrected
            compensated = stepped

        assert life.filled_vol[0] == compensated
        assert compensated != drifting  # the two really do part company here

    def test_fill_tolerance_is_the_callers_to_set(self) -> None:
        # An order left a hair short of its placed size reads as filled under
        # the default tolerance and as partial under a stricter one.
        second = 1_000_000_000
        events = OrderEvents(
            order_id=np.array([9, 9], dtype=np.int64),
            timestamp=np.array([0, 1], dtype=np.int64) * second,
            price=np.array([100, 100], dtype=np.int64),
            volume=np.array([1.0, 0.0], dtype=np.float64),
            direction=np.full(2, Direction.BID, dtype=np.int8),
            action=np.array([Action.CREATED, Action.DELETED], dtype=np.int8),
            fill=np.array([0.0, 1.0 - 1e-12], dtype=np.float64),
        )
        default = engine.order_lifecycles(events)
        strict = engine.order_lifecycles(events, fill_tolerance=0.0)
        assert default.outcome.tolist() == [Outcome.FILLED]
        assert strict.outcome.tolist() == [Outcome.PARTIAL]

    def test_visible_rows_skip_hidden_orders(self) -> None:
        # The hidden-order rule lives on the input type, so the engine and any
        # caller sizing a window over the same stream cannot disagree.
        second = 1_000_000_000
        events = OrderEvents(
            order_id=np.array([0, 4, 0, 5], dtype=np.int64),
            timestamp=np.arange(4, dtype=np.int64) * second,
            price=np.full(4, 100, dtype=np.int64),
            volume=np.ones(4, dtype=np.float64),
            direction=np.array(
                [Direction.BID, Direction.BID, Direction.ASK, Direction.ASK], np.int8
            ),
            action=np.full(4, Action.CREATED, dtype=np.int8),
        )
        assert events.visible().tolist() == [1, 3]
        assert events.visible(side=Direction.BID).tolist() == [1]
        assert events.visible(side=Direction.ASK).tolist() == [3]

    def test_codes_and_labels_cannot_drift(self) -> None:
        # The schema strings are derived from the code names, so there is no
        # second list to fall out of step.
        assert engine.DIRECTIONS == ("bid", "ask")
        assert engine.ACTIONS == ("created", "changed", "deleted")
        assert engine.OUTCOMES == ("resting", "filled", "partial", "cancelled")
        for vocabulary in (Direction, Action, Outcome):
            labels = vocabulary.labels()
            assert [labels[int(m)] for m in vocabulary] == [m.label for m in vocabulary]

    def test_lifecycles_need_the_fill_column(self) -> None:
        events = _stream()
        without_fill = OrderEvents(
            order_id=events.order_id,
            timestamp=events.timestamp,
            price=events.price,
            volume=events.volume,
            direction=events.direction,
            action=events.action,
        )
        with pytest.raises(ValueError, match="fill is required"):
            engine.order_lifecycles(without_fill)
