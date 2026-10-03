"""Tests for ``Pipeline.run_windows``: one input, cut into time windows.

The claim under test is that a windowed run with the book carried across each
cut writes the same tables a single run gives, so cutting an input to bound
memory costs nothing in the output.  Each test compares against a single run
saved with ``save_data`` and read back with ``load_data``, because that is the
folder a windowed run is meant to be interchangeable with.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ob_analytics import Pipeline, sample_csv_path
from ob_analytics._windows import window_bounds
from ob_analytics.data import load_data, save_data
from ob_analytics.datasets import toy_l2_depth, toy_l2_trades
from ob_analytics.depth import DepthMetricsEngine, depth_metrics
from ob_analytics.depth_l2 import DepthCsvWriter
from ob_analytics.exceptions import ConfigError
from ob_analytics.synth import SyntheticLoader, SyntheticTradeSource, generate_session


def _cuts(timestamps: pd.Series, *quantiles: float) -> list[pd.Timestamp]:
    """Cut times at the given quantiles of *timestamps*, on whole seconds."""
    ordered = timestamps.sort_values().reset_index(drop=True)
    return [ordered.iloc[int(q * (len(ordered) - 1))].floor("s") for q in quantiles]


def _saved_single_run(pipeline: Pipeline, source, dest) -> dict[str, pd.DataFrame]:
    result = pipeline.run(source)
    save_data(result._frames(), dest, config=result.config)
    return load_data(dest)


def _assert_tables_equal(
    single: dict[str, pd.DataFrame],
    windowed: dict[str, pd.DataFrame],
    *,
    skip_event_columns: tuple[str, ...] = (),
) -> None:
    assert sorted(windowed) == ["depth", "depth_summary", "events", "trades"]
    for name in ("trades", "depth", "depth_summary"):
        pd.testing.assert_frame_equal(windowed[name], single[name], obj=name)
    # A windowed run writes its events window by window, so compare them in
    # event_id order.
    a, b = (
        frame.drop(columns=list(skip_event_columns))
        .sort_values("event_id")
        .reset_index(drop=True)
        for frame in (single["events"], windowed["events"])
    )
    pd.testing.assert_frame_equal(b, a, obj="events")


# ── L3 ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def synth_session():
    return generate_session(seed=7, duration=900.0)


def _synth_pipeline(session) -> Pipeline:
    return Pipeline(
        loader=SyntheticLoader(session),
        trade_source=SyntheticTradeSource(session),
    )


def test_windowed_run_matches_a_single_run(synth_session, tmp_path):
    single = _saved_single_run(
        _synth_pipeline(synth_session), None, tmp_path / "single"
    )
    cuts = _cuts(single["events"]["timestamp"], 0.2, 0.5, 0.8)

    out = _synth_pipeline(synth_session).run_windows(None, cuts, tmp_path / "w")

    assert out == tmp_path / "w"
    _assert_tables_equal(single, load_data(out))


def test_bitstamp_windowed_run_matches_except_aggressiveness(tmp_path):
    # Bitstamp numbers its events by order, not by time, and aggressiveness
    # looks up the standing quote by event_id, so that one column can read a
    # quote from another window.  Everything else must match.
    single = _saved_single_run(Pipeline(), sample_csv_path(), tmp_path / "single")
    cuts = _cuts(single["events"]["timestamp"], 0.5)

    out = Pipeline().run_windows(sample_csv_path(), cuts, tmp_path / "w")

    _assert_tables_equal(
        single, load_data(out), skip_event_columns=("aggressiveness_bps",)
    )


def test_without_carry_each_window_starts_from_an_empty_book(synth_session, tmp_path):
    single = _saved_single_run(
        _synth_pipeline(synth_session), None, tmp_path / "single"
    )
    cut = _cuts(single["events"]["timestamp"], 0.5)[0]

    out = _synth_pipeline(synth_session).run_windows(
        None, [cut], tmp_path / "w", carry=False
    )
    windowed = load_data(out)

    # Nothing changes before the cut.  After it the orders resting at the cut
    # are missing: their later changes reach no price level, and the book is
    # thinner than a single run's.
    single_summary, windowed_summary = (
        tables["depth_summary"] for tables in (single, windowed)
    )
    single_after = single_summary["timestamp"] >= cut
    windowed_after = windowed_summary["timestamp"] >= cut
    pd.testing.assert_frame_equal(
        windowed_summary.loc[~windowed_after], single_summary.loc[~single_after]
    )
    assert windowed_after.sum() < single_after.sum()
    volume = [c for c in single_summary.columns if "_vol" in c]
    assert (
        windowed_summary.loc[windowed_after, volume].iloc[0].sum()
        < single_summary.loc[single_after, volume].iloc[0].sum()
    )
    # Every row is still written once.
    assert len(windowed["events"]) == len(single["events"])
    assert len(windowed["trades"]) == len(single["trades"])


def test_a_window_with_no_rows_is_skipped(synth_session, tmp_path):
    single = _saved_single_run(
        _synth_pipeline(synth_session), None, tmp_path / "single"
    )
    first = single["events"]["timestamp"].min()
    # Two cuts before the data starts: the first two windows are empty.
    cuts = [first - pd.Timedelta(hours=2), first - pd.Timedelta(hours=1)]

    out = _synth_pipeline(synth_session).run_windows(None, cuts, tmp_path / "w")

    _assert_tables_equal(single, load_data(out))


# ── L2 ────────────────────────────────────────────────────────────────


@pytest.fixture
def toy_l2_dir(tmp_path):
    """The bundled toy L2 fixture, with the trades' sides left out.

    No sides, so the run has to classify every trade and the test covers the
    trade signs too.
    """
    trades = toy_l2_trades()
    trades["direction"] = pd.NA
    DepthCsvWriter().write({"depth": toy_l2_depth(), "trades": trades}, tmp_path)
    return tmp_path


def test_l2_windowed_run_matches_a_single_run(toy_l2_dir, tmp_path):
    single = _saved_single_run(
        Pipeline.from_source("depth_csv"), toy_l2_dir, tmp_path / "single"
    )
    assert single["trades"]["direction"].notna().all()
    cuts = _cuts(single["depth"]["timestamp"], 0.3, 0.7)

    out = Pipeline.from_source("depth_csv").run_windows(
        toy_l2_dir, cuts, tmp_path / "w"
    )

    _assert_tables_equal(single, load_data(out))


# ── The pieces ────────────────────────────────────────────────────────


def test_summary_engine_carries_on_from_its_own_book():
    depth = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-05", periods=6, freq="s", tz="UTC"),
            "price": [100, 102, 99, 101, 100, 103],
            "volume": [5, 4, 3, 2, 0, 1],
            "direction": ["bid", "ask", "bid", "ask", "bid", "ask"],
        }
    )
    whole = depth_metrics(depth)

    engine = DepthMetricsEngine()
    parts = pd.concat(
        [engine.compute(depth.iloc[:3]), engine.compute(depth.iloc[3:])],
        ignore_index=True,
    )

    pd.testing.assert_frame_equal(parts, whole)


def test_bounds_cover_the_whole_input():
    bounds = window_bounds(["2026-01-05 10:00", "2026-01-05 11:00"])

    start = pd.Timestamp("2026-01-05 10:00", tz="UTC")
    end = pd.Timestamp("2026-01-05 11:00", tz="UTC")
    assert bounds == [(None, start), (start, end), (end, None)]


def test_bounds_keep_a_zone_they_are_given():
    (_, cut), _ = window_bounds(
        [pd.Timestamp("2026-01-05 10:00", tz="America/New_York")]
    )

    assert cut == pd.Timestamp("2026-01-05 15:00", tz="UTC")


@pytest.mark.parametrize(
    "boundaries",
    [[], ["2026-01-05 11:00", "2026-01-05 10:00"], ["2026-01-05", "2026-01-05"]],
)
def test_bounds_must_be_given_and_strictly_increase(boundaries):
    with pytest.raises(ConfigError):
        window_bounds(boundaries)


# ── Databento ─────────────────────────────────────────────────────────


def test_databento_windowed_run_matches_a_single_run(tmp_path):
    from ob_analytics.databento import DatabentoSource
    from tests.test_databento import T0, mbo_frame, px

    records = [
        (10, "A", "B", px(100.00), 100),
        (11, "A", "B", px(100.10), 50),
        (20, "A", "A", px(100.30), 70),
        (21, "A", "A", px(100.40), 40),
        (10, "C", "B", px(100.00), 30),  # partial cancel -> 70 left
        # cut 1: orders 10, 11, 20 and 21 are carried
        (11, "C", "B", px(100.10), 20),  # partial cancel of a carried order
        (0, "T", "B", px(100.30), 25),
        (20, "F", "A", px(100.30), 25),
        (20, "M", "A", px(100.30), 45),  # the fill's book record
        (21, "M", "A", px(100.50), 40),  # a carried order moves
        (12, "A", "B", px(100.20), 60),
        # cut 2
        (0, "T", "A", px(100.20), 60),
        (12, "F", "B", px(100.20), 60),
        (12, "C", "B", px(100.20), 60),
        (10, "C", "B", px(100.00), 70),  # a carried order leaves
        (0, "R", "N", 0, 0),  # clear: every resting order goes
        (11, "A", "A", px(100.60), 10),  # id 11 again, now on the ask side
        # cut 3: the new order 11 is carried, on its own side and price
        (13, "A", "B", px(100.15), 5),
        (11, "C", "A", px(100.60), 4),
        (14, "A", "B", px(100.10), 8),
        (0, "T", "A", px(100.15), 5),
        (13, "F", "B", px(100.15), 5),
        (13, "C", "B", px(100.15), 5),
    ]
    frame = mbo_frame(records)
    cuts = [pd.Timestamp(T0 + k * 1_000_000, tz="UTC") for k in (5, 11, 17)]

    single = _saved_single_run(
        Pipeline(source=DatabentoSource()), frame, tmp_path / "single"
    )
    out = Pipeline(source=DatabentoSource()).run_windows(frame, cuts, tmp_path / "w")

    _assert_tables_equal(single, load_data(out))


# ── The output folder ─────────────────────────────────────────────────


def test_a_failed_run_leaves_the_folder_as_it_was(synth_session, tmp_path, monkeypatch):
    import ob_analytics.pipeline as pipeline_module

    out = tmp_path / "w"
    out.mkdir()
    (out / "events.parquet").write_bytes(b"earlier run")
    calls = []

    def fail_on_second_window(window, quotes):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("stopped part-way")
        return window.assign(aggressiveness_bps=0.0)

    monkeypatch.setattr(pipeline_module, "order_aggressiveness", fail_on_second_window)
    cut = synth_session.events["timestamp"].median()

    with pytest.raises(RuntimeError, match="part-way"):
        _synth_pipeline(synth_session).run_windows(None, [cut], out)

    assert sorted(p.name for p in out.iterdir()) == ["events.parquet"]
    assert (out / "events.parquet").read_bytes() == b"earlier run"


def test_every_table_must_be_written(tmp_path):
    from ob_analytics._windows import ParquetAppender

    with ParquetAppender(tmp_path, None) as out:
        out.write("events", pd.DataFrame({"a": [1]}))
        with pytest.raises(ConfigError, match="depth"):
            out.finish(("events", "depth"))

    assert list(tmp_path.iterdir()) == []


def test_a_column_empty_in_the_first_window_takes_its_type_from_the_run(tmp_path):
    from ob_analytics._windows import ParquetAppender

    whole = pd.DataFrame({"n": [1, 2], "code": [None, "A"]})
    with ParquetAppender(tmp_path, None) as out:
        out.write("events", whole.iloc[:1], like=whole)
        out.write("events", whole.iloc[1:], like=whole)
        out.finish(("events",))

    code = load_data(tmp_path)["events"]["code"]
    assert code.isna().tolist() == [True, False]
    assert code.iloc[1] == "A"


def test_one_boundary_can_be_given_on_its_own():
    assert len(window_bounds("2026-01-05 10:00")) == 2


def test_rows_with_no_time_fall_in_the_first_window():
    from ob_analytics._windows import window_positions

    times = pd.Series(
        [
            pd.Timestamp("2026-01-05 11:30", tz="UTC"),
            pd.NaT,
            pd.Timestamp("2026-01-05 09:00", tz="UTC"),
            pd.Timestamp("2026-01-05 10:00", tz="UTC"),
        ],
        dtype="datetime64[ns, UTC]",
    )
    windows = window_bounds(["2026-01-05 10:00", "2026-01-05 11:00"])

    positions = window_positions(times, windows)

    assert [p.tolist() for p in positions] == [[1, 2], [3], [0]]
