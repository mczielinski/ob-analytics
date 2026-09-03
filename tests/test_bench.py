"""Tests for the speed test, ``scripts/bench_scale.py`` (issue #144).

The script measures one number per stage of a run and compares it with a saved
baseline, so that CI fails when a change makes the pipeline slower.

Nothing here asserts a duration.  A wall-clock number is a property of the
machine that produced it, so the tests cover the two things that are the
script's own: the rule that decides whether a measurement is a slow-down, and
the shape of what a measured run reports.  Expected values are written literals,
never re-derived with the comparison's own arithmetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ob_analytics.synth import generate_session
from scripts import bench_scale
from scripts.bench_scale import (
    STAGES,
    Margins,
    Measurement,
    Regression,
    compare,
    exit_code,
    load_baseline,
    machine,
    main,
    measure_run,
    measure_runs,
    save_baseline,
    unmeasured,
)


def _measurement(
    stage: str = "depth_metrics",
    *,
    run: str = "sample",
    events: int = 314_000,
    seconds: float = 10.0,
    peak_rss_mib: float = 400.0,
) -> Measurement:
    return Measurement(
        run=run, stage=stage, events=events, seconds=seconds, peak_rss_mib=peak_rss_mib
    )


class TestCompare:
    """What counts as a slow-down against the saved baseline."""

    def test_reports_a_stage_that_is_slower_than_the_margin_allows(self) -> None:
        baseline = [_measurement(seconds=10.0)]
        measured = [_measurement(seconds=13.0)]

        regressions = compare(
            measured, baseline, margins=Margins(seconds=0.2, peak_rss_mib=0.2)
        )

        assert [(r.stage, r.metric) for r in regressions] == [
            ("depth_metrics", "seconds")
        ]
        assert regressions[0].baseline == 10.0
        assert regressions[0].measured == 13.0

    def test_reports_a_stage_that_uses_more_memory_than_the_margin_allows(
        self,
    ) -> None:
        baseline = [_measurement(peak_rss_mib=400.0)]
        measured = [_measurement(peak_rss_mib=500.0)]

        regressions = compare(
            measured, baseline, margins=Margins(seconds=0.2, peak_rss_mib=0.2)
        )

        assert [(r.stage, r.metric) for r in regressions] == [
            ("depth_metrics", "peak_rss_mib")
        ]
        assert regressions[0].baseline == 400.0
        assert regressions[0].measured == 500.0

    def test_a_stage_too_short_to_time_is_not_judged_on_seconds(self) -> None:
        baseline = [_measurement(seconds=0.2, peak_rss_mib=400.0)]
        measured = [_measurement(seconds=0.5, peak_rss_mib=400.0)]

        regressions = compare(
            measured,
            baseline,
            margins=Margins(seconds=0.2, peak_rss_mib=0.2, floor_seconds=1.0),
        )

        assert regressions == []

    def test_a_stage_too_short_to_time_is_still_judged_on_memory(self) -> None:
        baseline = [_measurement(seconds=0.2, peak_rss_mib=400.0)]
        measured = [_measurement(seconds=0.2, peak_rss_mib=500.0)]

        regressions = compare(
            measured,
            baseline,
            margins=Margins(seconds=0.2, peak_rss_mib=0.2, floor_seconds=1.0),
        )

        assert [r.metric for r in regressions] == ["peak_rss_mib"]

    def test_a_stage_the_baseline_does_not_name_is_not_a_slow_down(self) -> None:
        baseline = [_measurement("depth_metrics", seconds=10.0)]
        measured = [
            _measurement("depth_metrics", seconds=10.0),
            _measurement("queue_positions", seconds=99.0),
        ]

        regressions = compare(
            measured, baseline, margins=Margins(seconds=0.2, peak_rss_mib=0.2)
        )

        assert regressions == []


class TestUnmeasured:
    """A stage the baseline cannot judge is named, not passed over in silence."""

    def test_names_the_stages_the_baseline_does_not_cover(self) -> None:
        baseline = [_measurement("depth_metrics")]
        measured = [
            _measurement("depth_metrics"),
            _measurement("queue_positions"),
            _measurement("depth_metrics", run="synth"),
        ]

        assert unmeasured(measured, baseline) == [
            ("sample", "queue_positions"),
            ("synth", "depth_metrics"),
        ]


@pytest.fixture(autouse=True)
def _short_synth_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the synthetic run to a second of work.

    The committed baseline is measured on the real thing; these tests are about
    the wiring, and a 50k-event session would make the suite pay for it.
    """
    monkeypatch.setattr(bench_scale, "SYNTH_DURATION", 60.0)


class TestMeasureRun:
    """What one measured run reports.

    The numbers themselves belong to the machine, so only their shape is
    asserted: one record per stage, every stage of the pipeline covered, and
    each record carrying the run it came from and the events it ran over.
    """

    def test_reports_one_record_per_stage_of_the_pipeline(self) -> None:
        session = generate_session(seed=1, duration=60)

        measured = measure_run("synth", lambda: (session.events, session.trades))

        assert [m.stage for m in measured] == list(STAGES)
        assert {m.run for m in measured} == {"synth"}

    def test_every_record_carries_the_event_count_and_finite_numbers(self) -> None:
        session = generate_session(seed=1, duration=60)

        measured = measure_run("synth", lambda: (session.events, session.trades))

        assert {m.events for m in measured} == {len(session.events)}
        assert all(m.seconds >= 0 for m in measured)
        assert all(m.peak_rss_mib > 0 for m in measured)

    def test_peak_memory_is_the_high_water_mark_so_it_never_falls(self) -> None:
        session = generate_session(seed=1, duration=60)

        measured = measure_run("synth", lambda: (session.events, session.trades))

        peaks = [m.peak_rss_mib for m in measured]
        assert peaks == sorted(peaks)


class TestMeasureRuns:
    """Each run is measured on its own, so the runs do not read as one."""

    def test_reports_the_stages_of_every_run_asked_for(self) -> None:
        measured = measure_runs(["synth"])

        assert [m.stage for m in measured] == list(STAGES)
        assert {m.run for m in measured} == {"synth"}

    def test_a_runs_peak_memory_does_not_carry_into_the_next_run(self) -> None:
        measured = measure_runs(["synth", "synth"])

        first, second = measured[: len(STAGES)], measured[len(STAGES) :]
        assert second[0].peak_rss_mib <= first[-1].peak_rss_mib


class TestBaselineFile:
    """The saved baseline is a file people read in a diff."""

    def test_written_measurements_read_back_unchanged(self, tmp_path: Path) -> None:
        measurements = [
            _measurement("load", seconds=1.5, peak_rss_mib=64.0),
            _measurement("depth_metrics", seconds=10.0, peak_rss_mib=400.0),
        ]
        path = tmp_path / "bench-baseline.json"

        save_baseline(path, measurements)

        assert load_baseline(path) == measurements

    def test_records_the_machine_the_numbers_came_from(self, tmp_path: Path) -> None:
        path = tmp_path / "bench-baseline.json"

        save_baseline(path, [_measurement("load")])

        assert json.loads(path.read_text())["machine"] == machine()

    def test_is_written_as_named_fields_one_record_per_stage(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "bench-baseline.json"

        save_baseline(path, [_measurement("load", seconds=1.5, peak_rss_mib=64.0)])

        assert json.loads(path.read_text())["measurements"] == [
            {
                "run": "sample",
                "stage": "load",
                "events": 314_000,
                "seconds": 1.5,
                "peak_rss_mib": 64.0,
            }
        ]


class TestExitCode:
    """A slow-down fails the job; nothing else does."""

    def test_is_zero_when_no_stage_is_slower(self) -> None:
        assert exit_code([]) == 0

    def test_is_one_when_a_stage_is_slower(self) -> None:
        regression = Regression(
            run="sample",
            stage="depth_metrics",
            metric="seconds",
            baseline=10.0,
            measured=13.0,
        )

        assert exit_code([regression]) == 1


class TestMain:
    """The command CI runs, end to end on a small synthetic run.

    ``--runs synth`` keeps this to a second or so: the point is the wiring —
    measure, compare, report — not the size of the input.
    """

    def _baseline(self, tmp_path: Path, **fields: float) -> Path:
        path = tmp_path / "bench-baseline.json"
        save_baseline(
            path,
            [
                Measurement(run="synth", stage=stage, events=1, **fields)
                for stage in STAGES
            ],
        )
        return path

    def test_update_baseline_writes_one_record_per_stage(self, tmp_path: Path) -> None:
        path = tmp_path / "bench-baseline.json"

        code = main(["--runs", "synth", "--update-baseline", "--baseline", str(path)])

        assert code == 0
        written = load_baseline(path)
        assert [m.stage for m in written] == list(STAGES)
        assert {m.run for m in written} == {"synth"}

    def test_passes_when_every_stage_is_within_the_margin(self, tmp_path: Path) -> None:
        path = self._baseline(tmp_path, seconds=600.0, peak_rss_mib=100_000.0)

        assert main(["--runs", "synth", "--baseline", str(path)]) == 0

    def test_fails_when_a_stage_is_slower_than_the_baseline_allows(
        self, tmp_path: Path
    ) -> None:
        path = self._baseline(tmp_path, seconds=0.0, peak_rss_mib=0.0)

        assert main(["--runs", "synth", "--baseline", str(path)]) == 1


class TestMachineMismatch:
    """A wall-clock baseline belongs to the machine that recorded it."""

    def test_a_baseline_from_another_machine_is_called_out(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "bench-baseline.json"
        save_baseline(path, [_measurement("load", run="synth", seconds=600.0)])
        payload = json.loads(path.read_text())
        payload["machine"] = "Nintendo 64 python3.11"
        path.write_text(json.dumps(payload))

        main(["--runs", "synth", "--baseline", str(path)])

        assert "Nintendo 64 python3.11" in capsys.readouterr().out
