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

import pandas as pd
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
    fastest,
    load_baseline,
    machine,
    main,
    measure_run,
    measure_runs,
    save_baseline,
    time_reference,
    unmeasured,
    unscaled,
)


def _measurement(
    stage: str = "depth_metrics",
    *,
    run: str = "sample",
    events: int = 314_000,
    seconds: float = 10.0,
    peak_rss_mib: float = 400.0,
    reference_seconds: float | None = None,
) -> Measurement:
    return Measurement(
        run=run,
        stage=stage,
        events=events,
        seconds=seconds,
        peak_rss_mib=peak_rss_mib,
        reference_seconds=reference_seconds,
    )


def _empty() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A loader for tests that replace the pass and never call it."""
    return pd.DataFrame(), pd.DataFrame()


class TestCompare:
    """What counts as a slow-down against the saved baseline."""

    def test_reports_a_stage_that_is_slower_than_the_margin_allows(self) -> None:
        baseline = [_measurement(seconds=10.0, reference_seconds=0.1)]
        measured = [_measurement(seconds=13.0, reference_seconds=0.1)]

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
        baseline = [
            _measurement(seconds=0.2, peak_rss_mib=400.0, reference_seconds=0.1)
        ]
        measured = [
            _measurement(seconds=0.5, peak_rss_mib=400.0, reference_seconds=0.1)
        ]

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

    def test_the_default_margin_covers_the_variance_between_two_runners(
        self,
    ) -> None:
        """The numbers are the ones two runs of PR #219 measured (issue #220).

        Same commit, no timed code changed: 14.66s on one runner and 10.17s on
        the next, against a 9.36s baseline. Neither is a slow-down, so neither
        may fail the job. The baseline has no reference seconds, as every
        baseline before issue #536 had, so the unscaled margin applies.
        """
        baseline = [_measurement(seconds=9.36)]

        for measured_seconds in (14.66, 10.17):
            regressions = compare(
                [_measurement(seconds=measured_seconds)],
                baseline,
                margins=bench_scale.DEFAULT_MARGINS,
            )

            assert regressions == []

    def test_the_default_margin_still_fails_a_real_slow_down(self) -> None:
        """Widening the margin must not turn the check off (issue #220)."""
        baseline = [_measurement(seconds=9.36)]
        measured = [_measurement(seconds=23.4)]  # two and a half times

        regressions = compare(measured, baseline, margins=bench_scale.DEFAULT_MARGINS)

        assert [(r.stage, r.metric) for r in regressions] == [
            ("depth_metrics", "seconds")
        ]

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


class TestCompareScaled:
    """Seconds are judged at the baseline machine's speed (issue #536).

    Each side carries the time of the same reference workload, so a stage's
    seconds over the reference seconds says how slow it is for that machine.
    """

    MARGINS = Margins(seconds=0.2, peak_rss_mib=0.2, unscaled_seconds=1.0)

    def test_a_slower_machine_is_not_a_slow_down(self) -> None:
        # Three times slower at the stage and at the reference alike: the
        # runner is slower, the code is not.
        baseline = [_measurement(seconds=10.0, reference_seconds=0.1)]
        measured = [_measurement(seconds=30.0, reference_seconds=0.3)]

        assert compare(measured, baseline, margins=self.MARGINS) == []

    def test_a_slow_down_on_a_faster_machine_is_still_found(self) -> None:
        # Raw seconds match the baseline, but this machine runs the reference
        # in half the time, so the stage takes twice as long for it.
        baseline = [_measurement(seconds=10.0, reference_seconds=0.2)]
        measured = [_measurement(seconds=10.0, reference_seconds=0.1)]

        regressions = compare(measured, baseline, margins=self.MARGINS)

        assert len(regressions) == 1
        assert regressions[0].metric == "seconds"
        assert regressions[0].baseline == 10.0
        assert regressions[0].measured == pytest.approx(20.0)
        assert regressions[0].scale == pytest.approx(2.0)

    def test_a_ratio_within_the_margin_passes(self) -> None:
        # Ratio 100 in the baseline, 115 now: within a margin of 20%.
        baseline = [_measurement(seconds=10.0, reference_seconds=0.1)]
        measured = [_measurement(seconds=23.0, reference_seconds=0.2)]

        assert compare(measured, baseline, margins=self.MARGINS) == []

    def test_a_ratio_past_the_margin_fails(self) -> None:
        # Ratio 100 in the baseline, 125 now: past a margin of 20%.
        baseline = [_measurement(seconds=10.0, reference_seconds=0.1)]
        measured = [_measurement(seconds=25.0, reference_seconds=0.2)]

        regressions = compare(measured, baseline, margins=self.MARGINS)

        assert [(r.metric, r.measured) for r in regressions] == [
            ("seconds", pytest.approx(12.5))
        ]

    def test_the_default_margin_passes_a_slower_runner(self) -> None:
        """The record runs for issue #521: about 7 s, and one at 11.87 s.

        The slow one is 1.7 times the others, past the scaled margin, but its
        runner is that much slower at the reference too.
        """
        baseline = [_measurement(seconds=6.91, reference_seconds=0.10)]
        measured = [_measurement(seconds=11.87, reference_seconds=0.17)]

        assert compare(measured, baseline, margins=bench_scale.DEFAULT_MARGINS) == []

    def test_the_default_margin_fails_a_slow_down_under_two_times(self) -> None:
        # A 1.6x slow-down on the same machine: the old margin of twice the
        # baseline let this pass.
        baseline = [_measurement(seconds=7.0, reference_seconds=0.1)]
        measured = [_measurement(seconds=11.2, reference_seconds=0.1)]

        regressions = compare(measured, baseline, margins=bench_scale.DEFAULT_MARGINS)

        assert [r.metric for r in regressions] == ["seconds"]

    def test_the_default_margin_is_tighter_than_two_times(self) -> None:
        assert bench_scale.DEFAULT_MARGINS.seconds < 1.0

    def test_memory_is_never_scaled(self) -> None:
        baseline = [_measurement(peak_rss_mib=400.0, reference_seconds=0.1)]
        measured = [_measurement(peak_rss_mib=500.0, reference_seconds=0.3)]

        regressions = compare(measured, baseline, margins=self.MARGINS)

        assert [(r.metric, r.measured, r.scale) for r in regressions] == [
            ("peak_rss_mib", 500.0, 1.0)
        ]


class TestCompareOldBaseline:
    """A baseline recorded before the reference existed still works."""

    MARGINS = Margins(seconds=0.2, peak_rss_mib=0.2, unscaled_seconds=1.0)

    def test_old_baseline_seconds_are_compared_unscaled(self) -> None:
        # Within twice the baseline: passes, although past the scaled margin.
        baseline = [_measurement(seconds=10.0)]
        measured = [_measurement(seconds=19.0, reference_seconds=0.1)]

        assert compare(measured, baseline, margins=self.MARGINS) == []

    def test_old_baseline_still_fails_past_the_unscaled_margin(self) -> None:
        baseline = [_measurement(seconds=10.0)]
        measured = [_measurement(seconds=21.0, reference_seconds=0.1)]

        regressions = compare(measured, baseline, margins=self.MARGINS)

        assert [(r.metric, r.measured, r.scale) for r in regressions] == [
            ("seconds", 21.0, 1.0)
        ]

    def test_an_old_baseline_file_reads_with_no_reference(self, tmp_path: Path) -> None:
        path = tmp_path / "bench-baseline.json"
        path.write_text(
            json.dumps(
                {
                    "machine": "Linux x86_64 ubuntu-24.04 python3.11",
                    "measurements": [
                        {
                            "run": "sample",
                            "stage": "depth_metrics",
                            "events": 314057,
                            "seconds": 11.87,
                            "peak_rss_mib": 428.2,
                        }
                    ],
                }
            )
        )

        baseline = load_baseline(path)

        assert baseline == [
            Measurement(
                run="sample",
                stage="depth_metrics",
                events=314057,
                seconds=11.87,
                peak_rss_mib=428.2,
                reference_seconds=None,
            )
        ]
        assert unscaled(baseline)

    def test_a_new_baseline_is_not_unscaled(self) -> None:
        assert not unscaled([_measurement(reference_seconds=0.1)])

    def test_the_committed_baseline_reads(self) -> None:
        # Whatever its format, the file in the repository must load.
        assert load_baseline(bench_scale.BASELINE)


class TestFastest:
    """Each stage keeps its fastest time; memory comes from the first pass."""

    def test_keeps_each_stages_fastest_seconds_across_passes(self) -> None:
        passes = [
            [("load", 1.0, 100.0), ("depth_metrics", 12.0, 400.0)],
            [("load", 0.8, 410.0), ("depth_metrics", 14.0, 410.0)],
            [("load", 1.3, 420.0), ("depth_metrics", 11.0, 420.0)],
        ]

        assert fastest(passes) == [
            ("load", 0.8, 100.0),
            ("depth_metrics", 11.0, 400.0),
        ]

    def test_one_pass_is_returned_as_it_is(self) -> None:
        passes = [[("load", 1.0, 100.0), ("depth_metrics", 12.0, 400.0)]]

        assert fastest(passes) == passes[0]


class TestReference:
    """The reference workload is timed in the same process as the stages."""

    def test_is_a_positive_number_of_seconds(self) -> None:
        assert time_reference(repeats=1) > 0

    def test_keeps_the_fastest_of_its_repeats(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = iter([0.0, 0.5, 1.0, 1.2, 2.0, 2.9])
        monkeypatch.setattr(bench_scale, "reference_workload", lambda: None)
        monkeypatch.setattr(bench_scale.time, "perf_counter", lambda: next(clock))

        assert time_reference(repeats=3) == pytest.approx(0.2)


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

    def test_freed_memory_is_returned_before_every_stage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Without it the queue_positions mark depended on where glibc's heap
        # landed and came out near 418 or near 471 MiB on CI from one push to
        # the next.
        trims: list[int] = []
        monkeypatch.setattr(bench_scale, "_MALLOC_TRIM", trims.append)
        session = generate_session(seed=1, duration=60)

        measure_run("synth", lambda: (session.events, session.trades), repeats=1)

        # One before each of the six stages, and one before the reference.
        assert trims == [0, 0, 0, 0, 0, 0, 0]

    def test_measure_run_keeps_the_fastest_of_its_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        passes = iter(
            [
                ([("load", 2.0, 100.0), ("queue_positions", 5.0, 300.0)], 7),
                ([("load", 1.0, 310.0), ("queue_positions", 6.0, 310.0)], 7),
                ([("load", 3.0, 320.0), ("queue_positions", 4.0, 320.0)], 7),
            ]
        )
        references = iter([0.3, 0.1, 0.2])
        monkeypatch.setattr(bench_scale, "_pass", lambda load, config: next(passes))
        monkeypatch.setattr(bench_scale, "time_reference", lambda: next(references))

        measured = measure_run("synth", _empty, repeats=3)

        assert [(m.stage, m.seconds, m.peak_rss_mib) for m in measured] == [
            ("load", 1.0, 100.0),
            ("queue_positions", 4.0, 300.0),
        ]
        assert {m.reference_seconds for m in measured} == {0.1}
        assert {m.events for m in measured} == {7}

    def test_every_record_carries_the_reference_seconds(self) -> None:
        session = generate_session(seed=1, duration=60)

        measured = measure_run(
            "synth", lambda: (session.events, session.trades), repeats=1
        )

        (reference,) = {m.reference_seconds for m in measured}
        assert reference is not None
        assert reference > 0

    def test_repeats_below_one_are_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            measure_run("synth", _empty, repeats=0)


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
                "reference_seconds": None,
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

    def test_update_baseline_records_the_reference_seconds(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "bench-baseline.json"

        main(
            [
                "--runs",
                "synth",
                "--repeats",
                "1",
                "--update-baseline",
                "--baseline",
                str(path),
            ]
        )

        written = load_baseline(path)
        assert all(m.reference_seconds is not None for m in written)
        assert not unscaled(written)

    def test_an_old_baseline_is_called_out(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = self._baseline(tmp_path, seconds=600.0, peak_rss_mib=100_000.0)

        code = main(["--runs", "synth", "--repeats", "1", "--baseline", str(path)])

        assert code == 0
        assert "no reference seconds" in capsys.readouterr().out

    def test_repeats_below_one_are_refused(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            main(["--runs", "synth", "--repeats", "0"])


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

    def test_a_baseline_from_another_os_release_is_called_out(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The case that hid a 40 MiB jump: ubuntu-latest moved from 24.04 to
        # 26.04 on the same CPU and Python, and the warning stayed silent.
        path = tmp_path / "bench-baseline.json"
        _os_release(monkeypatch, {"ID": "ubuntu", "VERSION_ID": "24.04"})
        save_baseline(path, [_measurement("load", run="synth", seconds=600.0)])
        _os_release(monkeypatch, {"ID": "ubuntu", "VERSION_ID": "26.04"})

        main(["--runs", "synth", "--baseline", str(path)])

        out = capsys.readouterr().out
        assert "recorded on" in out
        assert "ubuntu-24.04" in out
        assert "ubuntu-26.04" in out

    def test_a_baseline_from_this_machine_is_not_called_out(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "bench-baseline.json"
        save_baseline(path, [_measurement("load", run="synth", seconds=600.0)])

        main(["--runs", "synth", "--baseline", str(path)])

        assert "recorded on" not in capsys.readouterr().out


def _os_release(monkeypatch: pytest.MonkeyPatch, info: dict[str, str] | None) -> None:
    """Make the os-release file read as *info*, or be missing when ``None``."""

    def read() -> dict[str, str]:
        if info is None:
            raise OSError("no os-release file")
        return info

    monkeypatch.setattr(bench_scale.platform, "freedesktop_os_release", read)
    monkeypatch.setattr(bench_scale.platform, "release", lambda: "23.6.0")


class TestMachine:
    """The machine string names everything that moves the numbers."""

    def test_names_the_os_release(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _os_release(monkeypatch, {"ID": "ubuntu", "VERSION_ID": "24.04"})

        assert "ubuntu-24.04" in machine()

    def test_names_a_release_without_a_version_by_its_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _os_release(monkeypatch, {"ID": "arch"})

        assert " arch " in machine()

    def test_falls_back_to_the_kernel_release_without_an_os_release_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _os_release(monkeypatch, None)

        assert " 23.6.0 " in machine()

    def test_still_names_the_system_cpu_and_python(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _os_release(monkeypatch, {"ID": "ubuntu", "VERSION_ID": "24.04"})
        monkeypatch.setattr(bench_scale.platform, "system", lambda: "Linux")
        monkeypatch.setattr(bench_scale.platform, "machine", lambda: "x86_64")

        assert machine().startswith("Linux x86_64 ubuntu-24.04 python3.")
