#!/usr/bin/env python3
"""Measure how long each stage of a run takes, and fail CI on a slow-down.

A tool that promises to handle scale needs speed as a tracked number. This
script produces that number: it runs the pipeline one stage at a time over a
named workload, records wall-clock seconds and peak memory for each stage, and
compares the result with the baseline committed next to it in
``bench-baseline.json``. A stage slower or larger than the baseline allows fails
the job.

    uv run python scripts/bench_scale.py                    # check the baseline
    uv run python scripts/bench_scale.py --update-baseline  # record a new one
    uv run python scripts/bench_scale.py --envelope         # the scale table

Per stage, not per run: #138 should not start without proof that the rebuild
loop, and not file reading or the depth kernels, is the slow part.

Memory is the process high-water mark read after each stage, not the allocation
each stage makes: ``tracemalloc`` would attribute it properly but slows the code
it watches several times over, and this runs on every pull request.

``--envelope`` keeps the original mode this script started as: the bundled
sample tiled to several sizes, each size in its own subprocess, peak RSS from
``getrusage``. That is what the "Scale envelope" table in ``ARCHITECTURE.md`` is
measured with, and it answers a different question — how memory grows with the
event count — so it is not part of the CI check.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ob_analytics import sample_csv_path
from ob_analytics._engine_frames import to_order_events
from ob_analytics.analytics import order_aggressiveness, set_order_types
from ob_analytics.bitstamp import BitstampLoader, BitstampTradeReader
from ob_analytics.config import PipelineConfig
from ob_analytics.depth import depth_metrics, price_level_volume
from ob_analytics.engine import queue_positions
from ob_analytics.synth import generate_session

BASELINE = Path(__file__).resolve().parent.parent / "bench-baseline.json"
"""Where the recorded numbers live: one file, committed, read in a diff."""

MULTIPLES = [1, 2, 3, 4]
"""Tiling multiples of the bundled sample, for ``--envelope`` only."""

SYNTH_DURATION = 18_000.0
"""Seconds of synthetic session in the ``synth`` run: about 315k events.

Sized so that load, depth metrics and queue positions each run for over a
second, which is where a timing is worth judging (see
:attr:`Margins.floor_seconds`).  It matches the bundled sample's event count,
so the two runs are readable side by side."""


# ──── what one stage measurement is ─────────────────────────────────────


@dataclass(frozen=True)
class Measurement:
    """One stage of one run, timed and sized."""

    run: str
    stage: str
    events: int
    seconds: float
    peak_rss_mib: float


@dataclass(frozen=True)
class Margins:
    """How much slower or larger than the baseline is still acceptable."""

    seconds: float
    peak_rss_mib: float
    floor_seconds: float = 1.0
    """Below this, a stage is too short to time and its seconds are not judged.

    A stage that runs for a tenth of a second varies by half from run to run on
    an idle machine, so a percentage margin over it reports noise as a
    slow-down. Its memory is still judged: the same input allocates the same
    arrays however briefly it runs.
    """


@dataclass(frozen=True)
class Regression:
    """One measurement that exceeds its baseline by more than the margin."""

    run: str
    stage: str
    metric: str
    baseline: float
    measured: float


# ──── measuring a run ───────────────────────────────────────────────────

STAGES: tuple[str, ...] = (
    "load",
    "set_order_types",
    "price_level_volume",
    "depth_metrics",
    "order_aggressiveness",
    "queue_positions",
)
"""The stages of a run, in the order the pipeline runs them.

``load`` covers reading events and trades off the source.  ``queue_positions``
is the engine's FIFO reconstruction: not part of ``Pipeline.run``, but the
stateful loop that #138 would compile, so it needs a number of its own.
"""


Load = Callable[[], tuple[pd.DataFrame, pd.DataFrame]]


def _stages(
    load: Load, config: PipelineConfig
) -> tuple[dict[str, Any], list[tuple[str, Callable[[], None]]]]:
    """One run as a list of named steps over shared state.

    Each call builds a fresh chain, so a pass can be run again from the start
    without the previous pass's frames still being alive.
    """
    state: dict[str, Any] = {}

    def _load() -> None:
        state["events"], state["trades"] = load()

    def _set_order_types() -> None:
        state["events"] = set_order_types(state["events"], state["trades"])

    def _price_level_volume() -> None:
        state["depth"] = price_level_volume(state["events"])

    def _depth_metrics() -> None:
        state["depth_summary"] = depth_metrics(
            state["depth"], bps=config.depth_bps, bins=config.depth_bins
        )

    def _order_aggressiveness() -> None:
        state["events"] = order_aggressiveness(state["events"], state["depth_summary"])

    def _queue_positions() -> None:
        queue_positions(to_order_events(state["events"]))

    steps = (
        _load,
        _set_order_types,
        _price_level_volume,
        _depth_metrics,
        _order_aggressiveness,
        _queue_positions,
    )
    return state, list(zip(STAGES, steps, strict=True))


def _peak_rss_mib() -> float:
    """The process high-water mark, in MiB.

    ``ru_maxrss`` only ever rises, so a spike inside a stage is still caught
    after it: what a stage reports is the largest the process has been by the
    time that stage ends, which is the number the scale envelope is written in.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _pass(load: Load, config: PipelineConfig) -> tuple[list[Measurement], int]:
    """Run every stage once, timing it and reading the high-water mark after it.

    One pass, not two. ``tracemalloc`` would attribute allocations to the stage
    that made them, but it slows the code it watches by several times over, and
    the job runs on every pull request. Peak RSS costs a syscall.
    """
    state, chain = _stages(load, config)
    records = []
    for stage, step in chain:
        start = time.perf_counter()
        step()
        records.append((stage, time.perf_counter() - start, _peak_rss_mib()))
    return records, len(state["events"])


def measure_run(
    name: str, load: Load, *, config: PipelineConfig | None = None
) -> list[Measurement]:
    """Measure every stage of one run named *name*.

    Parameters
    ----------
    name : str
        What the run is called in the baseline, for example ``"sample"``.
    load : callable
        Returns the run's events and trades.  Timed as the ``load`` stage.
    config : PipelineConfig, optional
        The configuration the depth stages run under.  Defaults to
        :class:`PipelineConfig`.
    """
    cfg = config if config is not None else PipelineConfig()
    records, events = _pass(load, cfg)
    return [
        Measurement(
            run=name,
            stage=stage,
            events=events,
            seconds=seconds,
            peak_rss_mib=peak,
        )
        for stage, seconds, peak in records
    ]


def measure_runs(names: list[str]) -> list[Measurement]:
    """Measure each named run, one child process per run.

    Peak RSS is a property of a process and only ever rises, so two runs
    measured in one process would report the larger one's peak for both. A
    child process per run gives each its own high-water mark, the same way
    ``--envelope`` measures each size.
    """
    measured = []
    for name in names:
        proc = subprocess.run(
            [
                sys.executable,
                __file__,
                "--measure-run",
                name,
                "--synth-duration",
                str(SYNTH_DURATION),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(proc.stdout)
        measured.extend(Measurement(**record) for record in payload["measurements"])
    return measured


# ──── grading it against the baseline ───────────────────────────────────


def exit_code(regressions: list[Regression]) -> int:
    """Fail the run when a stage is slower or larger than the baseline allows.

    Only a slow-down fails.  A stage the baseline cannot judge is reported and
    passes: a missing number is a gap in the baseline, not evidence of one.
    """
    return 1 if regressions else 0


def machine() -> str:
    """What the numbers were measured on, short enough to read in a diff."""
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    return f"{platform.system()} {platform.machine()} python{version}"


def save_baseline(path: Path, measurements: list[Measurement]) -> None:
    """Write *measurements* to *path* as the baseline CI compares against."""
    payload = {
        "machine": machine(),
        "measurements": [asdict(m) for m in measurements],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_baseline(path: Path) -> list[Measurement]:
    """Read a baseline written by :func:`save_baseline`."""
    payload = json.loads(path.read_text())
    return [Measurement(**record) for record in payload["measurements"]]


def baseline_machine(path: Path) -> str:
    """The machine named in the baseline at *path*."""
    return str(json.loads(path.read_text()).get("machine", "unrecorded"))


def unmeasured(
    measured: list[Measurement], baseline: list[Measurement]
) -> list[tuple[str, str]]:
    """The ``(run, stage)`` pairs *baseline* has no number for, in order."""
    known = {(m.run, m.stage) for m in baseline}
    return [(m.run, m.stage) for m in measured if (m.run, m.stage) not in known]


def compare(
    measured: list[Measurement],
    baseline: list[Measurement],
    *,
    margins: Margins,
) -> list[Regression]:
    """Every measurement in *measured* that its *baseline* says is a slow-down."""
    by_key = {(m.run, m.stage): m for m in baseline}
    regressions = []
    for m in measured:
        before = by_key.get((m.run, m.stage))
        if before is None:  # nothing to judge it against; see unmeasured()
            continue
        for metric, allowed in (
            ("seconds", margins.seconds),
            ("peak_rss_mib", margins.peak_rss_mib),
        ):
            was = getattr(before, metric)
            now = getattr(m, metric)
            if metric == "seconds" and was < margins.floor_seconds:
                continue  # too short to tell a slow-down from noise
            if now > was * (1 + allowed):
                regressions.append(
                    Regression(
                        run=m.run,
                        stage=m.stage,
                        metric=metric,
                        baseline=was,
                        measured=now,
                    )
                )
    return regressions


# ──── the workloads ─────────────────────────────────────────────────────


def _sample() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The bundled Bitstamp sample: real capture, about 314k events."""
    path = sample_csv_path()
    events = BitstampLoader().load(path)
    return events, BitstampTradeReader().load(events, Path(path).parent)


def _synth() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A made-up L3 session (#114): seeded, so every run measures the same work."""
    session = generate_session(seed=1, duration=SYNTH_DURATION)
    return session.events, session.trades


RUNS: dict[str, Load] = {"sample": _sample, "synth": _synth}
"""The workloads measured, by name. A baseline record names one of these."""

DEFAULT_MARGINS = Margins(seconds=0.5, peak_rss_mib=0.15)
"""How much worse than the baseline still passes.

Wall-clock time on a shared CI runner varies by tens of percent between runs of
identical code, so half again is the point where a slow-down is more likely than
noise. Memory does not vary that way: the same input allocates the same arrays,
so it is held to a tenth.
"""


def report(
    measured: list[Measurement],
    regressions: list[Regression],
    missing: list[tuple[str, str]],
) -> None:
    """Print the table, then whatever is wrong with it."""
    print(f"{'run':<8} {'stage':<22} {'events':>10} {'seconds':>9} {'peak MiB':>9}")
    for m in measured:
        print(
            f"{m.run:<8} {m.stage:<22} {m.events:>10,} "
            f"{m.seconds:>9.2f} {m.peak_rss_mib:>9.1f}"
        )
    for run, stage in missing:
        print(f"\nno baseline for {run}/{stage} - run --update-baseline to record it")
    for r in regressions:
        print(
            f"\nSLOWER {r.run}/{r.stage}: {r.metric} {r.measured:.2f} "
            f"vs {r.baseline:.2f} in the baseline"
        )


def check(
    runs: list[str], *, baseline_path: Path, margins: Margins
) -> tuple[list[Measurement], int]:
    """Measure *runs* and grade them against the saved baseline."""
    measured = measure_runs(runs)
    baseline = load_baseline(baseline_path)
    regressions = compare(measured, baseline, margins=margins)
    report(measured, regressions, unmeasured(measured, baseline))
    recorded = baseline_machine(baseline_path)
    if recorded != machine():
        print(
            f"\nthe baseline was recorded on {recorded}, this is {machine()} - "
            "seconds do not carry between machines, so record the baseline "
            "where it is checked"
        )
    return measured, exit_code(regressions)


# ──── the scale envelope, for the docs ──────────────────────────────────


def _tile(events: pd.DataFrame, mult: int) -> pd.DataFrame:
    """Replicate *events* *mult* times with offset ids and timestamps."""
    if mult == 1:
        return events
    span = (
        events["timestamp"].max() - events["timestamp"].min() + pd.Timedelta(seconds=1)
    )
    id_step = int(events["id"].max()) + 1
    copies = []
    for k in range(mult):
        c = events.copy()
        c["id"] = c["id"] + k * id_step
        c["timestamp"] = c["timestamp"] + k * span
        c["exchange_timestamp"] = c["exchange_timestamp"] + k * span
        copies.append(c)
    out = pd.concat(copies, ignore_index=True)
    out["event_id"] = range(1, len(out) + 1)
    return out


def _measure(mult: int) -> None:
    path = sample_csv_path()
    events = BitstampLoader().load(path)
    trades = BitstampTradeReader().load(events, Path(path).parent)
    events = _tile(set_order_types(events, trades), mult)
    n = len(events)
    t0 = time.perf_counter()
    summary = depth_metrics(price_level_volume(events))
    wall = time.perf_counter() - t0
    rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB -> MiB
    assert len(summary) > 0
    print(f"{n}\t{rss_mib:.0f}\t{wall:.1f}")


def envelope() -> None:
    """Print the scale-envelope table behind ``ARCHITECTURE.md``."""
    if "--mult" in sys.argv:
        _measure(int(sys.argv[sys.argv.index("--mult") + 1]))
        return
    print(f"{'events':>12}  {'peak RSS (MiB)':>14}  {'depth stages (s)':>16}")
    rows = []
    for m in MULTIPLES:
        proc = subprocess.run(
            [sys.executable, __file__, "--mult", str(m)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            print(f"  mult {m}: FAILED (rc={proc.returncode})\n{proc.stderr[-400:]}")
            continue
        n, rss, wall = proc.stdout.strip().split("\t")
        rows.append((int(n), int(rss), float(wall)))
        print(f"{int(n):>12,}  {int(rss):>14,}  {float(wall):>16.1f}")
    if len(rows) >= 2:
        (n0, r0, _), (n1, r1, _) = rows[0], rows[-1]
        per_event = (r1 - r0) / (n1 - n0)
        five_m = (r0 + per_event * (5e6 - n0)) / 1024
        print(
            f"\n~{per_event * 1e6:.0f} MiB / 1M events; 5M events extrapolates to ~{five_m:.1f} GiB"
        )


# ──── the command ───────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs",
        default=",".join(RUNS),
        help="comma-separated workloads to measure (default: every one)",
    )
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="measure and overwrite the baseline instead of checking it",
    )
    parser.add_argument("--seconds-margin", type=float, default=DEFAULT_MARGINS.seconds)
    parser.add_argument(
        "--peak-margin", type=float, default=DEFAULT_MARGINS.peak_rss_mib
    )
    parser.add_argument(
        "--floor-seconds", type=float, default=DEFAULT_MARGINS.floor_seconds
    )
    parser.add_argument(
        "--envelope",
        action="store_true",
        help="print the scale-envelope table instead (see the module docstring)",
    )
    parser.add_argument("--mult", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--measure-run", help=argparse.SUPPRESS)
    parser.add_argument("--synth-duration", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    global SYNTH_DURATION
    if args.synth_duration is not None:
        SYNTH_DURATION = args.synth_duration

    if args.mult is not None:  # one child process of an --envelope run
        _measure(args.mult)
        return 0
    if args.measure_run is not None:  # one child process of a measured run
        measured = measure_run(args.measure_run, RUNS[args.measure_run])
        print(json.dumps({"measurements": [asdict(m) for m in measured]}))
        return 0
    if args.envelope:
        envelope()
        return 0

    runs = [name for name in args.runs.split(",") if name]
    unknown = [name for name in runs if name not in RUNS]
    if unknown:
        parser.error(f"unknown run(s): {', '.join(unknown)}")

    if args.update_baseline:
        measured = measure_runs(runs)
        save_baseline(args.baseline, measured)
        report(measured, [], [])
        print(f"\nwrote {args.baseline}")
        return 0

    margins = Margins(
        seconds=args.seconds_margin,
        peak_rss_mib=args.peak_margin,
        floor_seconds=args.floor_seconds,
    )
    _, code = check(runs, baseline_path=args.baseline, margins=margins)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
