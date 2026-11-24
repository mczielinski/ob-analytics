#!/usr/bin/env python3
"""Measure how long each stage of a run takes, and fail CI on a slow-down.

A tool that promises to handle scale needs speed as a tracked number. This
script produces that number: it runs the pipeline one stage at a time over a
named workload, records wall-clock seconds and peak memory for each stage, and
compares the result with the baseline committed next to it in
``bench-baseline.json``. A stage slower or larger than the baseline allows fails
the job.

Each stage is timed several times (:data:`REPEATS`) and the fastest time is
kept: other work on the machine only ever adds time, so the fastest run is the
closest to what the stage costs. Runners also differ in speed as a whole, which
repeating cannot fix. So each run also times a small fixed workload of the same
kind (:func:`reference_workload`), and a stage is judged on its seconds divided
by the reference seconds, against the same ratio in the baseline.

    uv run python scripts/bench_scale.py                    # check the baseline
    uv run python scripts/bench_scale.py --update-baseline  # record a new one
    uv run python scripts/bench_scale.py --envelope         # the scale table
    uv run python scripts/bench_scale.py --out-of-core      # reading saved runs in parts

Per stage, not per run: #138 should not start without proof that the rebuild
loops, and not file reading or analytics, are the slow part. There are two:
``depth_metrics`` replays the L2 book one event at a time, and
``queue_positions`` walks the per-order queues.

Compile time is not counted. The depth replay and the queue engine are loops
that numba compiles to machine code the first time they run in a process, or
loads from numba's cache on disk. Each process runs both once on a two-row
frame (:func:`warm_up`) before it times anything, so no stage's seconds include
the compile, and every stage's memory includes the compiler and the compiled
code.

Memory is the process high-water mark read after each stage, not the allocation
each stage makes: ``tracemalloc`` would attribute it properly but slows the code
it watches several times over, and this runs on every pull request. Before each
stage the script hands freed heap memory back to the operating system (see
:func:`_return_freed_memory`), so the mark reflects what the stage holds and not
where the allocator happened to leave the previous stage's garbage.

``--envelope`` keeps the original mode this script started as: the bundled
sample tiled to several sizes, each size in its own subprocess, peak RSS from
``getrusage``. That is what the "Scale envelope" table in ``ARCHITECTURE.md`` is
measured with, and it answers a different question — how memory grows with the
event count — so it is not part of the CI check.

``--out-of-core`` measures the other half of that table: reading a saved run in
parts with :mod:`ob_analytics.scan`. It saves the bundled sample's run once,
tiles the saved ``depth`` and ``depth_summary`` files to several sizes one row
group at a time (so building them needs no memory), and in a subprocess per
size runs :func:`~ob_analytics.scan.aggregate_by_time` and
:func:`~ob_analytics.scan.summarize_depth_file` over them. The largest size is
larger, as a pandas frame, than the memory of a 16 GB machine.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ob_analytics import pipeline, sample_csv_path
from ob_analytics.analytics import set_order_types
from ob_analytics.bitstamp import BitstampLoader, BitstampTradeReader
from ob_analytics.config import PipelineConfig
from ob_analytics.data import ROW_GROUP_ROWS, save_data
from ob_analytics.depth import price_level_volume, summarize_depth
from ob_analytics.queue import queue_positions
from ob_analytics.scan import aggregate_by_time, summarize_depth_file
from ob_analytics.synth import generate_session

BASELINE = Path(__file__).resolve().parent.parent / "bench-baseline.json"
"""Where the recorded numbers live: one file, committed, read in a diff."""

MULTIPLES = [1, 2, 3, 4]
"""Tiling multiples of the bundled sample, for ``--envelope`` only."""

AGGREGATE_MULTIPLES = [1, 16, 64, 160]
"""Tiling multiples of the saved ``depth_summary``, for ``--out-of-core``.

At 160 the table is about 50M rows: about 17 GiB as a pandas frame, and about
1 GB on disk."""

SUMMARIZE_MULTIPLES = [1, 4, 16]
"""Tiling multiples of the saved ``depth``, for ``--out-of-core``.

Smaller than :data:`AGGREGATE_MULTIPLES` because the summary replay takes about
12 s per copy of the sample however it is read."""

SYNTH_DURATION = 18_000.0
"""Seconds of synthetic session in the ``synth`` run: about 315k events.

Sized so that load, depth metrics and queue positions each run for over a
second, which is where a timing is worth judging (see
:attr:`Margins.floor_seconds`).  It matches the bundled sample's event count,
so the two runs are readable side by side."""

REPEATS = 3
"""How many times each run's stages are timed; the fastest time is kept.

Three passes cost about a minute of CI time over one, and catch a pass that
another process slowed down.  Peak memory comes from the first pass only, since
the high-water mark only ever rises."""

REFERENCE_REPEATS = 5
"""How many times the reference workload is timed after each pass."""


# ──── what one stage measurement is ─────────────────────────────────────


@dataclass(frozen=True)
class Measurement:
    """One stage of one run, timed and sized."""

    run: str
    stage: str
    events: int
    seconds: float
    peak_rss_mib: float
    reference_seconds: float | None = None
    """The fastest time of :func:`reference_workload` in the same process.

    ``None`` in a baseline recorded before the reference existed. Such a
    baseline is still read, and its seconds are compared as they are, with the
    wider :attr:`Margins.unscaled_seconds`.
    """


@dataclass(frozen=True)
class Margins:
    """How much slower or larger than the baseline is still acceptable."""

    seconds: float
    """The margin on a stage's seconds after scaling by the reference."""
    peak_rss_mib: float
    unscaled_seconds: float = 1.0
    """The margin on seconds when either side has no reference time.

    Without the reference, the seconds of two runners are compared as they are,
    and runners differ by up to about three times. This is the margin the check
    used before the reference existed.
    """
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
    scale: float = 1.0
    """What *measured* was multiplied by to put it in the baseline machine's speed.

    The baseline's reference seconds divided by this run's. ``1.0`` for memory,
    and for seconds when there is no reference to scale by.
    """


# ──── measuring a run ───────────────────────────────────────────────────

STAGES: tuple[str, ...] = (
    "load",
    *(name for name, _ in pipeline.STAGES),
    "queue_positions",
)
"""The stages of a run, in the order the pipeline runs them.

The middle ones are the pipeline's own (``ob_analytics.pipeline.STAGES``), run
as ``Pipeline.run`` runs them.  ``load`` covers reading events and trades off
the source.  ``queue_positions`` is the public ``ob_analytics.queue`` call: the
sort into time order and the engine's FIFO reconstruction.  It is not part of
``Pipeline.run``, but it holds one of the two loops that #138 compiled, so it
needs a number of its own.
"""


Load = Callable[[], tuple[pd.DataFrame, pd.DataFrame]]
Step = Callable[[], None]


def _stages(
    load: Load, config: PipelineConfig
) -> tuple[dict[str, Any], list[tuple[str, Step]]]:
    """One run as a list of named steps over shared state.

    Each call builds a fresh chain, so a pass can be run again from the start
    without the previous pass's frames still being alive.
    """
    state: dict[str, Any] = {}

    def _load() -> None:
        state["events"], state["trades"] = load()

    def _queue_positions() -> None:
        # The public call, as a user makes it: it sorts the events into time
        # order before the engine replays them.  Fed the frame as the loader
        # left it (in order-id order for Bitstamp), the engine sees few live
        # levels at once and the stage reads about ten times too fast (#489).
        queue_positions(state["events"])

    def _stage(stage: Callable[[dict[str, Any], PipelineConfig], None]) -> Step:
        return lambda: stage(state, config)

    steps = (
        _load,
        *(_stage(stage) for _, stage in pipeline.STAGES),
        _queue_positions,
    )
    return state, list(zip(STAGES, steps, strict=True))


def _peak_rss_mib() -> float:
    """The process high-water mark, in MiB.

    ``ru_maxrss`` only ever rises, so a spike inside a stage is still caught
    after it: what a stage reports is the largest the process has been by the
    time that stage ends, which is the number the scale envelope is written in.
    """
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # getrusage reports bytes on macOS and KiB on Linux.
    return maxrss / 1024**2 if sys.platform == "darwin" else maxrss / 1024


def _load_malloc_trim() -> Callable[[int], int] | None:
    """glibc's ``malloc_trim``, or ``None`` where the C library has none."""
    if sys.platform != "linux":
        return None
    trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
    if trim is None:  # not glibc, e.g. musl
        return None
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    return trim


_MALLOC_TRIM = _load_malloc_trim()


def _return_freed_memory() -> None:
    """Hand the heap memory freed so far back to the operating system.

    glibc keeps freed memory inside its heap resident, and returns it on its
    own only from the top of the heap. Whether the frames an earlier stage
    dropped sit at the top depends on where the heap starts, which address
    space randomisation moves from one process to the next. So without this, the
    same stage over the same input entered ``queue_positions`` holding either
    about 305 or about 390 MiB, and its high-water mark came out near 418 or
    near 471 MiB on CI, a coin toss that failed half the pushes to main. Trimming
    before each stage gives the same mark on every run. It only releases pages
    nothing is using, so it cannot raise a stage's peak. It runs outside the
    timed window, but the stage then faults those pages back in, which adds a
    few percent to its seconds: well inside :data:`DEFAULT_MARGINS`.
    """
    if _MALLOC_TRIM is not None:
        _MALLOC_TRIM(0)


def warm_up() -> None:
    """Compile the depth replay and the queue engine, or load them from cache.

    Two-row frames with the dtypes the pipeline produces (integer tick prices,
    integer lot sizes) give the compiled loops the same argument types as a
    real run, so the real run reuses these compiles.
    """
    two_rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([0, 1], utc=True),
            "price": np.array([100, 101], dtype=np.int64),
            "volume": np.array([1, 1], dtype=np.int64),
            "direction": ["bid", "ask"],
        }
    )
    summarize_depth(two_rows)
    queue_positions(
        two_rows.assign(
            event_id=[1, 2],
            id=[1, 2],
            action="created",
            fill=np.zeros(2, dtype=np.int64),
            type="resting-limit",
        )
    )


def reference_workload() -> None:
    """A small fixed piece of work, timed to tell how fast this machine is.

    The stages are numpy and pandas code, so this is too: a seeded frame is
    sorted, given a running total per group, summed per group, and searched
    with ``searchsorted``. The input is the same on every call, so the time it takes
    depends only on the machine. It runs in about a tenth of a second on a
    desktop, and the fastest of several runs varies by about one percent.
    """
    rng = np.random.default_rng(0)
    n = 400_000
    frame = pd.DataFrame(
        {
            "level": rng.integers(0, 2_000, n),
            "price": rng.integers(10_000, 20_000, n),
            "volume": rng.random(n),
        }
    )
    frame = frame.sort_values(["level", "price"], kind="stable")
    frame["cumulative"] = frame.groupby("level")["volume"].cumsum()
    edges = np.cumsum(frame.groupby("level")["volume"].sum().to_numpy())
    np.searchsorted(edges, rng.random(n) * edges[-1])


def time_reference(repeats: int = REFERENCE_REPEATS) -> float:
    """The fastest of *repeats* runs of :func:`reference_workload`, in seconds."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        reference_workload()
        times.append(time.perf_counter() - start)
    return min(times)


def _pass(
    load: Load, config: PipelineConfig
) -> tuple[list[tuple[str, float, float]], int]:
    """Run every stage once, timing it and reading the high-water mark after it.

    One pass, not two. ``tracemalloc`` would attribute allocations to the stage
    that made them, but it slows the code it watches by several times over, and
    the job runs on every pull request. Peak RSS costs a syscall.
    """
    state, chain = _stages(load, config)
    records = []
    for stage, step in chain:
        _return_freed_memory()
        start = time.perf_counter()
        step()
        records.append((stage, time.perf_counter() - start, _peak_rss_mib()))
    return records, len(state["events"])


def fastest(
    passes: list[list[tuple[str, float, float]]],
) -> list[tuple[str, float, float]]:
    """Each stage's fastest seconds over *passes*, with the first pass's memory.

    Noise only adds time, so the fastest pass is the closest to the stage's
    cost. Peak memory is the process high-water mark, which only rises, so a
    later pass's mark carries the earlier passes' peaks: only the first pass
    reads a stage's own.
    """
    first = passes[0]
    return [
        (stage, min(p[i][1] for p in passes), peak)
        for i, (stage, _, peak) in enumerate(first)
    ]


def measure_run(
    name: str,
    load: Load,
    *,
    config: PipelineConfig | None = None,
    repeats: int = REPEATS,
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
    repeats : int, optional
        How many times to run every stage.  Each stage keeps its fastest
        time.  Defaults to :data:`REPEATS`.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    cfg = config if config is not None else PipelineConfig()
    warm_up()
    passes = []
    references = []
    for _ in range(repeats):
        records, events = _pass(load, cfg)
        passes.append(records)
        # Timed after every pass rather than once, so that one slow moment
        # does not set it, and after the heap is trimmed, as every stage is.
        # After the pass, not before: the first pass's memory marks are then
        # the same as with no reference at all.
        _return_freed_memory()
        references.append(time_reference())
    reference = min(references)
    return [
        Measurement(
            run=name,
            stage=stage,
            events=events,
            seconds=seconds,
            peak_rss_mib=peak,
            reference_seconds=reference,
        )
        for stage, seconds, peak in fastest(passes)
    ]


def measure_runs(names: list[str], *, repeats: int = REPEATS) -> list[Measurement]:
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
                "--repeats",
                str(repeats),
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


def os_release() -> str:
    """The OS release, such as ``ubuntu-24.04``.

    Two runners with the same kernel family, CPU and Python can still differ in
    peak memory when the OS release differs, so the release is part of the
    machine.  Where there is no os-release file (macOS, Windows), the kernel
    release stands in.
    """
    try:
        info = platform.freedesktop_os_release()
    except OSError:
        return platform.release()
    parts = [info.get("ID", ""), info.get("VERSION_ID", "")]
    return "-".join(part for part in parts if part) or platform.release()


def machine() -> str:
    """What the numbers were measured on, short enough to read in a diff."""
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    return f"{platform.system()} {platform.machine()} {os_release()} python{version}"


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


def unscaled(baseline: list[Measurement]) -> bool:
    """Whether *baseline* was recorded without reference seconds.

    Such a baseline still works, but its seconds are compared as they are, so
    a slow runner reads as a slow-down.  Record a new one to fix it.
    """
    return any(m.reference_seconds is None for m in baseline)


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
    """Every measurement in *measured* that its *baseline* says is a slow-down.

    Seconds are compared at the baseline machine's speed: a stage's measured
    seconds are multiplied by the baseline's reference seconds over this run's,
    which is the same as comparing the two ratios of stage to reference.  When
    either side has no reference time, the seconds are compared as they are,
    under the wider :attr:`Margins.unscaled_seconds`.
    """
    by_key = {(m.run, m.stage): m for m in baseline}
    regressions = []
    for m in measured:
        before = by_key.get((m.run, m.stage))
        if before is None:  # nothing to judge it against; see unmeasured()
            continue
        if before.reference_seconds and m.reference_seconds:
            scale = before.reference_seconds / m.reference_seconds
            seconds_margin = margins.seconds
        else:
            scale = 1.0
            seconds_margin = margins.unscaled_seconds
        if before.seconds >= margins.floor_seconds:  # else too short to judge
            now = m.seconds * scale
            if now > before.seconds * (1 + seconds_margin):
                regressions.append(
                    Regression(
                        run=m.run,
                        stage=m.stage,
                        metric="seconds",
                        baseline=before.seconds,
                        measured=now,
                        scale=scale,
                    )
                )
        if m.peak_rss_mib > before.peak_rss_mib * (1 + margins.peak_rss_mib):
            regressions.append(
                Regression(
                    run=m.run,
                    stage=m.stage,
                    metric="peak_rss_mib",
                    baseline=before.peak_rss_mib,
                    measured=m.peak_rss_mib,
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

DEFAULT_MARGINS = Margins(seconds=0.5, peak_rss_mib=0.15, unscaled_seconds=1.0)
"""How much worse than the baseline still passes.

Seconds vary for two reasons. Other work on the machine adds time to one
measurement, and GitHub runners differ in speed. On main, with one timing per
stage and no reference, ``summarize_depth`` on the sample took from 5.3 s to
17.3 s across about 28 runs of the same code, more than three times apart. Of
five baseline recordings in a row, four took about 7 s and one took 11.87 s. A
margin of twice the baseline was needed to keep that spread from failing, and
it let a slow-down of up to twice pass.

Both causes are now handled. The fastest of :data:`REPEATS` passes removes
most of the first, since noise only adds time. Dividing by the reference
seconds removes most of the second, since a slower machine is slower at the
reference too. What is left is the part of a runner's speed that the reference
does not share, such as a different cache size. So seconds may be half again
the baseline's, after scaling. That is the margin a single unscaled timing
could not hold (one commit measured 14.66 s against a 9.36 s baseline, 1.57
times), and it still fails any slow-down over 1.5 times. Check it against the
spread of the scaled seconds over several runs on main, and tighten it if the
spread allows.

A baseline without reference seconds is compared unscaled, under the old
margin of twice the baseline.

Memory does not vary that way: the same input allocates the same arrays however
fast the machine runs, so it is held to 15%.
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
    references = {m.run: m.reference_seconds for m in measured}
    for run, seconds in references.items():
        if seconds is not None:
            print(f"{run:<8} {'(reference workload)':<22} {'':>10} {seconds:>9.3f}")
    for run, stage in missing:
        print(f"\nno baseline for {run}/{stage} - run --update-baseline to record it")
    for r in regressions:
        scaled = (
            f" ({r.measured / r.scale:.2f} measured, scaled by the reference)"
            if r.scale != 1.0
            else ""
        )
        print(
            f"\nSLOWER {r.run}/{r.stage}: {r.metric} {r.measured:.2f}{scaled} "
            f"vs {r.baseline:.2f} in the baseline"
        )


def check(
    runs: list[str],
    *,
    baseline_path: Path,
    margins: Margins,
    repeats: int = REPEATS,
) -> tuple[list[Measurement], int]:
    """Measure *runs* and grade them against the saved baseline."""
    measured = measure_runs(runs, repeats=repeats)
    baseline = load_baseline(baseline_path)
    regressions = compare(measured, baseline, margins=margins)
    report(measured, regressions, unmeasured(measured, baseline))
    if unscaled(baseline):
        print(
            "\nthe baseline has no reference seconds, so seconds are compared "
            f"unscaled, at {1 + margins.unscaled_seconds:g} times the baseline - "
            "run --update-baseline on the runner to record a new one"
        )
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
    warm_up()
    t0 = time.perf_counter()
    summary = summarize_depth(price_level_volume(events))
    wall = time.perf_counter() - t0
    rss_mib = _peak_rss_mib()
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


# ──── reading saved runs in parts ───────────────────────────────────────


def _save_sample_run(folder: Path) -> None:
    """Run the bundled sample and save it to *folder*."""
    save_data(pipeline.Pipeline().run(sample_csv_path()), folder)


def _tile_file(source: Path, dest: Path, mult: int) -> int:
    """Write *source* *mult* times over to *dest*, one row group at a time.

    Each copy is moved later in time by the span of the source, and its
    ``event_id`` by the largest id, so the result reads as one long run in
    time order. Returns the number of rows written.
    """
    file = pq.ParquetFile(source)
    times = pd.read_parquet(source, columns=["timestamp", "event_id"])
    span = np.timedelta64(times["timestamp"].max() - times["timestamp"].min())
    span += np.timedelta64(1, "s")
    id_step = int(times["event_id"].max()) + 1
    del times
    schema = file.schema_arrow
    time_at = schema.get_field_index("timestamp")
    id_at = schema.get_field_index("event_id")
    rows = 0
    with pq.ParquetWriter(dest, file.schema_arrow) as writer:
        for k in range(mult):
            for batch in file.iter_batches(batch_size=ROW_GROUP_ROWS):
                table = pa.Table.from_batches([batch])
                shifted = table.column(time_at).to_numpy() + k * span
                ids = table.column(id_at).to_numpy() + k * id_step
                table = table.set_column(
                    time_at,
                    schema.field(time_at),
                    pa.array(shifted, schema.field(time_at).type),
                ).set_column(
                    id_at, schema.field(id_at), pa.array(ids, schema.field(id_at).type)
                )
                writer.write_table(table, row_group_size=ROW_GROUP_ROWS)
                rows += table.num_rows
    return rows


def _out_of_core_child(kind: str, folder: Path) -> None:
    """One measured call in a fresh process: rows, peak MiB, seconds."""
    t0 = time.perf_counter()
    if kind == "aggregate":
        out = aggregate_by_time(folder, "depth_summary", "1min", "mean")
        assert len(out) > 0
        rows = pq.ParquetFile(folder / "depth_summary.parquet").metadata.num_rows
    else:
        written = summarize_depth_file(folder, folder / "summary")
        rows = pq.ParquetFile(written).metadata.num_rows
    wall = time.perf_counter() - t0
    print(f"{rows}\t{_peak_rss_mib():.0f}\t{wall:.1f}")


def _measure_child(kind: str, folder: Path) -> tuple[int, int, float] | None:
    proc = subprocess.run(
        [sys.executable, __file__, "--out-of-core-child", kind, str(folder)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        print(f"  FAILED (rc={proc.returncode})\n{proc.stderr[-400:]}")
        return None
    rows, rss, wall = proc.stdout.strip().split("\t")
    return int(rows), int(rss), float(wall)


def out_of_core(work: Path | None = None) -> None:
    """Print the out-of-core table behind ``ARCHITECTURE.md``."""
    with tempfile.TemporaryDirectory(dir=work) as temporary:
        root = Path(temporary)
        base = root / "base"
        _save_sample_run(base)
        # The in-memory size of one copy, to state each file's size as pandas.
        one_copy = pd.read_parquet(base / "depth_summary.parquet")
        bytes_per_row = one_copy.memory_usage(deep=True).sum() / len(one_copy)
        del one_copy
        for kind, table, multiples in (
            ("aggregate", "depth_summary", AGGREGATE_MULTIPLES),
            ("summarize", "depth", SUMMARIZE_MULTIPLES),
        ):
            print(f"\n{kind}: {table}")
            print(
                f"{'rows':>12}  {'on disk (MiB)':>13}  {'as pandas (GiB)':>15}  "
                f"{'peak RSS (MiB)':>14}  {'seconds':>8}"
            )
            for mult in multiples:
                folder = root / f"{kind}-{mult}"
                folder.mkdir()
                path = folder / f"{table}.parquet"
                rows = _tile_file(base / f"{table}.parquet", path, mult)
                disk = path.stat().st_size / 1024**2
                measured = _measure_child(kind, folder)
                shutil.rmtree(folder)
                if measured is None:
                    continue
                _, rss, wall = measured
                # Only the depth_summary copy was sized; the depth table's
                # pandas size is left blank rather than guessed.
                pandas_gib = (
                    f"{rows * bytes_per_row / 1024**3:.1f}"
                    if kind == "aggregate"
                    else "-"
                )
                print(
                    f"{rows:>12,}  {disk:>13,.0f}  {pandas_gib:>15}  "
                    f"{rss:>14,}  {wall:>8.1f}"
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
    parser.add_argument(
        "--repeats",
        type=int,
        default=REPEATS,
        help=f"time each stage this many times and keep the fastest (default: {REPEATS})",
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
    parser.add_argument(
        "--out-of-core",
        action="store_true",
        help="print the table for reading saved runs in parts (see the module docstring)",
    )
    parser.add_argument(
        "--work",
        type=Path,
        help="the folder --out-of-core writes its tiled files in (default: the system's)",
    )
    parser.add_argument("--mult", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--out-of-core-child",
        nargs=2,
        metavar=("KIND", "FOLDER"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--measure-run", help=argparse.SUPPRESS)
    parser.add_argument("--synth-duration", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    global SYNTH_DURATION
    if args.synth_duration is not None:
        SYNTH_DURATION = args.synth_duration

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    if args.mult is not None:  # one child process of an --envelope run
        _measure(args.mult)
        return 0
    if args.measure_run is not None:  # one child process of a measured run
        measured = measure_run(
            args.measure_run, RUNS[args.measure_run], repeats=args.repeats
        )
        print(json.dumps({"measurements": [asdict(m) for m in measured]}))
        return 0
    if args.envelope:
        envelope()
        return 0
    if args.out_of_core_child is not None:  # one child process of --out-of-core
        kind, folder = args.out_of_core_child
        _out_of_core_child(kind, Path(folder))
        return 0
    if args.out_of_core:
        out_of_core(args.work)
        return 0

    runs = [name for name in args.runs.split(",") if name]
    unknown = [name for name in runs if name not in RUNS]
    if unknown:
        parser.error(f"unknown run(s): {', '.join(unknown)}")

    if args.update_baseline:
        measured = measure_runs(runs, repeats=args.repeats)
        save_baseline(args.baseline, measured)
        report(measured, [], [])
        print(f"\nwrote {args.baseline}")
        return 0

    margins = Margins(
        seconds=args.seconds_margin,
        peak_rss_mib=args.peak_margin,
        unscaled_seconds=DEFAULT_MARGINS.unscaled_seconds,
        floor_seconds=args.floor_seconds,
    )
    _, code = check(
        runs, baseline_path=args.baseline, margins=margins, repeats=args.repeats
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
