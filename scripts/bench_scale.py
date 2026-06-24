#!/usr/bin/env python3
"""Measure peak RSS and wall-time of the depth pipeline vs event count.

Throwaway benchmark for the docs "Scale envelope" section (roadmap WS-8.4a).
ob-analytics is in-memory pandas, so this probes how memory grows with the event
count to ground an honest ceiling. The bundled ~314k-event sample is tiled (ids
and timestamps offset per copy) up to each target size; the depth stages
(``price_level_volume`` -> ``depth_metrics``, the memory/CPU hot path) run on it.

Each size runs in its own subprocess so peak RSS is a clean per-process
high-water mark (``getrusage(RUSAGE_SELF).ru_maxrss``).

    uv run python scripts/bench_scale.py
"""

from __future__ import annotations

import resource
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from ob_analytics import sample_csv_path
from ob_analytics.analytics import set_order_types
from ob_analytics.bitstamp import BitstampLoader, BitstampTradeReader
from ob_analytics.depth import depth_metrics, price_level_volume

MULTIPLES = [1, 2, 3, 4]


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


def main() -> None:
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


if __name__ == "__main__":
    main()
