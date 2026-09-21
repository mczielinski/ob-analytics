#!/usr/bin/env python3
"""Size a Databento MBO query, download one window, and replay it.

A whole market-by-order day for every symbol runs to billions of records, which
is far past what ob-analytics holds in memory. One instrument over a
session-length window is not: see
[Scale and chunking](../docs/scale-and-chunking.md). This script is that
workflow as runnable code, in three steps:

1. **Size it first.** ``metadata.get_record_count`` and ``metadata.get_cost``
   answer how many records and how many dollars a query is before anything is
   downloaded. The count is compared against the ~5M-event envelope.
2. **Download one window.** ``timeseries.get_range`` writes a ``.dbn.zst`` file
   and reuses it on a second run.
3. **Replay it.** The file goes through ``Pipeline(source=DatabentoSource())``.
   Pass more than one window and each is run on its own and the results are
   concatenated, so peak memory is bounded by the largest window rather than by
   the whole day.

Usage::

    export DATABENTO_API_KEY=db-...
    uv run --extra databento python scripts/databento_window.py \\
        --dataset XNAS.ITCH --symbol AAPL \\
        --window 2024-02-12T14:30 2024-02-12T16:00

    # size the query and stop
    uv run --extra databento python scripts/databento_window.py \\
        --dataset XNAS.ITCH --symbol AAPL \\
        --window 2024-02-12T14:30 2024-02-12T21:00 --size-only

Databento data needs your own API key and subscription, so nothing here is
bundled with the package and the script downloads only what you ask it for.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# The comfortable in-memory ceiling from ARCHITECTURE.md's scale envelope:
# peak RSS grows at roughly 1 GiB per million events.
EVENT_ENVELOPE = 5_000_000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="e.g. XNAS.ITCH, GLBX.MDP3")
    parser.add_argument("--symbol", required=True, help="raw symbol, e.g. AAPL")
    parser.add_argument(
        "--window",
        nargs=2,
        action="append",
        metavar=("START", "END"),
        required=True,
        help="UTC start and end of one window; repeat to slice a longer span",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("databento-windows"),
        help="directory the .dbn.zst files are kept in (default: %(default)s)",
    )
    parser.add_argument(
        "--tick-size",
        type=float,
        default=0.01,
        help="the instrument's tick, in the quote currency (default: %(default)s)",
    )
    parser.add_argument(
        "--size-only",
        action="store_true",
        help="report record count and cost, then stop",
    )
    return parser.parse_args(argv)


def size_window(client, args, start: str, end: str) -> int:
    """Report what one window holds and what it costs, and return the count."""
    query = {
        "dataset": args.dataset,
        "symbols": [args.symbol],
        "schema": "mbo",
        "start": start,
        "end": end,
    }
    count = client.metadata.get_record_count(**query)
    cost = client.metadata.get_cost(**query)
    verdict = "fits" if count <= EVENT_ENVELOPE else "OVER the envelope"
    print(
        f"{start} → {end}: {count:,} records, ${cost:,.2f} "
        f"({verdict}; the envelope is {EVENT_ENVELOPE:,})"
    )
    return count


def fetch_window(client, args, start: str, end: str) -> Path:
    """Download one window to a local file, or reuse the one already there."""
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}-{args.symbol}-{start}-{end}".replace(":", "")
    path = args.out / f"{stem}.mbo.dbn.zst"
    if path.exists():
        print(f"reusing {path}")
        return path
    print(f"downloading {path}")
    client.timeseries.get_range(
        dataset=args.dataset,
        symbols=[args.symbol],
        schema="mbo",
        start=start,
        end=end,
        path=path,
    )
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        import databento as db
    except ImportError:
        print(
            'databento is not installed: pip install "ob-analytics[databento]"',
            file=sys.stderr,
        )
        return 1

    from ob_analytics import Pipeline, PipelineConfig, RunContext
    from ob_analytics.databento import DatabentoSource

    client = db.Historical()  # reads DATABENTO_API_KEY from the environment

    counts = [size_window(client, args, start, end) for start, end in args.window]
    print(f"total: {sum(counts):,} records over {len(counts)} window(s)")
    if args.size_only:
        return 0
    if any(count > EVENT_ENVELOPE for count in counts):
        print(
            "at least one window is over the envelope — split it further before "
            "running, or expect the run to need more memory than this machine has",
            file=sys.stderr,
        )

    paths = [fetch_window(client, args, start, end) for start, end in args.window]

    # Each window is its own run, so peak memory is bounded by the largest one.
    # What concatenates cleanly and what does not is set out in
    # docs/scale-and-chunking.md: windowed views do, whole-book per-order
    # questions (queue position, order lifetimes) do not span a cut.
    config = PipelineConfig(tick_size=args.tick_size)
    ctx = RunContext(symbol=args.symbol, venue=args.dataset)
    results = [
        Pipeline(config, source=DatabentoSource(), ctx=ctx).run(path) for path in paths
    ]

    events = pd.concat([r.events for r in results], ignore_index=True)
    trades = pd.concat([r.trades for r in results], ignore_index=True)
    depth = pd.concat([r.depth for r in results], ignore_index=True)
    print(f"{len(events):,} events, {len(trades):,} trades, {len(depth):,} depth rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
