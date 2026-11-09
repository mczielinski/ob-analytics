#!/usr/bin/env python3
"""Check the book rebuilt from a Databento MBO file against Databento's own BBO.

ob-analytics rebuilds the order book from Databento's market-by-order (MBO)
records. Databento also publishes the best bid and offer (BBO) for the same
instrument, built by Databento from the same feed. This script compares the
two, so the rebuilt book is checked against something other than itself.

It takes one MBO file and one top-of-book file for the same instrument and
time range. The top-of-book file can hold any of these schemas:

* ``mbp-1``: one record each time the top of the book changes. This is the
  stronger check, because every change is compared.
* ``bbo-1s`` or ``bbo-1m``: the best bid and offer once a second or once a
  minute. Smaller and cheaper, but a short wrong state between two samples is
  not seen.

The MBO file must start at the start of the venue's day (for ``XNAS.ITCH``,
midnight UTC). The rebuilt book starts empty, so orders resting before the
file starts are missing from it, and the touch is wrong until they leave.

How the two are lined up
------------------------
Both files stamp each record with ``ts_recv``, Databento's receive time. For
each top-of-book record the script takes the rebuilt book as it stood after
every MBO record received at or before that time: the last row of the run's
``depth_summary`` at or before it. Several records received at the same
instant are one update, so on both sides only the last row of each instant is
compared.

A record matches when the best bid price and the best ask price are both the
same. Sizes are compared too and reported apart, since a price can be right
while the size at it is wrong. An empty side is a price of zero on both sides.

The report gives the match rates and groups the mismatches into episodes: runs
of consecutive mismatched records, whether a price or only a size is wrong.
For the largest episodes it prints both books and the MBO events just before
the first mismatch, which is usually enough to see why the two disagree.

Usage::

    uv run --extra databento python scripts/databento_bbo_check.py \\
        XNAS.ITCH-AAPL-20240212.mbo.dbn.zst \\
        XNAS.ITCH-AAPL-20240212.mbp-1.dbn.zst

    # a file of several million records: rebuild one window at a time
    uv run --extra databento python scripts/databento_bbo_check.py \\
        mbo.dbn.zst mbp-1.dbn.zst --cut 2024-02-12T12:00 --cut 2024-02-12T14:00

Both files need a Databento account to download, so none is bundled with the
package. ``docs/data-quality.md`` describes the check and records its result.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: The top-of-book schemas this check reads. Each carries the best bid and
#: offer as ``bid_px_00`` / ``ask_px_00`` / ``bid_sz_00`` / ``ask_sz_00``.
BBO_SCHEMAS = ("mbp-1", "bbo-1s", "bbo-1m")

#: The columns of a top-of-book record this check reads.
BBO_COLUMNS = ["ts_recv", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]

#: The columns of the rebuilt ``depth_summary`` this check reads.
TOUCH_COLUMNS = [
    "timestamp",
    "event_id",
    "best_bid_price",
    "best_bid_vol",
    "best_ask_price",
    "best_ask_vol",
]

#: The columns of the rebuilt ``events`` printed before a mismatch.
EVENT_COLUMNS = [
    "event_id",
    "timestamp",
    "id",
    "raw_event_type",
    "action",
    "direction",
    "price",
    "volume",
    "fill",
]

#: How many top-of-book records are read from a file at a time.
READ_SIZE = 1_000_000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("mbo", type=Path, help="the MBO file (.dbn or .dbn.zst)")
    parser.add_argument(
        "bbo",
        type=Path,
        help=f"the top-of-book file for the same instrument: {', '.join(BBO_SCHEMAS)}",
    )
    parser.add_argument(
        "--instrument-id",
        type=int,
        help="the instrument to check, when a file holds more than one",
    )
    parser.add_argument(
        "--publisher-id",
        type=int,
        help="the publisher to check, when a file holds more than one",
    )
    parser.add_argument(
        "--tick-size",
        type=float,
        help="the instrument's tick, in the quote currency (default: 0.01)",
    )
    parser.add_argument(
        "--lot-size",
        type=float,
        help="the instrument's lot, in its own units (default: 1)",
    )
    parser.add_argument(
        "--cut",
        action="append",
        default=[],
        metavar="TIME",
        help=(
            "rebuild the book one window at a time, cut at this UTC time; "
            "repeat for more cuts. Use it for a file of several million records"
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="how many mismatch episodes to print in full (default: %(default)s)",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=8,
        help="how many MBO events to print before each one (default: %(default)s)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="write every mismatched record to this CSV file",
    )
    return parser.parse_args(argv)


# ── Reading the two sides ─────────────────────────────────────────────


def read_bbo_frame(
    source: Any,
    *,
    instrument_id: int | None = None,
    publisher_id: int | None = None,
) -> pd.DataFrame:
    """Return the top-of-book records of *source*, one instrument's only.

    Parameters
    ----------
    source : str, Path or pandas.DataFrame
        A ``.dbn`` / ``.dbn.zst`` file holding one of :data:`BBO_SCHEMAS`, or
        a frame of its records as ``DBNStore.to_df(price_type="fixed")`` gives
        them.
    instrument_id, publisher_id : int, optional
        Keep only the records of this instrument and publisher.

    Returns
    -------
    pandas.DataFrame
        :data:`BBO_COLUMNS`, with Databento's raw fixed-point prices.

    Raises
    ------
    ValueError
        If the file holds another schema, or the records cover more than one
        instrument or publisher and none was named.
    """

    def narrow(part: pd.DataFrame) -> pd.DataFrame:
        if instrument_id is not None:
            part = part[part["instrument_id"] == instrument_id]
        if publisher_id is not None:
            part = part[part["publisher_id"] == publisher_id]
        return part[[*BBO_COLUMNS, "instrument_id", "publisher_id"]]

    if isinstance(source, pd.DataFrame):
        kept = [narrow(source.reset_index() if source.index.name else source)]
    else:
        import databento as db

        from ob_analytics.databento import _close_dbn

        store = db.DBNStore.from_file(Path(source))
        try:
            schema = str(store.metadata.schema or "")
            if schema not in BBO_SCHEMAS:
                raise ValueError(
                    f"{source} holds the {schema!r} schema; this check reads "
                    f"{', '.join(BBO_SCHEMAS)}"
                )
            kept = [
                narrow(part.reset_index())
                for part in store.to_df(price_type="fixed", count=READ_SIZE)
            ]
        finally:
            _close_dbn(store)

    if not kept or not sum(map(len, kept)):
        raise ValueError(f"{source} holds no top-of-book records to compare")
    frame = pd.concat(kept, ignore_index=True)

    for column in ("instrument_id", "publisher_id"):
        values = frame[column].unique()
        if len(values) > 1:
            raise ValueError(
                f"the top-of-book records hold {len(values)} values of {column} "
                f"({', '.join(map(str, sorted(values)[:5]))}); name one with "
                f"--{column.replace('_', '-')}"
            )
    return frame[BBO_COLUMNS]


def rebuild_touch(
    mbo: Any,
    *,
    config: Any = None,
    settings: Any = None,
    cuts: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the book from *mbo* and return its touch and its events.

    The book is rebuilt by :class:`~ob_analytics.pipeline.Pipeline` with
    :class:`~ob_analytics.databento.DatabentoSource`: the same code a user
    runs. With *cuts*, it runs one window at a time through
    :meth:`~ob_analytics.pipeline.Pipeline.run_windows`, which carries the book
    across each cut, and only the columns this check needs are read back.

    Returns
    -------
    tuple of pandas.DataFrame
        The run's ``depth_summary`` (:data:`TOUCH_COLUMNS`) and ``events``
        (:data:`EVENT_COLUMNS`), prices in integer ticks and sizes in lots.
    """
    from ob_analytics import Pipeline
    from ob_analytics.databento import DatabentoSettings, DatabentoSource

    source = DatabentoSource(settings=settings or DatabentoSettings())
    pipeline = Pipeline(config, source=source)
    if not cuts:
        result = pipeline.run(mbo)
        return result.depth_summary[TOUCH_COLUMNS], result.events[EVENT_COLUMNS]
    with tempfile.TemporaryDirectory(prefix="databento-bbo-check-") as out:
        folder = pipeline.run_windows(mbo, cuts, out)
        touch = pd.read_parquet(folder / "depth_summary.parquet", columns=TOUCH_COLUMNS)
        events = pd.read_parquet(folder / "events.parquet", columns=EVENT_COLUMNS)
    return touch, events.sort_values("event_id", kind="stable", ignore_index=True)


# ── Comparing them ────────────────────────────────────────────────────


@dataclass(frozen=True)
class Comparison:
    """The result of :func:`compare`.

    Attributes
    ----------
    rows : pandas.DataFrame
        One row per compared top-of-book instant: ``ts_recv``; Databento's
        ``bbo_bid``, ``bbo_ask`` (ticks) and ``bbo_bid_sz``, ``bbo_ask_sz``
        (lots); the rebuilt ``book_bid``, ``book_ask``, ``book_bid_sz``,
        ``book_ask_sz``; ``book_ts`` and ``book_event_id``, the rebuilt row
        that was compared (missing before the first one); and the booleans
        ``bid_ok`` and ``ask_ok`` (the price), ``bid_sz_ok`` and ``ask_sz_ok``
        (the size), ``price_ok``, ``size_ok`` and ``ok`` (all four).
    off_grid : int
        Top-of-book prices that are not a whole number of ticks. They are
        rounded onto the tick, as the MBO loader rounds its own prices, so they
        can still match; a large count means the tick size is wrong.
    """

    rows: pd.DataFrame
    off_grid: int

    @property
    def price_match_rate(self) -> float:
        """The share of instants whose best bid and ask prices both match."""
        return float(self.rows["price_ok"].mean()) if len(self.rows) else float("nan")

    @property
    def full_match_rate(self) -> float:
        """The share of instants whose prices and sizes all match."""
        return float(self.rows["ok"].mean()) if len(self.rows) else float("nan")

    @property
    def mismatches(self) -> pd.DataFrame:
        """The rows where a price or a size does not match."""
        return self.rows[~self.rows["ok"]]


def _last_per_instant(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Keep the last row of each value of *column*, in time order."""
    frame = frame.sort_values(column, kind="stable")
    return frame.drop_duplicates(column, keep="last").reset_index(drop=True)


def _bbo_in_ticks(
    bbo: pd.DataFrame, tick_size: float, lot_size: float
) -> tuple[pd.DataFrame, int]:
    """Convert the raw top-of-book prices to ticks and sizes to lots.

    An undefined price (Databento's mark for an empty side) becomes zero, as
    an empty side is in ``depth_summary``. Uses the conversion the MBO loader
    uses, so a price is rounded onto the tick the same way on both sides.
    """
    from ob_analytics._utils import off_tick_grid, price_to_ticks, size_to_lots
    from ob_analytics.databento import DBN_PRICE_DIVISOR, UNDEF_PRICE

    out = pd.DataFrame({"ts_recv": pd.to_datetime(bbo["ts_recv"], utc=True)})
    off_grid = 0
    for side in ("bid", "ask"):
        raw = bbo[f"{side}_px_00"].to_numpy(dtype="int64")
        defined = raw != UNDEF_PRICE
        price = raw[defined] / DBN_PRICE_DIVISOR
        off_grid += int(off_tick_grid(price, tick_size).sum())
        ticks = np.zeros(len(raw), dtype="int64")
        ticks[defined] = price_to_ticks(price, tick_size)
        out[f"bbo_{side}"] = ticks
        size = bbo[f"{side}_sz_00"].to_numpy(dtype="int64")
        out[f"bbo_{side}_sz"] = np.where(defined, size_to_lots(size, lot_size), 0)
    return out, off_grid


def compare(
    touch: pd.DataFrame,
    bbo: pd.DataFrame,
    *,
    tick_size: float,
    lot_size: float,
) -> Comparison:
    """Compare the rebuilt *touch* with Databento's top of book *bbo*.

    Parameters
    ----------
    touch : pandas.DataFrame
        The rebuilt ``depth_summary``, as :func:`rebuild_touch` returns it.
    bbo : pandas.DataFrame
        Databento's records, as :func:`read_bbo_frame` returns them.
    tick_size, lot_size : float
        The instrument's grid, the one the MBO file was loaded on.
    """
    quotes, off_grid = _bbo_in_ticks(bbo, tick_size, lot_size)
    quotes = _last_per_instant(quotes, "ts_recv")

    book = _last_per_instant(touch[TOUCH_COLUMNS], "timestamp").rename(
        columns={
            "timestamp": "book_ts",
            "event_id": "book_event_id",
            "best_bid_price": "book_bid",
            "best_bid_vol": "book_bid_sz",
            "best_ask_price": "book_ask",
            "best_ask_vol": "book_ask_sz",
        }
    )
    book["book_ts"] = pd.to_datetime(book["book_ts"], utc=True)

    rows = pd.merge_asof(
        quotes,
        book,
        left_on="ts_recv",
        right_on="book_ts",
        direction="backward",
        allow_exact_matches=True,
    )
    # Before the first rebuilt row the book is empty.
    for column in ("book_bid", "book_ask", "book_bid_sz", "book_ask_sz"):
        rows[column] = rows[column].fillna(0).astype("int64")
    rows["book_event_id"] = rows["book_event_id"].astype("Int64")

    rows["bid_ok"] = rows["bbo_bid"] == rows["book_bid"]
    rows["ask_ok"] = rows["bbo_ask"] == rows["book_ask"]
    rows["bid_sz_ok"] = rows["bbo_bid_sz"] == rows["book_bid_sz"]
    rows["ask_sz_ok"] = rows["bbo_ask_sz"] == rows["book_ask_sz"]
    rows["price_ok"] = rows["bid_ok"] & rows["ask_ok"]
    rows["size_ok"] = rows["bid_sz_ok"] & rows["ask_sz_ok"]
    rows["ok"] = rows["price_ok"] & rows["size_ok"]
    return Comparison(rows=rows, off_grid=off_grid)


def episodes(comparison: Comparison) -> pd.DataFrame:
    """Group the mismatches into runs of consecutive mismatched instants.

    Returns
    -------
    pandas.DataFrame
        One row per run: ``first`` and ``last`` (the row positions in
        :attr:`Comparison.rows`), ``start`` and ``end`` (their times),
        ``records`` (how many instants), ``sides`` (``"bid"``, ``"ask"`` or
        ``"both"``: which side was wrong at any point in the run), and
        ``wrong`` (``"price"`` if a price was wrong at any point in the run,
        ``"size"`` if only sizes were).
    """
    columns = ["first", "last", "start", "end", "records", "sides", "wrong"]
    rows = comparison.rows
    position = np.flatnonzero(~rows["ok"].to_numpy())
    if not len(position):
        return pd.DataFrame(columns=columns)
    bad = rows.iloc[position]
    # A new run starts wherever the previous instant matched.
    run = np.cumsum(np.r_[True, np.diff(position) > 1])
    grouped = pd.DataFrame(
        {
            "position": position,
            "ts_recv": bad["ts_recv"].to_numpy(),
            "bid": ~(bad["bid_ok"] & bad["bid_sz_ok"]).to_numpy(),
            "ask": ~(bad["ask_ok"] & bad["ask_sz_ok"]).to_numpy(),
            "price": ~bad["price_ok"].to_numpy(),
            "run": run,
        }
    ).groupby("run", sort=True)
    out = grouped.agg(
        first=("position", "first"),
        last=("position", "last"),
        start=("ts_recv", "first"),
        end=("ts_recv", "last"),
        records=("position", "size"),
        bid=("bid", "any"),
        ask=("ask", "any"),
        price=("price", "any"),
    ).reset_index(drop=True)
    out["sides"] = np.where(
        out["bid"] & out["ask"], "both", np.where(out["bid"], "bid", "ask")
    )
    out["wrong"] = np.where(out["price"], "price", "size")
    return out[columns]


# ── Reporting ─────────────────────────────────────────────────────────


def report(
    comparison: Comparison,
    events: pd.DataFrame,
    *,
    tick_size: float,
    lot_size: float,
    show: int = 10,
    context: int = 8,
) -> str:
    """Return the text report: match rates, episodes, and the largest in full."""
    rows = comparison.rows
    runs = episodes(comparison)
    lines = [
        f"compared {len(rows):,} top-of-book instants",
        f"  price match (best bid and ask): {comparison.price_match_rate:.4%}",
        f"  price and size match:           {comparison.full_match_rate:.4%}",
        f"  price mismatches: {int((~rows['price_ok']).sum()):,}",
        f"  size-only mismatches: {int((rows['price_ok'] & ~rows['size_ok']).sum()):,}",
        f"  {len(runs):,} episodes",
    ]
    if comparison.off_grid:
        lines.append(
            f"  {comparison.off_grid:,} top-of-book prices are off the "
            f"{tick_size} tick: is --tick-size right?"
        )
    if runs.empty:
        return "\n".join(lines)

    def money(ticks: Any) -> str:
        return f"{int(ticks) * tick_size:.10g}" if int(ticks) else "-"

    def lots(n: Any) -> str:
        return f"{int(n) * lot_size:g}"

    lines.append("")
    lines.append(f"episodes by size (largest {min(show, len(runs))} of {len(runs)}):")
    for _, run in runs.sort_values("records", ascending=False).head(show).iterrows():
        row = rows.iloc[run["first"]]
        lines.append("")
        lines.append(
            f"{run['start']} → {run['end']}: {run['records']:,} instants, "
            f"{run['sides']} {run['wrong']} wrong"
        )
        lines.append(
            f"  databento: bid {money(row['bbo_bid'])} x {lots(row['bbo_bid_sz'])}"
            f"  ask {money(row['bbo_ask'])} x {lots(row['bbo_ask_sz'])}"
        )
        lines.append(
            f"  rebuilt:   bid {money(row['book_bid'])} x {lots(row['book_bid_sz'])}"
            f"  ask {money(row['book_ask'])} x {lots(row['book_ask_sz'])}"
            f"  (as of {row['book_ts']}, event {row['book_event_id']})"
        )
        before = events[events["timestamp"] <= row["ts_recv"]].tail(context)
        if len(before):
            shown = before.assign(
                price=[money(p) for p in before["price"]],
                volume=[lots(v) for v in before["volume"]],
                fill=[lots(f) for f in before["fill"]],
            )
            lines.append("  the MBO events before it:")
            lines.extend(
                "    " + line for line in shown.to_string(index=False).splitlines()
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        import databento  # noqa: F401
    except ImportError:
        print(
            'databento is not installed: pip install "ob-analytics[databento]"',
            file=sys.stderr,
        )
        return 1

    from ob_analytics import PipelineConfig
    from ob_analytics.databento import DATABENTO_GRID, DatabentoSettings

    grid = {
        name: value
        for name, value in (("tick_size", args.tick_size), ("lot_size", args.lot_size))
        if value is not None
    }
    config = DATABENTO_GRID.config(PipelineConfig(**grid) if grid else None)
    settings = DatabentoSettings(
        instrument_id=args.instrument_id, publisher_id=args.publisher_id
    )

    touch, events = rebuild_touch(
        args.mbo, config=config, settings=settings, cuts=args.cut
    )
    bbo = read_bbo_frame(
        args.bbo, instrument_id=args.instrument_id, publisher_id=args.publisher_id
    )
    comparison = compare(
        touch, bbo, tick_size=config.tick_size, lot_size=config.lot_size
    )
    print(
        report(
            comparison,
            events,
            tick_size=config.tick_size,
            lot_size=config.lot_size,
            show=args.episodes,
            context=args.context,
        )
    )
    if args.csv is not None:
        comparison.mismatches.to_csv(args.csv, index=False)
        print(f"\nwrote {len(comparison.mismatches):,} mismatches to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
