"""Tests for ``scripts/databento_bbo_check.py`` (issue #425) -- no network, no key.

The check compares the book rebuilt from a Databento MBO file with Databento's
own top of book. Here the MBO side is the short stream in :mod:`tests._mbo`,
and the top of book is built by hand from what that stream's book must be, so
a correct rebuild matches it on every instant. The tests that read or write a
``.dbn`` file skip without the ``databento`` extra.
"""

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from ob_analytics.databento import UNDEF_PRICE
from scripts import databento_bbo_check as check
from tests._mbo import MBO, T0, mbo_frame, px

requires_databento = pytest.mark.skipif(
    importlib.util.find_spec("databento") is None,
    reason="needs the databento extra",
)

TICK, LOT = 0.01, 1.0
INSTRUMENT_ID, PUBLISHER_ID = 1108, 2

#: The top of book after each millisecond of :data:`tests._mbo.MBO`:
#: ``(ms, bid, bid size, ask, ask size)``, ``None`` for an empty side.
TOP = [
    (0, 100.25, 100, None, 0),
    (1, 100.25, 100, None, 0),
    (2, 100.25, 100, 100.35, 50),
    (3, 100.25, 100, 100.35, 50),
    (4, 100.25, 60, 100.35, 50),
    (5, 100.25, 60, 100.40, 80),
    (6, 100.25, 60, 100.40, 50),
    (7, 100.30, 70, 100.40, 50),
]


def ts_recv(ms: float) -> int:
    """The receive time, in ns, of an MBO record at *ms*, as in :func:`mbo_frame`."""
    return T0 + round(ms * 1_000_000) + 500


def bbo_frame(top: list[tuple] = TOP) -> pd.DataFrame:
    """*top* as the frame ``DBNStore.to_df(price_type="fixed")`` gives for mbp-1."""
    return pd.DataFrame(
        {
            "ts_recv": pd.to_datetime([ts_recv(r[0]) for r in top], utc=True),
            "bid_px_00": [UNDEF_PRICE if r[1] is None else px(r[1]) for r in top],
            "bid_sz_00": [r[2] for r in top],
            "ask_px_00": [UNDEF_PRICE if r[3] is None else px(r[3]) for r in top],
            "ask_sz_00": [r[4] for r in top],
            "instrument_id": INSTRUMENT_ID,
            "publisher_id": PUBLISHER_ID,
        }
    )


@pytest.fixture(scope="module")
def rebuilt() -> tuple[pd.DataFrame, pd.DataFrame]:
    return check.rebuild_touch(mbo_frame())


def compare(touch: pd.DataFrame, top: list[tuple] = TOP) -> check.Comparison:
    return check.compare(
        touch, check.read_bbo_frame(bbo_frame(top)), tick_size=TICK, lot_size=LOT
    )


class TestCompare:
    def test_a_correct_book_matches_every_instant(self, rebuilt):
        result = compare(rebuilt[0])

        assert len(result.rows) == len(TOP)
        assert result.price_match_rate == 1.0
        assert result.full_match_rate == 1.0
        assert result.off_grid == 0
        assert check.episodes(result).empty

    def test_a_wrong_price_is_one_episode_on_its_side(self, rebuilt):
        # Databento keeps the 100.35 ask for two more instants: the rebuilt
        # book took it off at 5 ms.
        top = [*TOP[:5], (5, 100.25, 60, 100.35, 50), (6, 100.25, 60, 100.35, 50)]
        top.append(TOP[7])
        result = compare(rebuilt[0], top)

        assert result.price_match_rate == pytest.approx(6 / 8)
        runs = check.episodes(result)
        assert runs[["first", "last", "records", "sides", "wrong"]].to_dict(
            "records"
        ) == [{"first": 5, "last": 6, "records": 2, "sides": "ask", "wrong": "price"}]

    def test_a_wrong_size_counts_apart_from_the_price(self, rebuilt):
        top = [*TOP[:4], (4, 100.25, 100, 100.35, 50), *TOP[5:]]
        result = compare(rebuilt[0], top)

        assert result.price_match_rate == 1.0
        assert result.full_match_rate == pytest.approx(7 / 8)
        runs = check.episodes(result)
        assert runs[["first", "records", "sides", "wrong"]].to_dict("records") == [
            {"first": 4, "records": 1, "sides": "bid", "wrong": "size"}
        ]
        assert len(result.mismatches) == 1

    def test_separate_runs_are_separate_episodes(self, rebuilt):
        top = list(TOP)
        top[1] = (1, 100.20, 70, None, 0)
        top[3] = (3, 100.25, 100, 100.40, 80)
        top[4] = (4, 100.20, 70, 100.40, 80)
        runs = check.episodes(compare(rebuilt[0], top))

        assert runs[["first", "last", "sides"]].to_dict("records") == [
            {"first": 1, "last": 1, "sides": "bid"},
            {"first": 3, "last": 4, "sides": "both"},
        ]

    def test_only_the_last_record_of_an_instant_is_compared(self, rebuilt):
        # At 7 ms the modify moves the bid: the rebuilt summary has two rows
        # at that instant, and a top-of-book file can hold the state between
        # them too. Only the state after the whole instant is compared.
        touch = rebuilt[0]
        assert touch["timestamp"].eq(pd.Timestamp(ts_recv(7), tz="UTC")).sum() == 2
        top = [*TOP[:7], (7, 100.25, 60, 100.40, 50), TOP[7]]
        result = compare(touch, top)

        assert len(result.rows) == len(TOP)
        assert result.price_match_rate == 1.0

    def test_an_instant_between_book_updates_reads_the_book_before_it(self, rebuilt):
        # A once-a-second file samples between the MBO records.
        top = [(-1, None, 0, None, 0), (2.5, 100.25, 100, 100.35, 50)]
        result = compare(rebuilt[0], top)

        assert result.price_match_rate == 1.0
        assert result.rows["book_event_id"].isna().tolist() == [True, False]

    def test_a_windowed_rebuild_compares_the_same(self, rebuilt):
        cuts = [str(pd.Timestamp(ts_recv(3.5), tz="UTC"))]
        touch, events = check.rebuild_touch(mbo_frame(), cuts=cuts)

        pd.testing.assert_frame_equal(compare(touch).rows, compare(rebuilt[0]).rows)
        assert events["event_id"].is_monotonic_increasing

    def test_a_price_off_the_tick_is_counted(self, rebuilt):
        top = [*TOP[:7], (7, 100.305, 70, 100.40, 50)]
        result = compare(rebuilt[0], top)

        assert result.off_grid == 1

    def test_the_report_names_the_rate_and_the_events_before(self, rebuilt):
        top = [*TOP[:5], (5, 100.25, 60, 100.35, 50), *TOP[6:]]
        text = check.report(
            compare(rebuilt[0], top), rebuilt[1], tick_size=TICK, lot_size=LOT
        )

        assert "price match (best bid and ask): 87.5000%" in text
        assert "price mismatches: 1" in text
        assert "1 episodes" in text
        assert "ask price wrong" in text
        assert "databento: bid 100.25 x 60  ask 100.35 x 50" in text
        assert "rebuilt:   bid 100.25 x 60  ask 100.4 x 80" in text
        assert "the MBO events before it:" in text
        assert "deleted" in text


class TestReadBbo:
    def test_two_instruments_need_one_named(self):
        frame = bbo_frame()
        frame.loc[0, "instrument_id"] = 9
        with pytest.raises(ValueError, match="--instrument-id"):
            check.read_bbo_frame(frame)
        kept = check.read_bbo_frame(frame, instrument_id=INSTRUMENT_ID)
        assert len(kept) == len(TOP) - 1

    def test_no_records_is_refused(self):
        with pytest.raises(ValueError, match="no top-of-book records"):
            check.read_bbo_frame(bbo_frame(), publisher_id=99)


# ── Real .dbn files ───────────────────────────────────────────────────


def _metadata(schema: str) -> bytes:
    from databento_dbn import Metadata, Schema, SType

    meta = Metadata(
        dataset="XNAS.ITCH",
        start=T0,
        stype_in=SType.from_str("raw_symbol"),
        stype_out=SType.from_str("instrument_id"),
        schema=Schema.from_str(schema),
        symbols=["AAPL"],
    )
    return bytes(meta.encode())


def mbo_dbn() -> bytes:
    """:data:`tests._mbo.MBO` as a DBN MBO file."""
    from databento_dbn import Action, MBOMsg, Side

    payload = bytearray(_metadata("mbo"))
    for number, (ms, order_id, action, side, price, size) in enumerate(MBO, 1):
        payload += bytes(
            MBOMsg(
                publisher_id=PUBLISHER_ID,
                instrument_id=INSTRUMENT_ID,
                ts_event=T0 + ms * 1_000_000,
                order_id=order_id,
                price=price,
                size=size,
                action=Action.from_str(action),
                side=Side.from_str(side),
                ts_recv=ts_recv(ms),
                sequence=number,
            )
        )
    return bytes(payload)


def mbp1_dbn(top: list[tuple] = TOP) -> bytes:
    """*top* as a DBN mbp-1 file."""
    from databento_dbn import Action, BidAskPair, MBP1Msg, Side

    payload = bytearray(_metadata("mbp-1"))
    # databento-dbn's stubs declare ``levels`` as a list, but the constructor
    # takes one ``BidAskPair`` and refuses a list.
    for ms, bid, bid_sz, ask, ask_sz in top:
        payload += bytes(
            MBP1Msg(
                publisher_id=PUBLISHER_ID,
                instrument_id=INSTRUMENT_ID,
                ts_event=T0 + ms * 1_000_000,
                price=px(100.0),
                size=0,
                action=Action.from_str("A"),
                side=Side.from_str("N"),
                depth=0,
                ts_recv=ts_recv(ms),
                levels=BidAskPair(
                    bid_px=UNDEF_PRICE if bid is None else px(bid),
                    ask_px=UNDEF_PRICE if ask is None else px(ask),
                    bid_sz=bid_sz,
                    ask_sz=ask_sz,
                    bid_ct=int(bid_sz > 0),
                    ask_ct=int(ask_sz > 0),
                ),  # ty: ignore[invalid-argument-type]
            )
        )
    return bytes(payload)


@requires_databento
class TestFiles:
    def test_an_mbp1_file_reads_as_the_frame(self, tmp_path):
        path = tmp_path / "top.mbp-1.dbn"
        path.write_bytes(mbp1_dbn())

        from_file = check.read_bbo_frame(path)
        from_frame = check.read_bbo_frame(bbo_frame())
        pd.testing.assert_frame_equal(
            from_file, from_frame, check_dtype=False, check_index_type=False
        )

    def test_another_schema_is_refused(self, tmp_path):
        path = tmp_path / "orders.mbo.dbn"
        path.write_bytes(mbo_dbn())
        with pytest.raises(ValueError, match="'mbo' schema"):
            check.read_bbo_frame(path)

    def test_main_reports_a_full_match(self, tmp_path, capsys):
        mbo, top = tmp_path / "orders.mbo.dbn", tmp_path / "top.mbp-1.dbn"
        mbo.write_bytes(mbo_dbn())
        top.write_bytes(mbp1_dbn())
        csv = tmp_path / "mismatches.csv"

        assert check.main([str(mbo), str(top), "--csv", str(csv)]) == 0

        out = capsys.readouterr().out
        assert "compared 8 top-of-book instants" in out
        assert "price match (best bid and ask): 100.0000%" in out
        assert len(pd.read_csv(csv)) == 0


def test_help_runs_through_argparse(capsys):
    with pytest.raises(SystemExit) as exit_:
        check.parse_args(["--help"])
    assert exit_.value.code == 0
    assert "Databento's own BBO" in capsys.readouterr().out
