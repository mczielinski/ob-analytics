"""Hidden executions (LOBSTER event_type=5) must keep id=0.

After Plan 1 (remove-pacman) deleted the synthetic-id renumbering
workaround, hidden execs flow through with the native LOBSTER convention
(id=0). Lock that in.
"""

from pathlib import Path

from ob_analytics.lobster import LobsterLoader


def _write_minimal_message_csv(tmp_path: Path) -> Path:
    """Write a 3-row LOBSTER message file: one new limit order, then a
    hidden execution, then a deletion. Hidden execs use id=0 in the
    native format.

    LOBSTER columns: time, event_type, id, volume, price, direction
    (direction: 1=bid, -1=ask; price is integer * price_divisor).
    """
    p = tmp_path / "AAPL_2012-06-21_34200000_57600000_message_5.csv"
    p.write_text(
        "34200.000000000,1,11111,100,1000000,1\n"  # new bid order
        "34200.500000000,5,0,50,1000000,-1\n"  # hidden ask execution: id=0
        "34201.000000000,3,11111,100,1000000,1\n"  # delete the bid
    )
    return p


def test_hidden_executions_keep_id_zero(tmp_path: Path) -> None:
    msg_path = _write_minimal_message_csv(tmp_path)
    loader = LobsterLoader(trading_date="2012-06-21")
    events = loader.load(msg_path)

    hidden = events[events["raw_event_type"] == 5]
    assert len(hidden) == 1, "expected exactly one hidden execution row"
    assert (hidden["id"] == 0).all(), (
        "hidden executions must retain native LOBSTER id=0 — the synthetic "
        "renumbering workaround was removed in Plan 1"
    )
