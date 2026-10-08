"""The capture record: what a live capture writes to ``meta.json``.

One module writes and reads it, and the Python API and the CLI apply it the
same way: a capture replays with the tick and lot size it recorded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ob_analytics import Pipeline, PipelineConfig
from ob_analytics.capture_record import (
    CaptureRecord,
    read_record,
    source_declarations,
    write_record,
)
from ob_analytics.exceptions import ConfigError
from ob_analytics.protocols import (
    Clocks,
    FeedType,
    Level,
    SequenceKind,
    TradeAttribution,
)

_T0 = 1_760_000_000_000


def _l2_capture(directory: Path, rows: list[tuple], record: dict | None) -> Path:
    """Write an L2 capture: ``depth.csv`` from *rows* and its ``meta.json``.

    *rows* are ``(side, price, volume)``, one millisecond apart.
    """
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": [_T0 + i for i in range(len(rows))],
            "exchange_timestamp": [_T0 + i for i in range(len(rows))],
            "side": [r[0] for r in rows],
            "price": [r[1] for r in rows],
            "volume": [r[2] for r in rows],
            "sequence": list(range(1, len(rows) + 1)),
            "origin": ["snapshot"] * 2 + ["stream"] * (len(rows) - 2),
        }
    ).to_csv(directory / "depth.csv", index=False)
    if record is not None:
        (directory / "meta.json").write_text(json.dumps(record))
    return directory


#: Four rows on a 0.001 grid, as a ccxt capture of a small-tick coin.
_FINE_ROWS = [
    ("bid", 0.123, 10.0),
    ("ask", 0.125, 12.0),
    ("bid", 0.124, 5.0),
    ("ask", 0.125, 0.0),
]


class TestReadRecord:
    def test_reads_every_field(self, tmp_path):
        (tmp_path / "meta.json").write_text(
            json.dumps(
                {
                    "source": "ccxt",
                    "feed_type": "price_levels",
                    "trade_attribution": "none",
                    "sequence_kind": "monotonic",
                    "clocks": "receive_only",
                    "sequence_restarts": 2,
                    "tick_size": 0.001,
                    "lot_size": 1.0,
                    "book_updates": 7,
                }
            )
        )
        assert read_record(tmp_path) == CaptureRecord(
            source="ccxt",
            feed_type=FeedType.PRICE_LEVELS,
            trade_attribution=TradeAttribution.NONE,
            sequence_kind=SequenceKind.MONOTONIC,
            clocks=Clocks.RECEIVE_ONLY,
            sequence_restarts=2,
            tick_size=0.001,
            lot_size=1.0,
        )

    def test_a_file_in_the_capture_finds_it(self, tmp_path):
        (tmp_path / "meta.json").write_text('{"tick_size": 0.001}')
        (tmp_path / "depth.csv").write_text("")
        assert read_record(tmp_path / "depth.csv").tick_size == 0.001

    @pytest.mark.parametrize("text", [None, '{"tick_size": null}', "not json", "[]"])
    def test_records_nothing_without_a_readable_value(self, tmp_path, text):
        if text is not None:
            (tmp_path / "meta.json").write_text(text)
        assert read_record(tmp_path) == CaptureRecord()

    def test_input_that_is_not_a_path_has_no_record(self):
        assert read_record(pd.DataFrame()) == CaptureRecord()

    def test_an_unknown_declaration_is_warned_about_and_left_out(self, tmp_path):
        (tmp_path / "meta.json").write_text('{"feed_type": "sideways"}')
        with pytest.warns(UserWarning, match="feed_type"):
            assert read_record(tmp_path).feed_type is None

    @pytest.mark.parametrize("value", ['"n/a"', "-3", "2.9", "true"])
    def test_a_restart_count_that_is_not_a_count_counts_none(self, tmp_path, value):
        (tmp_path / "meta.json").write_text(f'{{"sequence_restarts": {value}}}')
        assert read_record(tmp_path).sequence_restarts == 0

    @pytest.mark.parametrize("value", ['"inf"', '"nan"', "true", "-0.01", '"x"'])
    def test_a_step_that_is_not_a_step_is_not_recorded(self, tmp_path, value):
        (tmp_path / "meta.json").write_text(f'{{"tick_size": {value}}}')
        assert read_record(tmp_path).tick_size is None

    def test_the_instrument_gives_the_config_fields(self):
        record = CaptureRecord(tick_size=0.001, lot_size=1.0)
        assert record.instrument() == {
            "tick_size": 0.001,
            "price_decimals": 3,
            "lot_size": 1.0,
            "volume_decimals": 0,
        }
        assert CaptureRecord().instrument() == {}


class _Declaring:
    """A live source's declarations, without the live machinery."""

    name = "example"
    level = Level.L2
    feed_type = FeedType.PRICE_LEVELS
    trade_attribution = TradeAttribution.NONE
    sequence_kind = SequenceKind.MONOTONIC
    clocks = Clocks.RECEIVE_ONLY


class TestWriteRecord:
    def test_what_is_written_reads_back(self, tmp_path):
        write_record(
            tmp_path,
            {**source_declarations(_Declaring()), "tick_size": 0.5, "lot_size": 0.1},
        )
        assert read_record(tmp_path) == CaptureRecord(
            source="example",
            feed_type=FeedType.PRICE_LEVELS,
            trade_attribution=TradeAttribution.NONE,
            sequence_kind=SequenceKind.MONOTONIC,
            clocks=Clocks.RECEIVE_ONLY,
            tick_size=0.5,
            lot_size=0.1,
        )

    def test_leaves_no_temporary_file(self, tmp_path):
        write_record(tmp_path, {"source": "example"})
        assert [p.name for p in tmp_path.iterdir()] == ["meta.json"]


class TestPipelineAppliesTheRecord:
    def test_a_capture_replays_on_its_recorded_tick(self, tmp_path):
        cap = _l2_capture(tmp_path / "cap", _FINE_ROWS, {"tick_size": 0.001})
        result = Pipeline.from_source("depth_csv").run(cap)

        assert result.config.tick_size == 0.001
        assert result.config.price_decimals == 3
        assert sorted(result.depth["price"].unique().tolist()) == [123, 124, 125]

    def test_a_file_in_the_capture_is_read_the_same_way(self, tmp_path):
        cap = _l2_capture(tmp_path / "cap", _FINE_ROWS, {"tick_size": 0.001})
        result = Pipeline.from_source("depth_csv").run(cap / "depth.csv")
        assert result.config.tick_size == 0.001

    def test_the_recorded_lot_size_is_applied(self, tmp_path):
        cap = _l2_capture(
            tmp_path / "cap", _FINE_ROWS, {"tick_size": 0.001, "lot_size": 1.0}
        )
        result = Pipeline.from_source("depth_csv").run(cap)

        assert result.config.lot_size == 1.0
        assert result.config.volume_decimals == 0
        assert result.depth["volume"].tolist() == [10, 12, 5, 0]

    def test_a_tick_the_caller_set_is_kept(self, tmp_path):
        from ob_analytics.depth_l2 import DepthCsvSource

        cap = _l2_capture(tmp_path / "cap", _FINE_ROWS, {"tick_size": 0.001})
        pipeline = Pipeline(
            PipelineConfig(tick_size=0.0005, price_decimals=4),
            source=DepthCsvSource(),
        )
        result = pipeline.run(cap)

        assert result.config.tick_size == 0.0005
        assert sorted(result.depth["price"].unique().tolist()) == [246, 248, 250]

    def test_other_settings_the_caller_made_are_kept(self, tmp_path):
        from ob_analytics.depth_l2 import DepthCsvSource

        cap = _l2_capture(tmp_path / "cap", _FINE_ROWS, {"tick_size": 0.001})
        result = Pipeline(PipelineConfig(depth_bps=50), source=DepthCsvSource()).run(
            cap
        )

        assert result.config.depth_bps == 50
        assert result.config.tick_size == 0.001

    def test_run_windows_uses_the_record_too(self, tmp_path):
        from ob_analytics.data import load_result

        cap = _l2_capture(tmp_path / "cap", _FINE_ROWS, {"tick_size": 0.001})
        Pipeline.from_source("depth_csv").run_windows(
            cap, [pd.Timestamp(_T0 + 2, unit="ms")], tmp_path / "out"
        )
        assert load_result(tmp_path / "out").config.tick_size == 0.001

    def test_a_declaration_the_run_does_not_use_is_not_warned_about(self, tmp_path):
        import warnings

        record = {"tick_size": 0.001, "sequence_kind": "plugin_custom"}
        cap = _l2_capture(tmp_path / "cap", _FINE_ROWS, record)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = Pipeline.from_source("depth_csv").run(cap)
        assert result.config.tick_size == 0.001

    def test_a_size_between_lots_is_warned_about(self, tmp_path):
        rows = [("bid", 0.123, 10.004), ("ask", 0.125, 12.0)]
        cap = _l2_capture(
            tmp_path / "cap", rows, {"tick_size": 0.001, "lot_size": 0.01}
        )
        with pytest.warns(UserWarning, match="lot_size"):
            Pipeline.from_source("depth_csv").run(cap)

    def test_a_large_size_on_the_lot_grid_is_not_warned_about(self, tmp_path):
        # 1251.81488791 / 1e-8 is 125181488790.99998 in float arithmetic.
        import warnings

        rows = [("bid", 0.123, 1251.81488791), ("ask", 0.125, 95603.47115)]
        cap = _l2_capture(tmp_path / "cap", rows, {"tick_size": 0.001})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = Pipeline.from_source("depth_csv").run(cap)
        assert result.depth["volume"].tolist() == [125_181_488_791, 9_560_347_115_000]

    def test_a_level_too_large_for_the_lot_size_is_refused(self, tmp_path):
        # A low-priced token: 1.2e11 units at the default 1e-8 lot overflow.
        rows = [("bid", 0.00001, 5e10), ("ask", 0.00002, 1.2e11)]
        cap = _l2_capture(tmp_path / "cap", rows, {"tick_size": 1e-8})
        with pytest.raises(ConfigError, match="lot_size"):
            Pipeline.from_source("depth_csv").run(cap)

    def test_a_level_fits_on_the_venue_size_step(self, tmp_path):
        rows = [("bid", 0.00001, 5e10), ("ask", 0.00002, 1.2e11)]
        cap = _l2_capture(tmp_path / "cap", rows, {"tick_size": 1e-8, "lot_size": 1.0})
        result = Pipeline.from_source("depth_csv").run(cap)
        assert result.depth["volume"].tolist() == [50_000_000_000, 120_000_000_000]


def test_the_cli_and_the_python_api_agree_on_a_capture(cli_runner, tmp_path):
    from ob_analytics.data import load_result

    cap = _l2_capture(
        tmp_path / "cap", _FINE_ROWS, {"source": "ccxt", "tick_size": 0.001}
    )
    out = tmp_path / "out"
    r = cli_runner("process", str(cap), "--source", "depth_csv", "--output", str(out))
    assert r.returncode == 0, r.stderr

    direct = Pipeline.from_source("depth_csv").run(cap)
    processed = load_result(out)
    assert processed.config == direct.config
    for name in ("trades", "depth", "depth_summary"):
        pd.testing.assert_frame_equal(
            getattr(processed, name), getattr(direct, name), check_categorical=False
        )
