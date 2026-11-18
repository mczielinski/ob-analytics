"""End-to-end smoke tests for the ob-analytics CLI.

Tests invoke ``python -m ob_analytics`` as a subprocess so the actual
argparse + entry-point wiring is exercised — not just the inner functions.
Each test uses ``tmp_path`` for output so nothing leaks.

The "happy path" process/demo tests use ``tiny_bitstamp_orders_csv`` —
the bundled sample data takes minutes through the full pipeline, which
would balloon the CLI suite past any reasonable timeout. Loader/Pipeline
behaviour on the real sample data is covered in test_bitstamp.py and
test_pipeline.py.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from tests import _lobster_few_levels as few
from tests._logging import warnings_logged

_CCXT_INSTALLED = importlib.util.find_spec("ccxt") is not None

# ---------------------------------------------------------------------------
# --help / unknown subcommands
# ---------------------------------------------------------------------------


class TestCLIBasics:
    def test_help(self, cli_runner):
        r = cli_runner("--help")
        assert r.returncode == 0
        assert "process" in r.stdout
        assert "gallery" in r.stdout
        assert "bitstamp-demo" in r.stdout
        assert "lobster-demo" in r.stdout

    def test_no_args_exits_nonzero(self, cli_runner):
        r = cli_runner()
        # argparse requires a subcommand; we don't pin which exit code.
        assert r.returncode != 0

    def test_unknown_subcommand(self, cli_runner):
        r = cli_runner("nonsense-cmd")
        assert r.returncode != 0

    def test_a_missing_input_file_is_one_line(self, cli_runner, tmp_path):
        """A missing file is the user's to fix: one line on stderr, exit 1."""
        missing = tmp_path / "nowhere" / "orders.csv"
        r = cli_runner("process", str(missing), "-o", str(tmp_path / "out"))
        assert r.returncode == 1
        assert "Traceback" not in r.stderr
        assert "| ERROR" not in r.stderr  # no log line besides the message
        errors = [
            line for line in r.stderr.splitlines() if line.startswith("ob-analytics:")
        ]
        assert errors == [f"ob-analytics: error: No such file: {missing}"]

    def test_a_package_error_is_one_line(self, monkeypatch, capsys):
        """An ObAnalyticsError from any command is reported, not raised."""
        from ob_analytics import cli
        from ob_analytics.exceptions import ConfigError

        def fail(args):
            raise ConfigError("the run cannot start")

        monkeypatch.setattr(cli, "_cmd_sources", fail)
        with pytest.raises(SystemExit) as exc:
            cli.main(["sources"])
        assert exc.value.code == 1
        assert capsys.readouterr().err == "ob-analytics: error: the run cannot start\n"

    def test_the_error_line_hides_a_known_key(self, monkeypatch, capsys):
        """The message passes through the same redaction as the log."""
        from ob_analytics import _secrets, cli
        from ob_analytics.exceptions import ConfigError

        # Restore the remembered secrets afterwards.
        monkeypatch.setattr(_secrets, "_known", set(_secrets._known))
        monkeypatch.setattr(_secrets, "_ordered", _secrets._ordered)
        _secrets.remember("sk-test-0123456789abcdef")

        def fail(args):
            raise ConfigError("venue said: bad key sk-test-0123456789abcdef")

        monkeypatch.setattr(cli, "_cmd_sources", fail)
        with pytest.raises(SystemExit):
            cli.main(["sources"])
        err = capsys.readouterr().err
        assert "sk-test-0123456789abcdef" not in err
        assert err.startswith("ob-analytics: error: venue said: bad key ")


class TestGalleryOptions:
    """Every command that draws a gallery takes the same gallery options."""

    @staticmethod
    def _options(command):
        import argparse

        from ob_analytics import cli

        parser = cli._build_parser()
        sub = next(
            a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
        )
        return {
            flag
            for action in sub.choices[command]._actions
            for flag in action.option_strings
        }

    def test_the_four_commands_share_one_set(self):
        import argparse

        from ob_analytics import cli

        alone = argparse.ArgumentParser(add_help=False)
        cli._add_gallery_args(alone)
        shared = {flag for a in alone._actions for flag in a.option_strings}
        assert shared == {"--view", "--backend", "--title"}
        for command in ("process", "gallery", "bitstamp-demo", "lobster-demo"):
            assert shared <= self._options(command), command

    @pytest.mark.parametrize(
        ("command", "demo", "extra"),
        [
            ("bitstamp-demo", "run_bitstamp_demo", []),
            ("lobster-demo", "run_lobster_demo", ["--trading-date", "2012-06-21"]),
        ],
    )
    def test_a_demo_passes_them_on(self, monkeypatch, tmp_path, command, demo, extra):
        from ob_analytics import _demos, cli

        seen = {}

        def fake_demo(*args, **kwargs):
            seen.update(kwargs)

        monkeypatch.setattr(_demos, demo, fake_demo)
        cli.main(
            [command, str(tmp_path), *extra, "-o", str(tmp_path), "--view", "l3",
             "--backend", "matplotlib", "--title", "Mine"]
        )  # fmt: skip
        assert seen["view"] == "l3"
        assert seen["backends"] == ["matplotlib"]
        assert seen["title"] == "Mine"

    def test_the_view_defaults_to_both(self, monkeypatch, tmp_path):
        from ob_analytics import _demos, cli

        seen = {}
        monkeypatch.setattr(
            _demos, "run_bitstamp_demo", lambda *a, **kw: seen.update(kw)
        )
        cli.main(["bitstamp-demo", "-o", str(tmp_path)])
        assert seen["view"] == "both"
        assert seen["title"] is None


# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------


class TestProcessSubcommand:
    def test_bitstamp_process(self, cli_runner, tmp_path, tiny_bitstamp_orders_csv):
        out = tmp_path / "out"
        r = cli_runner(
            "process",
            str(tiny_bitstamp_orders_csv),
            "--source",
            "bitstamp",
            "--output",
            str(out),
        )
        assert r.returncode == 0, r.stderr
        assert (out / "events.parquet").exists()
        assert (out / "trades.parquet").exists()
        assert (out / "depth.parquet").exists()
        assert (out / "depth_summary.parquet").exists()

    def test_lobster_process_requires_trading_date(
        self, cli_runner, tmp_path, tiny_bitstamp_orders_csv
    ):
        """Without --trading-date, the LOBSTER path must surface an error."""
        r = cli_runner(
            "process",
            str(tiny_bitstamp_orders_csv),
            "--source",
            "lobster",
            "--output",
            str(tmp_path / "out"),
        )
        assert r.returncode != 0
        combined = (r.stderr + r.stdout).lower()
        assert "trading" in combined  # trading-date / trading_date / "trading date"
        # The message names the source, not a literal format placeholder.
        assert "for the lobster source" in combined

    def test_process_with_gallery_flag(
        self, cli_runner, tmp_path, tiny_bitstamp_orders_csv
    ):
        out = tmp_path / "out"
        r = cli_runner(
            "process",
            str(tiny_bitstamp_orders_csv),
            "--source",
            "bitstamp",
            "--output",
            str(out),
            "--gallery",
        )
        assert r.returncode == 0, r.stderr
        assert (out / "gallery" / "gallery.html").exists()

    def test_the_run_settings_reach_the_config(
        self, cli_runner, tmp_path, tiny_bitstamp_orders_csv
    ):
        from ob_analytics.data import load_result

        out = tmp_path / "out"
        r = cli_runner(
            "process", str(tiny_bitstamp_orders_csv), "-o", str(out),
            "--tick-size", "0.001", "--lot-size", "0.0001",
            "--depth-bins", "4", "--depth-bps", "50",
        )  # fmt: skip
        assert r.returncode == 0, r.stderr
        config = load_result(out).config
        assert config.tick_size == 0.001
        assert config.price_decimals == 3
        assert config.lot_size == 0.0001
        assert config.volume_decimals == 4
        assert config.depth_bins == 4
        assert config.depth_bps == 50

    def test_a_size_that_is_not_more_than_0_is_refused(
        self, cli_runner, tmp_path, tiny_bitstamp_orders_csv
    ):
        r = cli_runner(
            "process", str(tiny_bitstamp_orders_csv), "-o", str(tmp_path / "o"),
            "--tick-size", "0",
        )  # fmt: skip
        assert r.returncode == 2
        assert "'0' is not more than 0" in r.stderr

    def test_session_tz_places_the_session_on_the_clock(
        self, cli_runner, tmp_path, lobster_day_dir
    ):
        from ob_analytics.data import load_result

        first = {}
        for tz in ("America/New_York", "Europe/London"):
            out = tmp_path / tz.replace("/", "_")
            r = cli_runner(
                "process", str(lobster_day_dir), "--source", "lobster",
                "--trading-date", "2024-01-02", "--session-tz", tz, "-o", str(out),
            )  # fmt: skip
            assert r.returncode == 0, r.stderr
            first[tz] = load_result(out).events["timestamp"].min()
        # New York is five hours behind London in January.
        assert first["America/New_York"] - first["Europe/London"] == pd.Timedelta(
            hours=5
        )

    def test_an_unknown_time_zone_is_refused(
        self, cli_runner, tmp_path, lobster_day_dir
    ):
        r = cli_runner(
            "process", str(lobster_day_dir), "--source", "lobster",
            "--trading-date", "2024-01-02", "--session-tz", "Mars/Olympus",
            "-o", str(tmp_path / "o"),
        )  # fmt: skip
        assert r.returncode == 2
        assert "not a time zone name" in r.stderr

    def test_format_writes_another_format(
        self, cli_runner, tmp_path, tiny_bitstamp_orders_csv
    ):
        out = tmp_path / "out"
        r = cli_runner(
            "process", str(tiny_bitstamp_orders_csv), "-o", str(out),
            "--format", "bitstamp",
        )  # fmt: skip
        assert r.returncode == 0, r.stderr
        assert (out / "orders.csv").exists()
        assert (out / "trades.csv").exists()

    def test_a_format_that_writes_a_file(
        self, cli_runner, tmp_path, tiny_bitstamp_orders_csv
    ):
        out = tmp_path / "run.npz"
        r = cli_runner(
            "process", str(tiny_bitstamp_orders_csv), "-o", str(out),
            "--format", "hftbacktest", "--gallery", "--backend", "matplotlib",
        )  # fmt: skip
        assert r.returncode == 0, r.stderr
        assert out.is_file()
        assert (tmp_path / "run_gallery" / "gallery.html").exists()

    def test_a_file_format_without_a_file_name_is_reported(
        self, cli_runner, tmp_path, tiny_bitstamp_orders_csv
    ):
        r = cli_runner(
            "process", str(tiny_bitstamp_orders_csv), "-o", str(tmp_path / "run"),
            "--format", "hftbacktest",
        )  # fmt: skip
        assert r.returncode == 1
        assert "with its extension" in r.stderr
        assert "Traceback" not in r.stderr

    def test_cut_runs_one_window_at_a_time(
        self, cli_runner, tmp_path, tiny_bitstamp_orders_csv
    ):
        from ob_analytics.data import load_result

        whole, windowed = tmp_path / "whole", tmp_path / "windowed"
        r = cli_runner("process", str(tiny_bitstamp_orders_csv), "-o", str(whole))
        assert r.returncode == 0, r.stderr
        r = cli_runner(
            "process", str(tiny_bitstamp_orders_csv), "-o", str(windowed),
            "--cut", "1970-01-01T00:00:10", "--cut", "1970-01-01T00:00:20",
        )  # fmt: skip
        assert r.returncode == 0, r.stderr
        assert "in 3 windows" in r.stderr
        a, b = load_result(whole), load_result(windowed)
        # With the book carried over, the cuts do not show in the output.
        pd.testing.assert_frame_equal(
            a.trades.reset_index(drop=True), b.trades.reset_index(drop=True)
        )
        assert len(a.depth) == len(b.depth)

    @pytest.mark.parametrize(
        ("extra", "message"),
        [
            (("--cut", "1970-01-01T00:00:10", "--format", "bitstamp"), "--cut writes Parquet"),
            (("--cut", "1970-01-01T00:00:10", "--gallery"), "--cut does not build a gallery"),
            (("--no-carry",), "--no-carry applies only with --cut"),
            (("--view", "l3"), "--view applies only with --gallery"),
            (("--backend", "plotly"), "--backend applies only with --gallery"),
            (("--title", "Mine"), "--title applies only with --gallery"),
            (("--cut", "soon"), "is not a time"),
        ],
    )  # fmt: skip
    def test_cut_flags_that_do_not_fit_are_refused(
        self, cli_runner, tmp_path, tiny_bitstamp_orders_csv, extra, message
    ):
        r = cli_runner(
            "process", str(tiny_bitstamp_orders_csv), "-o", str(tmp_path / "o"),
            *extra,
        )  # fmt: skip
        assert r.returncode == 2
        assert message in r.stderr
        assert not (tmp_path / "o").exists()


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


class TestAuditSubcommand:
    def test_audit_reports_summary(self, cli_runner, tiny_bitstamp_orders_csv):
        r = cli_runner("audit", str(tiny_bitstamp_orders_csv))
        assert r.returncode == 0, r.stderr
        assert "Data quality summary" in r.stdout
        assert "feed type" in r.stdout
        # Bitstamp classifies as a diff feed regardless of this micro-book's
        # crossing.
        assert "diff_feed" in r.stdout

    def test_audit_json(self, cli_runner, tiny_bitstamp_orders_csv):
        import json

        r = cli_runner("audit", str(tiny_bitstamp_orders_csv), "--json")
        assert r.returncode == 0, r.stderr
        payload = json.loads(r.stdout)
        assert payload["feed_type"] == "diff_feed"
        assert 0.0 <= payload["crossed_pct"] <= 100.0
        assert payload["ok"] is True
        assert set(payload) >= {
            "crossed_pct",
            "unmatched_trades_pct",
            "duplicate_event_ids",
            "pre_existing_orders",
            "orphan_orders",
            "negative_volume_rows",
            "checks",
        }
        assert {c["name"] for c in payload["checks"]} >= {
            "duplicate_event_ids",
            "sequence_gaps",
            "orphan_orders",
        }

    def test_audit_lobster_requires_trading_date(
        self, cli_runner, tiny_bitstamp_orders_csv
    ):
        r = cli_runner("audit", str(tiny_bitstamp_orders_csv), "--source", "lobster")
        assert r.returncode != 0
        assert "for the lobster source" in (r.stderr + r.stdout).lower()

    def test_audit_listed_in_help(self, cli_runner):
        r = cli_runner("--help")
        assert "audit" in r.stdout

    def test_validate_is_still_accepted(self, cli_runner, tiny_bitstamp_orders_csv):
        """``validate`` is the old name for ``audit``; it must keep working."""
        r = cli_runner("validate", str(tiny_bitstamp_orders_csv))
        assert r.returncode == 0, r.stderr
        assert "Data quality summary" in r.stdout

    def test_audit_fails_on_corrupted_feed(
        self, cli_runner, corrupt_bitstamp_orders_csv
    ):
        """A corrupted feed exits non-zero and names the checks that failed."""
        r = cli_runner("audit", str(corrupt_bitstamp_orders_csv))
        assert r.returncode != 0
        assert "Data quality summary" in r.stdout
        combined = r.stdout + r.stderr
        assert "sequence_gaps" in combined
        assert "exchange_time_after_receive" in combined

    def test_audit_corrupted_feed_json_reports_not_ok(
        self, cli_runner, corrupt_bitstamp_orders_csv
    ):
        import json

        r = cli_runner("audit", str(corrupt_bitstamp_orders_csv), "--json")
        assert r.returncode != 0
        payload = json.loads(r.stdout)
        assert payload["ok"] is False
        assert payload["sequence_gaps"] > 0
        assert payload["orphan_orders"] > 0
        failed = {c["name"] for c in payload["checks"] if not c["passed"]}
        assert "sequence_gaps" in failed

    def test_dropped_created_is_a_warning_until_strict(
        self, cli_runner, dropped_created_orders_csv
    ):
        """An order with no ``created`` row warns; --strict makes it fail."""
        lenient = cli_runner("audit", str(dropped_created_orders_csv))
        assert lenient.returncode == 0, lenient.stderr
        assert "orphan_orders" in lenient.stdout

        strict = cli_runner("audit", str(dropped_created_orders_csv), "--strict")
        assert strict.returncode != 0
        assert "orphan_orders" in (strict.stdout + strict.stderr)

    def test_audit_from_parquet(self, cli_runner, tmp_path, tiny_bitstamp_orders_csv):
        """A saved 'process' output can be audited without re-running it."""
        out = tmp_path / "out"
        r = cli_runner("process", str(tiny_bitstamp_orders_csv), "--output", str(out))
        assert r.returncode == 0, r.stderr

        # The output records the run's declarations, so no --source is needed.
        r = cli_runner("audit", str(out), "--from-parquet")
        assert r.returncode == 0, r.stderr
        assert "Data quality summary" in r.stdout
        assert "diff_feed" in r.stdout

    def test_audit_an_output_that_records_no_declarations(
        self, cli_runner, tmp_path, tiny_bitstamp_orders_csv
    ):
        """An output from an older version is declared by --source, or not at all."""
        import pyarrow.parquet as pq

        from ob_analytics.schemas import DECLARATIONS_KEY

        out = tmp_path / "out"
        r = cli_runner("process", str(tiny_bitstamp_orders_csv), "--output", str(out))
        assert r.returncode == 0, r.stderr
        for file in out.glob("*.parquet"):
            table = pq.read_table(file)
            metadata = dict(table.schema.metadata)
            del metadata[DECLARATIONS_KEY]
            pq.write_table(table.replace_schema_metadata(metadata), file)

        # No --source, so the feed type stays undeclared rather than guessed.
        r = cli_runner("audit", str(out), "--from-parquet")
        assert r.returncode == 0, r.stderr
        assert "unknown" in r.stdout

        r = cli_runner("audit", str(out), "--from-parquet", "--source", "bitstamp")
        assert r.returncode == 0, r.stderr
        assert "diff_feed" in r.stdout

    @staticmethod
    def _capture_by_cryptofeed(tmp_path, tiny_bitstamp_orders_csv):
        """The tiny capture, with the meta.json a cryptofeed capture writes."""
        import json
        import shutil

        cap = tmp_path / "cap"
        cap.mkdir()
        for name in ("orders.csv", "trades.csv"):
            shutil.copy(tiny_bitstamp_orders_csv.parent / name, cap / name)
        (cap / "meta.json").write_text(
            json.dumps(
                {
                    "source": "cryptofeed",
                    "feed_type": "matched_book",
                    "trade_attribution": "maker_only",
                }
            )
        )
        return cap

    def test_process_keeps_the_capture_record(
        self, cli_runner, tmp_path, tiny_bitstamp_orders_csv
    ):
        """meta.json travels into the output, so a later audit can read it (#284)."""
        cap = self._capture_by_cryptofeed(tmp_path, tiny_bitstamp_orders_csv)
        out = tmp_path / "out"
        r = cli_runner("process", str(cap / "orders.csv"), "--output", str(out))
        assert r.returncode == 0, r.stderr
        assert (out / "meta.json").read_text() == (cap / "meta.json").read_text()

    def test_process_removes_a_record_an_earlier_run_left(
        self, cli_runner, tmp_path, tiny_bitstamp_orders_csv
    ):
        """A reused output directory never keeps another capture's record."""
        cap = self._capture_by_cryptofeed(tmp_path, tiny_bitstamp_orders_csv)
        out = tmp_path / "out"
        r = cli_runner("process", str(cap / "orders.csv"), "--output", str(out))
        assert r.returncode == 0, r.stderr
        assert (out / "meta.json").exists()

        r = cli_runner("process", str(tiny_bitstamp_orders_csv), "--output", str(out))
        assert r.returncode == 0, r.stderr
        assert not (out / "meta.json").exists()
        r = cli_runner("audit", str(out), "--from-parquet", "--source", "bitstamp")
        assert r.returncode == 0, r.stderr
        assert "diff_feed" in r.stdout

    def test_audit_checks_against_the_source_that_made_the_capture(
        self, cli_runner, tmp_path, tiny_bitstamp_orders_csv
    ):
        """--source bitstamp reads the files; the record sets the expectations.

        A cryptofeed L3 capture can only be read as bitstamp, whose feed shows
        takers and may cross.  Checking it against bitstamp's declarations
        would fault the capture for what its feed cannot show (#284).
        """
        cap = self._capture_by_cryptofeed(tmp_path, tiny_bitstamp_orders_csv)
        out = tmp_path / "out"
        r = cli_runner("process", str(cap / "orders.csv"), "--output", str(out))
        assert r.returncode == 0, r.stderr

        # The run records the capture's declarations in its output, and a
        # run on the capture itself reads them from the record.
        for target, extra in ((out, ["--from-parquet"]), (cap / "orders.csv", [])):
            r = cli_runner("audit", str(target), *extra, "--source", "bitstamp")
            assert r.returncode == 0, r.stderr
            assert "matched_book" in r.stdout
            assert "maker only" in r.stdout
        assert "declares" in r.stderr

    @pytest.mark.parametrize(
        "error", [KeyError("not installed"), TypeError("needs settings")]
    )
    def test_a_recording_source_that_cannot_be_built_keeps_the_default(
        self, monkeypatch, error
    ):
        """Reading what the capture's source declares never stops ``audit``."""
        from ob_analytics import capture_record
        from ob_analytics.protocols import SequenceKind

        def get_source(name):
            raise error

        monkeypatch.setattr("ob_analytics.sources.get_source", get_source)
        with warnings_logged() as logged:
            kind = capture_record._declared_sequence_kind(
                "plugin", default=SequenceKind.CONTIGUOUS
            )
        assert kind is SequenceKind.CONTIGUOUS
        # The fallback changes how audit reads the sequence, so it is said.
        assert any("contiguous instead" in m for m in logged)

    def test_audit_reports_what_a_single_capture_lost(
        self, cli_runner, tmp_path, tiny_bitstamp_orders_csv
    ):
        """dropped and capture_error in meta.json fail the capture checks."""
        cap = self._capture_by_cryptofeed(tmp_path, tiny_bitstamp_orders_csv)
        meta = json.loads((cap / "meta.json").read_text())
        meta.update(
            dropped=2,
            capture_error="ConnectionError('closed')",
            capture_error_phase="stream",
        )
        (cap / "meta.json").write_text(json.dumps(meta))

        r = cli_runner("audit", str(cap / "orders.csv"), "--source", "bitstamp")
        assert r.returncode == 0, r.stderr
        assert "dropped_messages: 2 message(s)" in r.stdout
        assert "capture_errors" in r.stdout
        assert "ConnectionError('closed')" in r.stdout

        r = cli_runner(
            "audit", str(cap / "orders.csv"), "--source", "bitstamp", "--strict"
        )
        assert r.returncode == 1
        assert "dropped_messages" in r.stderr

        r = cli_runner(
            "audit", str(cap / "orders.csv"), "--source", "bitstamp", "--json"
        )
        report = json.loads(r.stdout)
        failed = {c["name"] for c in report["capture_checks"] if not c["passed"]}
        assert failed == {"dropped_messages", "capture_errors"}

    def test_a_clean_single_capture_passes_strict(
        self, cli_runner, tmp_path, tiny_bitstamp_orders_csv
    ):
        cap = self._capture_by_cryptofeed(tmp_path, tiny_bitstamp_orders_csv)
        r = cli_runner(
            "audit", str(cap / "orders.csv"), "--source", "bitstamp", "--json"
        )
        report = json.loads(r.stdout)
        assert all(c["passed"] for c in report["capture_checks"])


class TestAuditFewLevels:
    """A LOBSTER message file holds only the events inside the requested
    levels, so the book rebuilt from it can cross while the orderbook file
    does not.  ``audit`` reports that crossing as well as the depth table's."""

    def _audit(self, cli_runner, directory, *extra):
        return cli_runner(
            "audit",
            str(directory),
            "--source",
            "lobster",
            "--trading-date",
            few.DATE,
            *extra,
        )

    def test_a_crossed_per_order_book_is_reported(self, cli_runner, tmp_path):
        files = few.write_files(tmp_path, few.CROSSING_MESSAGES, few.CROSSING_ORDERBOOK)

        r = self._audit(cli_runner, files, "--json")

        assert r.returncode == 0, r.stderr
        payload = json.loads(r.stdout)
        # The depth table, read from the orderbook file, never crosses.
        assert payload["crossed_pct"] == 0.0
        # Crossed from 34204 to 34210 of a file spanning 34200 to 34210.
        assert payload["per_order_crossed_pct"] == pytest.approx(60.0)
        assert payload["per_order_crossed_episodes"] == 1
        check = next(
            c for c in payload["checks"] if c["name"] == "per_order_book_crossed"
        )
        assert check["passed"] is False
        assert check["severity"] == "warning"

    def test_the_report_says_what_the_crossing_means(self, cli_runner, tmp_path):
        files = few.write_files(tmp_path, few.CROSSING_MESSAGES, few.CROSSING_ORDERBOOK)

        r = self._audit(cli_runner, files)

        assert r.returncode == 0, r.stderr
        line = next(
            ln for ln in r.stdout.splitlines() if ln.lstrip().startswith("per-order")
        )
        assert "60.00%" in line
        assert "request more levels" in line

    def test_strict_fails_on_it(self, cli_runner, tmp_path):
        files = few.write_files(tmp_path, few.CROSSING_MESSAGES, few.CROSSING_ORDERBOOK)

        r = self._audit(cli_runner, files, "--strict")

        assert r.returncode != 0
        assert "per_order_book_crossed" in r.stdout + r.stderr

    def test_a_book_that_never_crosses_passes(self, cli_runner, tmp_path):
        files = few.write_files(tmp_path, few.CLEAN_MESSAGES, few.CLEAN_ORDERBOOK)

        r = self._audit(cli_runner, files, "--json", "--strict")

        assert r.returncode == 0, r.stdout + r.stderr
        payload = json.loads(r.stdout)
        assert payload["per_order_crossed_pct"] == 0.0
        check = next(
            c for c in payload["checks"] if c["name"] == "per_order_book_crossed"
        )
        assert check["passed"] is True


class TestAuditVenueBook:
    """``audit`` compares the book rebuilt from the LOBSTER messages with the
    orderbook file, row by row."""

    @staticmethod
    def _audit(cli_runner, directory, *extra):
        return cli_runner(
            "audit",
            str(directory),
            "--source",
            "lobster",
            "--trading-date",
            few.DATE,
            *extra,
        )

    @staticmethod
    def _check(payload):
        return next(c for c in payload["checks"] if c["name"] == "venue_book")

    def test_the_lobster_sample_matches_its_orderbook_file(
        self, cli_runner, lobster_day_dir
    ):
        r = self._audit(cli_runner, lobster_day_dir)

        assert r.returncode == 0, r.stderr
        assert "rebuilt book matches the venue's book in all 5 row(s)" in r.stdout

        payload = json.loads(self._audit(cli_runner, lobster_day_dir, "--json").stdout)
        assert payload["venue_book_rows"] == 5
        assert payload["venue_book_differing_rows"] == 0
        assert self._check(payload)["passed"] is True

    def test_one_changed_level_warns_and_fails_strict(
        self, cli_runner, lobster_day_dir
    ):
        (path,) = lobster_day_dir.glob("*_orderbook_*.csv")
        lines = path.read_text().splitlines()
        lines[2] = "1010000,160,1000000,90"  # the bid after message 3
        path.write_text("\n".join(lines) + "\n")

        r = self._audit(cli_runner, lobster_day_dir, "--json")
        assert r.returncode == 0, r.stderr
        payload = json.loads(r.stdout)
        check = self._check(payload)
        assert check["passed"] is False
        assert check["severity"] == "warning"
        assert payload["venue_book_first_difference"]["message_row"] == 3

        strict = self._audit(cli_runner, lobster_day_dir, "--strict")
        assert strict.returncode != 0
        assert "venue_book" in strict.stdout + strict.stderr

    def test_a_source_without_a_venue_book_skips_it(
        self, cli_runner, tiny_bitstamp_orders_csv
    ):
        r = cli_runner("audit", str(tiny_bitstamp_orders_csv), "--json")

        assert r.returncode == 0, r.stderr
        check = self._check(json.loads(r.stdout))
        assert check["skipped"] is True
        assert check["passed"] is False
        assert check["severity"] == "info"

    def test_a_saved_run_skips_it(self, cli_runner, lobster_day_dir, tmp_path):
        out = tmp_path / "saved"
        processed = cli_runner(
            "process",
            str(lobster_day_dir),
            "--source",
            "lobster",
            "--trading-date",
            few.DATE,
            "--output",
            str(out),
        )
        assert processed.returncode == 0, processed.stderr

        r = cli_runner(
            "audit", str(out), "--from-parquet", "--source", "lobster", "--json"
        )

        assert r.returncode == 0, r.stderr
        assert self._check(json.loads(r.stdout))["skipped"] is True


# ---------------------------------------------------------------------------
# gallery
# ---------------------------------------------------------------------------


class TestAuditRunFlags:
    def test_run_settings_are_refused_on_a_saved_result(
        self, cli_runner, tmp_path, tiny_bitstamp_orders_csv
    ):
        r = cli_runner(
            "audit", str(tmp_path), "--from-parquet", "--tick-size", "0.001",
            "--session-tz", "UTC",
        )  # fmt: skip
        assert r.returncode == 2
        assert "--session-tz, --tick-size set how the pipeline runs" in r.stderr

    def test_run_settings_reach_the_audited_run(
        self, cli_runner, tiny_bitstamp_orders_csv
    ):
        r = cli_runner(
            "audit", str(tiny_bitstamp_orders_csv), "--tick-size", "0.001", "--json"
        )
        assert r.returncode == 0, r.stderr


class TestGallerySubcommand:
    def test_backend_picks_the_gallery_columns(
        self, monkeypatch, tmp_path, tiny_bitstamp_orders_csv
    ):
        from ob_analytics import cli
        from ob_analytics.visualization import gallery

        out = tmp_path / "parq"
        cli.main(["process", str(tiny_bitstamp_orders_csv), "-o", str(out)])
        shown = {}

        def generate_gallery(result, output, **kwargs):
            shown.update(kwargs)
            return output / "gallery.html"

        monkeypatch.setattr(gallery, "generate_gallery", generate_gallery)
        cli.main(
            ["gallery", str(out), "-o", str(tmp_path / "g"),
             "--backend", "matplotlib", "--backend", "bokeh"]
        )  # fmt: skip
        assert shown["backends"] == ["matplotlib", "bokeh"]

    def test_gallery_from_process_output(
        self, cli_runner, tmp_path, tiny_bitstamp_orders_csv
    ):
        # First run process to get parquet
        parq = tmp_path / "parq"
        r1 = cli_runner(
            "process",
            str(tiny_bitstamp_orders_csv),
            "--source",
            "bitstamp",
            "--output",
            str(parq),
        )
        assert r1.returncode == 0, r1.stderr

        gallery = tmp_path / "gallery"
        r2 = cli_runner("gallery", str(parq), "--output", str(gallery))
        assert r2.returncode == 0, r2.stderr
        assert (gallery / "gallery.html").exists()

    def test_a_lobster_run_reloads_with_its_own_units(
        self, cli_runner, tmp_path, monkeypatch, lobster_day_dir, lobster_day_pipeline
    ):
        """Whole shares stay whole shares when ``gallery`` reads a saved run."""
        import sys

        from ob_analytics import cli
        from ob_analytics.visualization import gallery
        from ob_analytics.visualization.gallery import display_result

        out = tmp_path / "out"
        r = cli_runner(
            "process",
            str(lobster_day_dir),
            "--source",
            "lobster",
            "--trading-date",
            "2024-01-02",
            "--output",
            str(out),
        )
        assert r.returncode == 0, r.stderr

        shown = {}

        def generate_gallery(result, output, **kwargs):
            shown["result"] = result
            return output / "gallery.html"

        monkeypatch.setattr(gallery, "generate_gallery", generate_gallery)
        monkeypatch.setattr(cli, "_setup_logging", lambda verbose: None)
        monkeypatch.setattr(
            sys,
            "argv",
            ["ob-analytics", "gallery", str(out), "--output", str(tmp_path)],
        )
        cli.main()

        before = display_result(lobster_day_pipeline.run(lobster_day_dir))
        after = display_result(shown["result"])
        assert after.trades["volume"].tolist() == [40.0, 25.0]
        assert after.trades["price"].tolist() == [101.0, 100.0]
        for name in ("trades", "depth"):
            for column in ("price", "volume"):
                assert (
                    getattr(after, name)[column].tolist()
                    == getattr(before, name)[column].tolist()
                ), (name, column)

    def test_gallery_missing_data(self, cli_runner, tmp_path):
        r = cli_runner("gallery", str(tmp_path / "nonexistent"))
        assert r.returncode != 0
        assert "not a Parquet folder" in r.stderr
        assert "Traceback" not in r.stderr


# ---------------------------------------------------------------------------
# bitstamp-demo
# ---------------------------------------------------------------------------


class TestBitstampDemoSubcommand:
    def test_explicit_input(self, cli_runner, tmp_path, tiny_bitstamp_orders_csv):
        """bitstamp-demo <dir-with-orders-and-trades> succeeds.

        Uses the tiny programmatic fixture instead of the bundled sample
        (bundled-sample bitstamp-demo takes minutes through full pipeline +
        gallery; that path is exercised manually).
        """
        out = tmp_path / "bs_out"
        r = cli_runner(
            "bitstamp-demo",
            str(tiny_bitstamp_orders_csv.parent),
            "--output",
            str(out),
        )
        assert r.returncode == 0, r.stderr
        assert (out / "parquet" / "events.parquet").exists()
        assert (out / "gallery" / "gallery.html").exists()

    def test_view_comparison(self, cli_runner, tmp_path, tiny_bitstamp_orders_csv):
        """``--view comparison`` threads through to the gallery (L2-vs-L3)."""
        out = tmp_path / "bs_cmp"
        r = cli_runner(
            "bitstamp-demo",
            str(tiny_bitstamp_orders_csv.parent),
            "--output",
            str(out),
            "--view",
            "comparison",
        )
        assert r.returncode == 0, r.stderr
        assert (out / "gallery" / "gallery.html").exists()

    def test_invalid_view_rejected(
        self, cli_runner, tmp_path, tiny_bitstamp_orders_csv
    ):
        """An unknown ``--view`` is rejected by argparse before any work."""
        r = cli_runner(
            "bitstamp-demo",
            str(tiny_bitstamp_orders_csv.parent),
            "--output",
            str(tmp_path / "bs_bad"),
            "--view",
            "nope",
        )
        assert r.returncode != 0


# ---------------------------------------------------------------------------
# lobster-demo
# ---------------------------------------------------------------------------


class TestLobsterDemoSubcommand:
    def test_missing_source_errors(self, cli_runner, tmp_path):
        r = cli_runner(
            "lobster-demo",
            str(tmp_path / "does_not_exist"),
            "--trading-date",
            "2012-06-21",
            "--output",
            str(tmp_path / "lob_out"),
        )
        assert r.returncode != 0

    def test_missing_trading_date_errors(self, cli_runner, tmp_path):
        r = cli_runner(
            "lobster-demo",
            str(tmp_path),
            "--output",
            str(tmp_path / "lob_out"),
        )
        assert r.returncode != 0
        combined = (r.stderr + r.stdout).lower()
        assert "trading-date" in combined or "required" in combined


# ---------------------------------------------------------------------------
# sources (WS-5.3)
# ---------------------------------------------------------------------------


class TestSourcesSubcommand:
    def test_lists_registered_sources(self, cli_runner):
        r = cli_runner("sources")
        assert r.returncode == 0, r.stderr
        assert "bitstamp" in r.stdout
        assert "lobster" in r.stdout

    def test_shows_capability_and_required_context(self, cli_runner):
        # LOBSTER advertises its trading_date requirement; bitstamp has none.
        r = cli_runner("sources")
        lobster_line = next(ln for ln in r.stdout.splitlines() if "lobster" in ln)
        assert "trading_date" in lobster_line
        assert "offline" in lobster_line
        # Bitstamp is both offline and live, with no required context.
        bitstamp_line = next(ln for ln in r.stdout.splitlines() if "bitstamp" in ln)
        assert "requires" not in bitstamp_line
        assert "offline" in bitstamp_line and "live" in bitstamp_line

    def test_process_source_choices_are_dynamic(self, cli_runner):
        # --source choices come from list_sources(), shown in --help.
        r = cli_runner("process", "--help")
        assert r.returncode == 0
        assert "bitstamp" in r.stdout and "lobster" in r.stdout


class TestRequiredContext:
    def test_source_required_context(self) -> None:
        from ob_analytics.bitstamp import BitstampSource
        from ob_analytics.lobster import LobsterSource
        from ob_analytics.protocols import required_context_of

        assert required_context_of(BitstampSource()) == []
        assert required_context_of(LobsterSource()) == ["trading_date"]


# ---------------------------------------------------------------------------
# capture (CCXT wiring, #106)
# ---------------------------------------------------------------------------


def _fake_capture(monkeypatch, *, error=None):
    """Replace the capture loop; return the dict it fills with what it was given."""
    import pandas as pd

    from ob_analytics.live import CaptureResult

    captured: dict = {}

    async def _fake_run(source, config, sink=None, **_kwargs):
        captured["source"] = source
        captured["config"] = config
        now = pd.Timestamp.now(tz="UTC")
        return CaptureResult(
            out_dir=config.out_dir,
            n_order_events=0,
            n_trade_events=0,
            n_raw_frames=0,
            started=now,
            ended=now,
            capture_error=error,
            capture_error_phase="stream" if error else None,
        )

    monkeypatch.setattr("ob_analytics.live._supervisor.run_capturer", _fake_run)
    return captured


class TestCaptureSubcommand:
    def test_capture_help_points_at_the_source_help(self, cli_runner):
        r = cli_runner("capture", "--help")
        assert r.returncode == 0
        assert "for that source's flags" in r.stdout

    def test_ccxt_flags_in_its_help(self, cli_runner):
        r = cli_runner("capture", "ccxt", "--help")
        assert r.returncode == 0
        for flag in ("--venue", "--depth-limit", "--poll-interval", "--output"):
            assert flag in r.stdout
        assert "--exchange" not in r.stdout
        assert "depth.csv" in r.stdout  # L2 output documented

    def test_old_flag_names_are_refused(self, cli_runner, tmp_path):
        r = cli_runner(
            "capture", "ccxt", "--pair", "BTC/USD", "--exchange", "kraken",
            "--out", str(tmp_path / "c"),
        )  # fmt: skip
        assert r.returncode == 2
        assert "unrecognized arguments" in r.stderr

    def test_output_is_required(self, cli_runner):
        r = cli_runner("capture", "bitstamp")
        assert r.returncode == 2
        assert "-o/--output" in r.stderr

    def test_pair_is_required(self, cli_runner, tmp_path):
        """No source shares a pair symbol, so --pair has no default."""
        r = cli_runner("capture", "bitstamp", "-o", str(tmp_path / "cap"))
        assert r.returncode == 2
        assert "--pair" in r.stderr
        assert not (tmp_path / "cap").exists()

    def test_a_source_is_required(self, cli_runner):
        r = cli_runner("capture")
        assert r.returncode == 2
        assert "source is required" in r.stderr

    def test_offline_only_source_is_named(self, cli_runner, tmp_path):
        r = cli_runner(
            "capture", "lobster", "--pair", "x", "--output", str(tmp_path / "cap")
        )  # fmt: skip
        assert r.returncode == 1
        assert "Source 'lobster' has no live capture" in r.stderr

    @pytest.mark.skipif(not _CCXT_INSTALLED, reason="ccxt extra not installed")
    def test_list_includes_ccxt(self, cli_runner):
        r = cli_runner("capture", "--list")
        assert r.returncode == 0, r.stderr
        assert "bitstamp" in r.stdout
        assert "ccxt" in r.stdout

    def test_venue_flag_flows_into_typed_settings(self, monkeypatch, tmp_path):
        """--venue / --depth-limit flow into typed CcxtSettings (no network)."""
        from ob_analytics import cli
        from ob_analytics.live import CaptureConfig, CcxtSettings, CcxtSource

        captured = _fake_capture(monkeypatch)
        # This test checks the flag wiring, not the extra: skip the check for it.
        monkeypatch.setattr(
            "ob_analytics.live.ccxt_source.CcxtSource.preflight", lambda self: None
        )
        cli.main(
            [
                "capture", "ccxt", "--pair", "BTC/USDT", "--venue", "binance",
                "--depth-limit", "50", "--market-data-mirror", "--minutes", "0.001",
                "-o", str(tmp_path / "o"), "--no-raw",
            ]
        )  # fmt: skip

        # The venue knobs land as typed settings on the source, not on the config.
        source = captured["source"]
        assert isinstance(source, CcxtSource)
        assert isinstance(source.settings, CcxtSettings)
        assert source.settings.exchange == "binance"
        assert source.settings.depth_limit == 50
        assert source.settings.market_data_mirror is True
        assert source.settings.poll_interval == 1.0  # not passed -> default

        cfg = captured["config"]
        assert isinstance(cfg, CaptureConfig)
        assert cfg.pair == "BTC/USDT"
        assert cfg.out_dir.parent == tmp_path / "o"  # one folder per segment
        assert cfg.keep_raw is False

    def test_missing_extra_exits_nonzero_before_output(self, monkeypatch, tmp_path):
        """Without the cryptofeed extra the capture stops before it starts (#283)."""
        import sys

        from ob_analytics import cli

        # A None entry in sys.modules makes the import raise ImportError.
        for mod in ("cryptofeed", "cryptofeed.exchanges"):
            monkeypatch.setitem(sys.modules, mod, None)

        with pytest.raises(SystemExit) as exc:
            cli.main(
                ["capture", "cryptofeed", "--venue", "bitstamp", "--pair", "BTC-USD",
                 "-o", str(tmp_path / "o")]
            )  # fmt: skip
        assert exc.value.code == 1
        assert not (tmp_path / "o").exists()

    @pytest.mark.skipif(not _CCXT_INSTALLED, reason="ccxt extra not installed")
    def test_unknown_ccxt_venue_exits_nonzero_before_output(self, tmp_path):
        """A misspelt --venue stops the capture before it starts (#283)."""
        from ob_analytics import cli

        with pytest.raises(SystemExit) as exc:
            cli.main(
                ["capture", "ccxt", "--venue", "binanse", "--pair", "BTC/USDT",
                 "-o", str(tmp_path / "o")]
            )  # fmt: skip
        assert exc.value.code == 1
        assert not (tmp_path / "o").exists()

    def test_capture_error_exits_nonzero(self, monkeypatch, tmp_path):
        """A capture that fails part way through fails the command (#283)."""
        from ob_analytics import cli

        _fake_capture(monkeypatch, error="ValueError('boom')")
        monkeypatch.setattr(
            "ob_analytics.bitstamp.BitstampSource.preflight", lambda self: None
        )
        with pytest.raises(SystemExit) as exc:
            cli.main(
                ["capture", "bitstamp", "--pair", "btcusd", "--minutes", "0.001",
                 "-o", str(tmp_path / "o")]
            )  # fmt: skip
        assert exc.value.code == 1


class TestCaptureFlagsFromSettings:
    """``capture <source>`` builds its flags from the source's settings class."""

    @staticmethod
    def _register(monkeypatch, name, source_cls):
        from ob_analytics.sources import SOURCES

        monkeypatch.setitem(SOURCES._items, name, source_cls)

    def test_a_plug_in_source_is_set_up_from_the_command_line(
        self, monkeypatch, tmp_path, capsys
    ):
        from ob_analytics import cli
        from ob_analytics.protocols import Level
        from tests._fake_plugin_source import (
            TOKEN_ENV,
            FakePluginSettings,
            FakePluginSource,
        )

        self._register(monkeypatch, "fakeplugin", FakePluginSource)
        monkeypatch.setenv(TOKEN_ENV, "fake-plugin-token")
        captured = _fake_capture(monkeypatch)

        with pytest.raises(SystemExit):
            cli.main(["capture", "fakeplugin", "--help"])
        shown = capsys.readouterr().out
        for flag in (
            "--market MARKET",
            "--levels LEVELS",
            "--speed SPEED",
            "--book-side {bid,ask}",
            "--resolution {L2,L3}",
            "--fast, --no-fast",
            "--where WHERE",
        ):
            assert flag in shown, flag
        assert "The market to follow" in shown
        assert "(default: 10)" in shown
        # No key on the command line, and no flag for a type it cannot read.
        assert "--token" not in shown
        assert "--handle" not in shown

        cli.main(
            [
                "capture", "fakeplugin", "--market", "M-1", "--levels", "3",
                "--speed", "0.5", "--book-side", "ASK", "--resolution", "l2",
                "--fast", "--where", "far/away", "--pair", "x",
                "--minutes", "0.001", "-o", str(tmp_path / "o"),
            ]
        )  # fmt: skip
        settings = captured["source"].settings
        assert isinstance(settings, FakePluginSettings)
        assert settings.market == "M-1"
        assert settings.depth == 3
        assert settings.speed == 0.5
        assert settings.book_side == "ask"
        assert settings.resolution is Level.L2
        assert settings.fast is True
        assert settings.where == Path("far/away")
        # The key came from the environment, not from a flag.
        assert settings.token.get_secret_value() == "fake-plugin-token"

    def test_a_flag_left_out_keeps_the_default(self, monkeypatch, tmp_path):
        from ob_analytics import cli
        from tests._fake_plugin_source import TOKEN_ENV, FakePluginSource

        self._register(monkeypatch, "fakeplugin", FakePluginSource)
        monkeypatch.setenv(TOKEN_ENV, "fake-plugin-token")
        captured = _fake_capture(monkeypatch)
        cli.main(
            ["capture", "fakeplugin", "--market", "M-1", "--pair", "x",
             "--minutes", "0.001", "-o", str(tmp_path / "o")]
        )  # fmt: skip
        settings = captured["source"].settings
        assert settings.depth == 10
        assert settings.book_side == "bid"
        assert settings.fast is False

    def test_a_source_that_requires_its_settings_runs_with_no_flag(
        self, monkeypatch, tmp_path
    ):
        from ob_analytics import cli
        from tests._fake_plugin_source import OptionalSettings, SettingsRequiredSource

        self._register(monkeypatch, "settingsrequired", SettingsRequiredSource)
        captured = _fake_capture(monkeypatch)
        cli.main(
            ["capture", "settingsrequired", "--pair", "x", "--minutes", "0.001",
             "-o", str(tmp_path / "o")]
        )  # fmt: skip
        assert captured["source"].settings == OptionalSettings()

    def test_a_required_field_is_a_required_flag(self, monkeypatch, tmp_path, capsys):
        from ob_analytics import cli
        from tests._fake_plugin_source import FakePluginSource

        self._register(monkeypatch, "fakeplugin", FakePluginSource)
        with pytest.raises(SystemExit) as exc:
            cli.main(["capture", "fakeplugin", "-o", str(tmp_path / "o")])
        assert exc.value.code == 2
        assert "--market" in capsys.readouterr().err

    def test_a_value_the_settings_refuse_stops_before_output(
        self, monkeypatch, tmp_path, capfd
    ):
        from ob_analytics import cli
        from tests._fake_plugin_source import FakePluginSource

        self._register(monkeypatch, "fakeplugin", FakePluginSource)
        with pytest.raises(SystemExit) as exc:
            cli.main(
                ["capture", "fakeplugin", "--market", "M-1", "--levels", "0",
                 "--pair", "x", "-o", str(tmp_path / "o")]
            )  # fmt: skip
        assert exc.value.code == 2
        assert "Bad settings for source 'fakeplugin'" in capfd.readouterr().err
        assert not (tmp_path / "o").exists()

    def test_a_bad_choice_is_refused(self, monkeypatch, tmp_path, capsys):
        from ob_analytics import cli
        from tests._fake_plugin_source import FakePluginSource

        self._register(monkeypatch, "fakeplugin", FakePluginSource)
        with pytest.raises(SystemExit) as exc:
            cli.main(
                ["capture", "fakeplugin", "--market", "M-1", "--book-side", "mid",
                 "--pair", "x", "-o", str(tmp_path / "o")]
            )  # fmt: skip
        assert exc.value.code == 2
        assert "'mid' is not one of bid, ask" in capsys.readouterr().err

    def test_a_field_that_clashes_with_a_capture_flag_is_named(
        self, monkeypatch, tmp_path, capfd
    ):
        from ob_analytics import cli
        from tests._fake_plugin_source import ClashingSource

        self._register(monkeypatch, "clashing", ClashingSource)
        with pytest.raises(SystemExit) as exc:
            cli.main(["capture", "clashing", "-o", str(tmp_path / "o")])
        assert exc.value.code == 1
        err = capfd.readouterr().err
        assert "ClashingSettings.minutes would be the flag --minutes" in err


# ---------------------------------------------------------------------------
# process / audit: the reading source and its settings (#502)
# ---------------------------------------------------------------------------

#: The opening book and two changes of a ccxt capture, as the runner writes
#: them: ``(side, price, volume, origin)``.
_CCXT_BOOK = [
    ("bid", 100.0, 1.5, "snapshot"),
    ("ask", 101.0, 2.0, "snapshot"),
    ("bid", 100.5, 0.5, "stream"),
    ("ask", 101.0, 1.0, "stream"),
]


def _ccxt_capture(folder: Path, *, declarations: dict | None = None) -> Path:
    """Write a small ccxt capture into *folder*, as the capture runner does.

    The rows go through the capture's own file writer, and ``meta.json``
    gets what a ccxt source declares.  *declarations* replaces those, to
    stand for a capture written by an older version.
    """
    from ob_analytics.capture_files import FileCaptureSink
    from ob_analytics.capture_record import source_declarations
    from ob_analytics.live import CaptureResult
    from ob_analytics.live.ccxt_source import CcxtSettings, CcxtSource
    from ob_analytics.protocols import Level

    source = CcxtSource(settings=CcxtSettings(exchange="kraken"))
    sink = FileCaptureSink(folder, keep_raw=False, level=Level.L2)
    t0 = pd.Timestamp("2026-11-17 12:00", tz="UTC")
    for i, (side, price, volume, origin) in enumerate(_CCXT_BOOK):
        ts = t0 + pd.Timedelta(milliseconds=i)
        sink.write_depth(
            {"timestamp": ts, "exchange_timestamp": ts, "direction": side,
             "price": price, "volume": volume, "sequence": i + 1,
             "origin": origin}
        )  # fmt: skip
    ts = t0 + pd.Timedelta(milliseconds=len(_CCXT_BOOK))
    sink.write_trade(
        {"trade_id": "t1", "timestamp": ts, "exchange_timestamp": ts,
         "price": 101.0, "volume": 1.0, "buy_order_id": "", "sell_order_id": "",
         "direction": "buy"}
    )  # fmt: skip
    extras = source_declarations(source) if declarations is None else declarations
    sink.finalize(
        CaptureResult(
            out_dir=folder,
            n_order_events=0,
            n_trade_events=1,
            n_raw_frames=0,
            started=t0,
            ended=ts,
            extras={**extras, "tick_size": 0.5, "lot_size": 0.5},
            n_depth_events=len(_CCXT_BOOK),
        )
    )
    return folder


def _segmented_ccxt_capture(root: Path, *, old: bool = False) -> Path:
    """Write a ccxt capture of one closed segment, with its ``manifest.json``.

    An *old* capture records no reading source, as one written before
    captures recorded it.
    """
    from ob_analytics.live._manifest import CaptureManifest, EndReason, Segment
    from ob_analytics.live._supervisor import _declarations
    from ob_analytics.live.ccxt_source import CcxtSettings, CcxtSource

    declarations = _declarations(CcxtSource(settings=CcxtSettings(exchange="kraken")))
    if old:
        declarations.pop("replay_source")
    seg_dir = _ccxt_capture(
        root / "seg-0001",
        declarations={"source": "ccxt", **declarations} if old else None,
    )
    started = pd.Timestamp("2026-11-17 12:00", tz="UTC")
    CaptureManifest(
        source="ccxt",
        pair="BTC/USD",
        level="L2",
        started=started,
        declarations=declarations,
        segments=[
            Segment(
                name=seg_dir.name,
                started=started,
                ended=started + pd.Timedelta(seconds=1),
                end_reason=EndReason.STOPPED,
                n_book_events=len(_CCXT_BOOK),
                n_trade_events=1,
            )
        ],
    ).write(root)
    return root


class TestReadingSourceFromTheRecord:
    """A capture names the source that reads it, so ``--source`` is not needed."""

    def test_the_capture_record_names_the_reading_source(self, tmp_path):
        from ob_analytics.capture_record import read_record
        from ob_analytics.protocols import Level

        record = read_record(_ccxt_capture(tmp_path / "cap"))
        assert record.level is Level.L2
        assert record.replay_source == "depth_csv"

    def test_process_a_ccxt_capture_without_source(self, cli_runner, tmp_path):
        cap = _ccxt_capture(tmp_path / "cap")
        out = tmp_path / "out"
        r = cli_runner("process", str(cap), "-o", str(out))
        assert r.returncode == 0, r.stderr
        assert "source=depth_csv" in r.stderr
        assert (out / "depth.parquet").is_file()

    def test_process_a_segmented_ccxt_capture_without_source(
        self, cli_runner, tmp_path
    ):
        cap = _segmented_ccxt_capture(tmp_path / "cap")
        out = tmp_path / "out"
        r = cli_runner("process", str(cap), "-o", str(out))
        assert r.returncode == 0, r.stderr
        assert (out / "seg-0001" / "depth.parquet").is_file()

    def test_audit_a_ccxt_capture_without_source(self, cli_runner, tmp_path):
        r = cli_runner("audit", str(_ccxt_capture(tmp_path / "cap")))
        assert r.returncode == 0, r.stderr
        assert "source=depth_csv" in r.stderr

    @pytest.mark.parametrize("command", ["process", "audit"])
    def test_a_source_of_another_level_is_named_with_the_record(
        self, cli_runner, tmp_path, command
    ):
        cap = _ccxt_capture(tmp_path / "cap")
        r = cli_runner(command, str(cap), "--source", "bitstamp")
        assert r.returncode == 2
        assert "--source bitstamp reads L3 data" in r.stderr
        assert "is L2" in r.stderr
        assert "depth_csv" in r.stderr

    def test_an_old_segmented_capture_is_read_by_its_level(self, cli_runner, tmp_path):
        """A manifest with no reading source still records the level."""
        cap = _segmented_ccxt_capture(tmp_path / "cap", old=True)
        r = cli_runner("process", str(cap), "-o", str(tmp_path / "out"))
        assert r.returncode == 0, r.stderr
        assert "source=depth_csv" in r.stderr

    def test_an_old_capture_that_records_nothing_keeps_the_default(
        self, cli_runner, tmp_path
    ):
        """With no reading source and no level, bitstamp reads it, as before."""
        cap = _ccxt_capture(tmp_path / "cap", declarations={"source": "ccxt"})
        r = cli_runner("process", str(cap), "-o", str(tmp_path / "out"))
        assert r.returncode != 0
        assert "source=bitstamp" in r.stderr
        r = cli_runner(
            "process", str(cap), "--source", "depth_csv", "-o", str(tmp_path / "o2")
        )
        assert r.returncode == 0, r.stderr


class TestReadingSourceSettings:
    """``process`` and ``audit`` take a flag per field of the reading source's settings."""

    @pytest.mark.parametrize("command", ["process", "audit"])
    def test_the_settings_flags_are_in_the_help(self, cli_runner, command):
        r = cli_runner(command, "x.dbn.zst", "--source", "databento", "--help")
        assert r.returncode == 0, r.stderr
        for flag in (
            "--instrument-id INSTRUMENT_ID",
            "--publisher-id PUBLISHER_ID",
            "--raw-symbol RAW_SYMBOL",
            "--trades-from {fills,prints}",
            "--dataset DATASET",
            "--symbol SYMBOL",
            "--venue VENUE",
        ):
            assert flag in r.stdout

    def test_a_capture_shows_the_settings_of_its_reading_source(
        self, cli_runner, tmp_path
    ):
        r = cli_runner("process", str(_ccxt_capture(tmp_path / "cap")), "--help")
        assert r.returncode == 0, r.stderr
        assert "--price-divisor PRICE_DIVISOR" in r.stdout

    def test_a_flag_of_another_source_is_refused(self, cli_runner, tmp_path):
        cap = _ccxt_capture(tmp_path / "cap")
        r = cli_runner("process", str(cap), "--instrument-id", "5")
        assert r.returncode == 2
        assert "unrecognized arguments: --instrument-id 5" in r.stderr

    def test_settings_and_identity_reach_the_run(self, monkeypatch, tmp_path):
        from ob_analytics import cli
        from ob_analytics.pipeline import Pipeline

        cap = _ccxt_capture(tmp_path / "cap")
        seen: dict = {}

        def run(self, path, *args, **kwargs):
            seen["settings"] = self._source.settings
            seen["ctx"] = self._ctx
            raise SystemExit(0)

        monkeypatch.setattr(Pipeline, "run", run)
        with pytest.raises(SystemExit):
            cli.main(
                ["process", str(cap), "--price-divisor", "10",
                 "--timestamp-unit", "us", "--symbol", "BTC/USD",
                 "--venue", "kraken", "-o", str(tmp_path / "out")]
            )  # fmt: skip
        assert seen["settings"].price_divisor == 10
        assert seen["settings"].timestamp_unit == "us"
        assert (seen["ctx"].symbol, seen["ctx"].venue) == ("BTC/USD", "kraken")

    def test_symbol_and_venue_tag_the_rows(self, cli_runner, tmp_path):
        cap = _ccxt_capture(tmp_path / "cap")
        out = tmp_path / "out"
        r = cli_runner(
            "process", str(cap), "--symbol", "BTC/USD", "--venue", "kraken",
            "-o", str(out),
        )  # fmt: skip
        assert r.returncode == 0, r.stderr
        depth = pd.read_parquet(out / "depth.parquet")
        assert set(depth["symbol"]) == {"BTC/USD"}
        assert set(depth["venue"]) == {"kraken"}

    def test_a_bad_setting_is_refused(self, cli_runner, tmp_path):
        cap = _ccxt_capture(tmp_path / "cap")
        r = cli_runner("process", str(cap), "--price-divisor", "0")
        assert r.returncode == 2
        assert "Bad settings for source 'depth_csv'" in r.stderr

    def test_settings_are_refused_on_a_saved_result(self, cli_runner, tmp_path):
        r = cli_runner(
            "audit", str(tmp_path), "--from-parquet", "--source", "depth_csv",
            "--price-divisor", "10", "--symbol", "X",
        )  # fmt: skip
        assert r.returncode == 2
        assert "--symbol, --price-divisor set how the pipeline runs" in r.stderr


class TestSettingsFlagBeforeThePath:
    """A settings flag works before the path as well as after it (#523)."""

    @pytest.mark.parametrize("command", ["process", "audit"])
    @pytest.mark.parametrize("before", [True, False], ids=["before", "after"])
    def test_the_flag_reaches_the_reading_source(
        self, cli_runner, tmp_path, command, before
    ):
        cap = str(_ccxt_capture(tmp_path / "cap"))
        flag = ["--price-divisor", "0"]
        r = cli_runner(command, *(flag + [cap] if before else [cap] + flag))
        # The value 0 is refused by the depth_csv settings, so the flag reached
        # them and was not read as the path.
        assert r.returncode == 2
        assert "Bad settings for source 'depth_csv'" in r.stderr

    def test_a_value_before_the_path_is_not_the_path(self, monkeypatch, tmp_path):
        from ob_analytics import cli
        from ob_analytics.pipeline import Pipeline

        cap = _ccxt_capture(tmp_path / "cap")
        seen: dict = {}

        def run(self, path, *args, **kwargs):
            seen["path"] = Path(path)
            seen["settings"] = self._source.settings
            raise SystemExit(0)

        monkeypatch.setattr(Pipeline, "run", run)
        with pytest.raises(SystemExit):
            cli.main(
                ["process", "--price-divisor", "10", "--timestamp-unit=us",
                 str(cap), "-o", str(tmp_path / "out")]
            )  # fmt: skip
        assert seen["path"] == cap
        assert seen["settings"].price_divisor == 10
        assert seen["settings"].timestamp_unit == "us"

    @pytest.mark.parametrize("command", ["process", "audit"])
    @pytest.mark.parametrize("on_off", ["--fast", "--no-fast"])
    def test_an_on_off_flag_takes_no_value(self, monkeypatch, command, on_off):
        from ob_analytics import cli
        from ob_analytics.sources import SOURCES
        from tests._fake_plugin_source import FakePluginSource

        class OfflineFake(FakePluginSource):
            def create_loader(self, config):  # pragma: no cover - not called
                raise NotImplementedError

        monkeypatch.setitem(SOURCES._items, "offlinefake", OfflineFake)
        ignored = cli._every_settings_flag()
        assert ignored["--fast"] is True
        assert ignored["--market"] is False
        parser = cli._build_parser(quiet=True, ignored=ignored)
        args, unknown = parser.parse_known_args(
            [command, on_off, "--market", "M-1", "cap/"]
        )
        assert args.path == "cap/"
        assert unknown == []

    def test_a_flag_with_no_value_is_named(self, cli_runner, tmp_path):
        cap = str(_ccxt_capture(tmp_path / "cap"))
        r = cli_runner("process", cap, "--timestamp-unit")
        assert r.returncode == 2
        assert "--timestamp-unit: expected one argument" in r.stderr

    def test_a_shortened_flag_still_finds_the_source(self, cli_runner, tmp_path):
        """``--trad`` is short for ``--trading-date`` only, among this source's flags."""
        cap = str(_ccxt_capture(tmp_path / "cap"))
        r = cli_runner(
            "process", cap, "--trad", "2015-05-01", "--price-divisor", "0"
        )  # fmt: skip
        assert r.returncode == 2
        assert "Bad settings for source 'depth_csv'" in r.stderr

    def test_a_flag_of_another_source_before_the_path_is_refused(
        self, cli_runner, tmp_path
    ):
        cap = str(_ccxt_capture(tmp_path / "cap"))
        r = cli_runner("process", "--instrument-id", "5", cap)
        assert r.returncode == 2
        assert "unrecognized arguments" in r.stderr
