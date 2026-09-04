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

import pytest

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
        assert "trading" in (r.stderr + r.stdout).lower()

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

        r = cli_runner("audit", str(out), "--from-parquet")
        assert r.returncode == 0, r.stderr
        assert "Data quality summary" in r.stdout
        # No --source, so the feed type stays undeclared rather than guessed.
        assert "unknown" in r.stdout

        r = cli_runner("audit", str(out), "--from-parquet", "--source", "bitstamp")
        assert r.returncode == 0, r.stderr
        assert "diff_feed" in r.stdout


# ---------------------------------------------------------------------------
# gallery
# ---------------------------------------------------------------------------


class TestGallerySubcommand:
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

    def test_gallery_missing_data(self, cli_runner, tmp_path):
        r = cli_runner("gallery", str(tmp_path / "nonexistent"))
        assert r.returncode != 0


# ---------------------------------------------------------------------------
# bitstamp-demo
# ---------------------------------------------------------------------------


class TestBitstampDemoSubcommand:
    def test_explicit_input(self, cli_runner, tmp_path, tiny_bitstamp_orders_csv):
        """bitstamp-demo --input <dir-with-orders-and-trades> succeeds.

        Uses the tiny programmatic fixture instead of the bundled sample
        (bundled-sample bitstamp-demo takes minutes through full pipeline +
        gallery; that path is exercised manually).
        """
        out = tmp_path / "bs_out"
        r = cli_runner(
            "bitstamp-demo",
            "--input",
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
            "--input",
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
            "--input",
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

        assert BitstampSource().required_context() == []
        assert LobsterSource().required_context() == ["trading_date"]


# ---------------------------------------------------------------------------
# capture (CCXT wiring, #106)
# ---------------------------------------------------------------------------


class TestCaptureSubcommand:
    def test_ccxt_flags_in_help(self, cli_runner):
        r = cli_runner("capture", "--help")
        assert r.returncode == 0
        for flag in ("--exchange", "--depth-limit", "--poll-interval"):
            assert flag in r.stdout
        assert "depth.csv" in r.stdout  # L2 output documented

    @pytest.mark.skipif(not _CCXT_INSTALLED, reason="ccxt extra not installed")
    def test_list_includes_ccxt(self, cli_runner):
        r = cli_runner("capture", "--list")
        assert r.returncode == 0, r.stderr
        assert "bitstamp" in r.stdout
        assert "ccxt" in r.stdout

    def test_exchange_flag_flows_into_typed_settings(self, monkeypatch, tmp_path):
        """--exchange / --depth-limit flow into typed CcxtSettings (no network)."""
        import argparse

        import pandas as pd

        from ob_analytics import cli
        from ob_analytics.live._base import CaptureConfig, CaptureResult
        from ob_analytics.live.ccxt_source import CcxtSettings, CcxtSource

        captured: dict = {}

        async def _fake_run(source, config, sink=None):
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
            )

        monkeypatch.setattr("ob_analytics.live._runner.run_capturer", _fake_run)

        args = argparse.Namespace(
            verbose=False,
            list=False,
            venue="ccxt",
            pair="BTC/USDT",
            exchange="binance",
            depth_limit=50,
            poll_interval=None,
            minutes=0.001,
            out=str(tmp_path / "o"),
            no_raw=True,
        )
        cli._cmd_capture(args)

        # The venue knobs land as typed settings on the source, not on the config.
        source = captured["source"]
        assert isinstance(source, CcxtSource)
        assert isinstance(source.settings, CcxtSettings)
        assert source.settings.exchange == "binance"
        assert source.settings.depth_limit == 50
        assert source.settings.poll_interval == 1.0  # not passed -> default

        cfg = captured["config"]
        assert isinstance(cfg, CaptureConfig)
        assert cfg.pair == "BTC/USDT"
        assert not hasattr(cfg, "extras")  # the untyped dict is gone
