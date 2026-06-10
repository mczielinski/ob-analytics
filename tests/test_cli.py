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
            "--format",
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
            "--format",
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
            "--format",
            "bitstamp",
            "--output",
            str(out),
            "--gallery",
        )
        assert r.returncode == 0, r.stderr
        assert (out / "gallery" / "gallery.html").exists()


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
            "--format",
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
