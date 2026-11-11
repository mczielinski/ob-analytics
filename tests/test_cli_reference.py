"""The CLI flag reference page matches the command's parser.

``docs/api/cli-flags.md`` is made by ``scripts/cli_reference.py``.  A flag
added, renamed or reworded without running the script fails here.
"""

from __future__ import annotations

import subprocess
import sys

from scripts.cli_reference import PAGE


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    # A new interpreter, so a source or writer another test registered in
    # this process cannot change the --source and --format choices.
    script = PAGE.parents[2] / "scripts" / "cli_reference.py"
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_flag_reference_page_is_up_to_date() -> None:
    result = _run("--check")
    assert result.returncode == 0, result.stderr


def test_the_page_lists_every_subcommand_and_live_source() -> None:
    text = PAGE.read_text(encoding="utf-8")
    for command in (
        "process",
        "audit",
        "gallery",
        "bitstamp-demo",
        "lobster-demo",
        "capture",
        "capture bitstamp",
        "capture ccxt",
        "capture cryptofeed",
        "sources",
    ):
        assert f"## `ob-analytics {command}`" in text


def test_help_text_is_used_as_written() -> None:
    # The script does not expand argparse's %(default)s placeholders.
    assert "%(" not in PAGE.read_text(encoding="utf-8")
