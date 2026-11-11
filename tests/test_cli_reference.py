"""The CLI flag reference page matches the command's parser.

``docs/api/cli-flags.md`` is made by ``scripts/cli_reference.py``.  A flag
added, renamed or reworded without running the script fails here.
"""

from __future__ import annotations

from scripts.cli_reference import PAGE, main, render


def test_the_flag_reference_page_is_up_to_date() -> None:
    assert main(["--check"]) == 0, (
        "docs/api/cli-flags.md is stale: run `uv run python scripts/cli_reference.py`"
    )


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
    assert "%(" not in render()
