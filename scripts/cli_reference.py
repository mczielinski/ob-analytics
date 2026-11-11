"""Write the CLI flag reference page from the command's own parser.

``docs/api/cli-flags.md`` lists every flag of every ``ob-analytics``
subcommand, and of ``capture`` for each built-in live source.  It is made from
the same ``argparse`` parsers the command reads, so it cannot drift from them.

Usage::

    uv run python scripts/cli_reference.py          # rewrite the page
    uv run python scripts/cli_reference.py --check  # exit 1 if it is stale

``tests/test_cli_reference.py`` runs the check.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from ob_analytics.cli import _build_parser, _capture_parser, _settings_class
from ob_analytics.sources import get_source, list_sources

PAGE = Path(__file__).resolve().parents[1] / "docs" / "api" / "cli-flags.md"

_HEADER = """\
---
title: CLI flag reference
---

<!-- Made by scripts/cli_reference.py from the command's parser. Do not edit
by hand: run `uv run python scripts/cli_reference.py` after a flag changes. -->

# CLI flag reference

Every flag of every `ob-analytics` subcommand. Each subcommand also takes
`-h` / `--help`, which prints the same list. Every subcommand accepts the
top-level `-v` / `--verbose` flag before its name, for debug-level logging:
`ob-analytics -v process orders.csv`.

For examples, see [Run from the command line](../howto/cli.md).
"""


def _flag(action: argparse.Action) -> str:
    """The flag as a user types it, with a placeholder for its value."""
    if not action.option_strings:
        return f"`{action.metavar or action.dest}`"
    takes_value = action.nargs != 0
    metavar = action.metavar or action.dest.upper()
    return ", ".join(
        f"`{option} {metavar}`" if takes_value else f"`{option}`"
        for option in action.option_strings
    )


def _description(action: argparse.Action) -> str:
    # The help is used as written: no flag of the command uses argparse's
    # %(default)s expansion, and reading it raw keeps the page the same on
    # every Python version.
    text = re.sub(r"\s+", " ", action.help or "").strip()
    choices = [str(c) for c in action.choices or ()]
    if choices and not all(c in text for c in choices):
        listed = ", ".join(f"`{c}`" for c in choices)
        text = f"{text.rstrip('.')}. One of {listed}." if text else f"One of {listed}."
    if action.required and action.option_strings:
        text = f"Required. {text}"
    return text.replace("|", "\\|")


def _table(parser: argparse.ArgumentParser) -> str:
    rows = [
        f"| {_flag(action)} | {_description(action)} |"
        for action in parser._actions
        if action.help != argparse.SUPPRESS
        and not isinstance(action, (argparse._HelpAction, argparse._SubParsersAction))
    ]
    if not rows:
        return "This subcommand takes no flags.\n"
    return "| Flag | Description |\n|------|-------------|\n" + "\n".join(rows) + "\n"


def _section(title: str, parser: argparse.ArgumentParser) -> str:
    about = re.sub(r"\s+", " ", parser.description or "").strip()
    parts = [f"## `{title}`\n"]
    if about:
        parts.append(about + "\n")
    parts.append(_table(parser))
    return "\n".join(parts)


def _live_sources() -> list[str]:
    """The built-in sources that can capture live, in registry order."""
    return [
        name
        for name in list_sources()
        if hasattr(get_source(name), "stream")
        and get_source(name).__module__.startswith("ob_analytics.")
    ]


def render() -> str:
    """The text of the flag reference page."""
    root = _build_parser()
    subparsers = next(
        a for a in root._actions if isinstance(a, argparse._SubParsersAction)
    )
    sections = [_HEADER]
    seen: set[int] = set()
    for name, sub in subparsers.choices.items():
        if id(sub) in seen:  # an alias of a subcommand already listed
            continue
        seen.add(id(sub))
        sections.append(_section(f"ob-analytics {name}", sub))
        if name == "capture":
            for source in _live_sources():
                capture = _capture_parser(source, _settings_class(get_source(source)))
                sections.append(_section(f"ob-analytics capture {source}", capture))
    return "\n".join(sections)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if the page is stale"
    )
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        if not PAGE.exists() or PAGE.read_text(encoding="utf-8") != text:
            print(
                f"{PAGE} is stale: run `uv run python scripts/cli_reference.py`",
                file=sys.stderr,
            )
            return 1
        return 0
    PAGE.write_text(text, encoding="utf-8")
    print(f"wrote {PAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
