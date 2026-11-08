"""Run the Python code blocks on each docs page, in order, and report failures.

Each page runs in a new Python process, so one page cannot see another's
names. The blocks of one page share one namespace and run from top to bottom,
as a reader would paste them. The working folder is a new folder that holds the
bundled Bitstamp sample as ``orders.csv``, ``trades.csv`` and ``meta.json``,
the names the pages give a reader's own capture. A page that writes files
writes them there.

HTML comments control the run. The published site does not show them. Blank
lines may stand between a marker and its code block.

``<!-- docs-test: skip REASON -->``
    before a Python block leaves that block out. Use it only for a block that
    cannot run as written: one that needs an API key, the network, a package
    the docs job does not install, or a file or code the reader must supply.
    The reason is required.

``<!-- docs-test`` then lines of Python, then ``-->``
    runs that code at that point of the page. Use it to make a file the page
    expects the reader to have, or to assert a figure the page quotes.

``<!-- docs-test: output -->``
    before a code block of any language fails the run unless the block equals
    what the Python block before it printed. That block must run: it cannot be
    skipped.

``<!-- docs-test: output EXPRESSION -->``
    the same, but compares the block with the text of the expression.

In a quoted output, a line that ends in ``...`` matches any line that starts
with the text before the ``...``. Trailing spaces and blank lines at the end
are ignored. Blocks in other languages (``bash``, ``text`` and so on) never run
unless an output marker names them.

    uv run python scripts/check_doc_examples.py                 # every page
    uv run python scripts/check_doc_examples.py docs/howto/bars.md
    uv run python scripts/check_doc_examples.py -j 3            # 3 pages at once
    uv run python scripts/check_doc_examples.py --list          # list the blocks
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The pages whose code blocks run. The tutorial chapters are not here:
#: ``build_tutorial.py`` builds them from ``docs/tutorial/src/*.py`` and runs
#: that code.
PAGE_GLOBS = ("docs/*.md", "docs/howto/*.md", "docs/api/*.md")

_FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<mark>`{3,}|~{3,})\s*(?P<info>[^`]*)$")
_SKIP = re.compile(r"^\s*<!--\s*docs-test:\s*skip\b\s*(?P<reason>.*?)\s*-->\s*$")
_OUTPUT = re.compile(r"^\s*<!--\s*docs-test:\s*output\b\s*(?P<expr>.*?)\s*-->\s*$")
_HIDDEN = re.compile(r"^\s*<!--\s*docs-test\s*$")


@dataclass(frozen=True)
class Block:
    """One step of a page: a code block, hidden code, or an output check."""

    line: int  # 1-based line of the code (or of the marker, for a check)
    code: str  # the Python code, or the expression of an output check
    kind: str = "run"  # "run", "hidden" or "output"
    skip: str | None = None  # why the block is left out
    expected: str | None = None  # the quoted text of an output check


class PageError(ValueError):
    """A page's docs-test markers are malformed."""


def _language(info: str) -> str:
    """Return the language of a fence's info string, such as ``python title="x"``."""
    info = info.strip()
    if info.startswith("{"):  # {.python .annotate}
        info = info.strip("{}").strip().lstrip(".")
    return info.split()[0].lower() if info else ""


def _fence_end(lines: list[str], start: int, mark: str) -> int:
    """Return the index of the line that closes the fence opened at *start*."""
    for end in range(start + 1, len(lines)):
        close = lines[end].strip()
        if close.startswith(mark) and not close.strip(mark[0]):
            return end
    raise PageError(f"line {start + 1}: code block is not closed")


def extract_blocks(text: str) -> list[Block]:
    """Return the steps of a page, in order."""
    lines = text.splitlines()
    blocks: list[Block] = []
    marker: tuple[int, re.Match[str]] | None = None  # skip or output, waiting
    last_run: Block | None = None  # the last Python code block, for an output
    i = 0
    while i < len(lines):
        line = lines[i]
        found = _SKIP.match(line) or _OUTPUT.match(line)
        if found:
            if marker is not None:
                raise PageError(f"line {i + 1}: two docs-test markers on one block")
            if found.re is _SKIP and not found["reason"]:
                raise PageError(f"line {i + 1}: docs-test: skip needs a reason")
            marker = (i + 1, found)
            i += 1
            continue
        if _HIDDEN.match(line):
            if marker is not None:
                raise PageError(f"line {marker[0]}: docs-test marker has no code block")
            end = i + 1
            while end < len(lines) and lines[end].strip() != "-->":
                end += 1
            if end == len(lines):
                raise PageError(f"line {i + 1}: docs-test comment is not closed")
            code = textwrap.dedent("\n".join(lines[i + 1 : end])) + "\n"
            blocks.append(Block(line=i + 2, code=code, kind="hidden"))
            i = end + 1
            continue
        fence = _FENCE.match(line)
        if not fence:
            if line.strip() and marker is not None:
                raise PageError(f"line {marker[0]}: docs-test marker has no code block")
            i += 1
            continue
        indent = fence["indent"]
        end = _fence_end(lines, i, fence["mark"])
        body = "\n".join(
            ln[len(indent) :] if ln.startswith(indent) else ln.lstrip()
            for ln in lines[i + 1 : end]
        )
        is_python = _language(fence["info"]) in ("python", "py")
        if marker is not None and marker[1].re is _OUTPUT:
            expr = marker[1]["expr"]
            if not expr and (last_run is None or last_run.skip is not None):
                raise PageError(
                    f"line {marker[0]}: docs-test: output has no Python block "
                    "before it that runs"
                )
            blocks.append(
                Block(line=marker[0], code=expr, kind="output", expected=body + "\n")
            )
        elif is_python:
            reason = marker[1]["reason"] if marker is not None else None
            last_run = Block(line=i + 2, code=body + "\n", skip=reason)
            blocks.append(last_run)
        elif marker is not None:
            raise PageError(
                f"line {marker[0]}: docs-test: skip is on a non-Python block"
            )
        marker = None
        i = end + 1
    if marker is not None:
        raise PageError(f"line {marker[0]}: docs-test marker has no code block")
    return blocks


#: Runs in the child process. Each block is compiled under its page and line,
#: so a traceback points at the docs source.
_RUNNER = r"""
import contextlib, io, json, sys
import matplotlib

matplotlib.use("Agg")

# A page may call fig.show(). Plotly would then open a browser and wait for it.
try:
    import plotly.io as pio
    from plotly.io._base_renderers import ExternalRenderer
except ImportError:
    pass
else:

    class _NoBrowser(ExternalRenderer):
        def render(self, fig_dict):
            pass

    pio.renderers["docs-test"] = _NoBrowser()
    pio.renderers.default = "docs-test"


def same(quoted, actual):
    quoted = quoted.rstrip().splitlines()
    actual = actual.rstrip().splitlines()
    if len(quoted) != len(actual):
        return False
    for q, a in zip(quoted, actual):
        q, a = q.rstrip(), a.rstrip()
        if q.endswith("...") and a.startswith(q[:-3].rstrip()):
            continue
        if q != a:
            return False
    return True


namespace = {"__name__": "__main__"}
printed = ""
for page, line, code, kind, expected in json.load(sys.stdin):
    print(f"--- {page}:{line}", flush=True)
    if kind == "output":
        if code:
            actual = str(eval(compile(code, f"{page}:{line}", "eval"), namespace))
        else:
            actual = printed
        if not same(expected, actual):
            sys.exit(
                f"{page}:{line}: the quoted output differs from "
                f"{code or 'what the block before printed'}\n"
                f"--- quoted\n{expected.rstrip()}\n--- actual\n{actual.rstrip()}"
            )
        continue
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(compile("\n" * (line - 1) + code, page, "exec"), namespace)
    finally:
        sys.stdout.write(buffer.getvalue())
    if kind == "run":
        printed = buffer.getvalue()
"""


def _put_sample(folder: Path) -> None:
    """Write the bundled sample into *folder*, with ``orders.csv`` unzipped."""
    sample = ROOT / "ob_analytics" / "_sample_data"
    with gzip.open(sample / "orders.csv.gz", "rb") as src:
        (folder / "orders.csv").write_bytes(src.read())
    for name in ("trades.csv", "meta.json"):
        shutil.copy(sample / name, folder / name)


def page_blocks(page: Path) -> list[Block]:
    """Return the steps of a page file; a malformed marker names the page."""
    try:
        return extract_blocks(page.read_text(encoding="utf-8"))
    except PageError as exc:
        raise PageError(f"{page.relative_to(ROOT).as_posix()}: {exc}") from None


def run_page(page: Path, timeout: float) -> tuple[bool, str, float]:
    """Run one page in a new process; return (passed, output, seconds)."""
    rel = page.relative_to(ROOT).as_posix()
    steps = [b for b in page_blocks(page) if b.skip is None]
    payload = json.dumps([(rel, b.line, b.code, b.kind, b.expected) for b in steps])
    env = {**os.environ, "MPLBACKEND": "Agg"}
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="doc-examples-") as cwd:
        _put_sample(Path(cwd))
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _RUNNER],
                input=payload,
                capture_output=True,
                text=True,
                cwd=cwd,
                env=env,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout or ""
            if isinstance(out, bytes):
                out = out.decode(errors="replace")
            seconds = time.perf_counter() - start
            return False, f"{out}\ntimed out after {timeout:.0f} s", seconds
    output = proc.stdout + proc.stderr
    return proc.returncode == 0, output, time.perf_counter() - start


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pages", nargs="*", type=Path, help="pages (default: all)")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="pages run at once")
    parser.add_argument("--timeout", type=float, default=900, help="seconds per page")
    parser.add_argument("--list", action="store_true", help="list the steps only")
    ns = parser.parse_args(argv)

    if ns.pages:
        pages = [p.resolve() for p in ns.pages]
    else:
        pages = sorted({p for g in PAGE_GLOBS for p in ROOT.glob(g)})
    try:
        steps = {page: page_blocks(page) for page in pages}
    except PageError as exc:
        print(exc, file=sys.stderr)
        return 2
    if ns.list:
        for page, blocks in steps.items():
            for b in blocks:
                kind = f"skip: {b.skip}" if b.skip else b.kind
                print(f"{page.relative_to(ROOT).as_posix()}:{b.line}\t{kind}")
        return 0
    pages = [p for p, blocks in steps.items() if any(b.skip is None for b in blocks)]

    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, ns.jobs)) as pool:
        results = pool.map(lambda p: run_page(p, ns.timeout), pages)
        for page, (ok, output, seconds) in zip(pages, results, strict=True):
            rel = page.relative_to(ROOT).as_posix()
            print(f"{'ok  ' if ok else 'FAIL'} {rel} ({seconds:.0f} s)", flush=True)
            if not ok:
                failed.append(rel)
                print(textwrap.indent(output.rstrip()[-6000:], "    "), flush=True)
    if failed:
        print(f"\n{len(failed)} page(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"\nall {len(pages)} page(s) ran")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
