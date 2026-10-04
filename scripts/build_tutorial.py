"""Execute tutorial chapter sources and render them into the docs site.

Chapters live as jupytext py:percent scripts in ``docs/tutorial/src/``
(``NN_slug.py``). This script converts each to a notebook, executes it
top-to-bottom (any error fails the build — the guarantee that the tutorial
can never drift from the API), and renders the result to
``docs/tutorial/NN_slug.md`` plus extracted figure PNGs in
``docs/tutorial/NN_slug_files/``. A warning printed by any cell would be
rendered into the page, and so would a traceback raised while a figure is
displayed (the cell itself succeeds), so both fail the build too, once every
chapter has been built. The generated files are gitignored;
CI runs this before ``zensical build`` (see .github/workflows/docs.yml),
and locally you do the same:

    uv run python scripts/build_tutorial.py            # all chapters
    uv run python scripts/build_tutorial.py 00         # chapters matching a prefix
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

import jupytext
from nbclient import NotebookClient
from nbconvert import MarkdownExporter

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "docs" / "tutorial" / "src"
OUT = REPO / "docs" / "tutorial"

# The first line of a printed warning: "path:line: UserWarning: message".
_WARNING_LINE = re.compile(r"^.*?:\d+: (\w*Warning): .*$", flags=re.MULTILINE)


def _title_of(markdown: str, fallback: str) -> str:
    m = re.search(r"^# (.+)$", markdown, flags=re.MULTILINE)
    return m.group(1).strip() if m else fallback


def _rendered_problems(nb) -> list[str]:
    """The warnings and tracebacks in an executed notebook's outputs.

    One ``cell N: ...`` line each.  These outputs would be rendered into the
    page as they are.
    """
    found = []
    for n, cell in enumerate(nb.cells):
        for output in cell.get("outputs", []):
            kind = output.get("output_type")
            if kind == "error":
                found.append(f"cell {n}: {output['ename']}: {output['evalue']}")
            elif kind == "stream" and output.get("name") == "stderr":
                found += [
                    f"cell {n}: {m.group(0).strip()}"
                    for m in _WARNING_LINE.finditer(output.get("text", ""))
                ]
    return found


def build_chapter(path: Path) -> tuple[str, list[str]]:
    stem = path.stem
    nb = jupytext.read(path)

    # Execute with the kernel cwd set to a throwaway scratch dir, so any
    # example that writes a relative file (save_figure, write_html, …) drops
    # it there instead of polluting docs/tutorial/src/. The chdir moves the
    # kernel off SRC, so pin SRC on sys.path first (chapters import the shared
    # `_docs_theme` from there). The bootstrap cell is removed before the
    # notebook reaches the renderer, so it never shows in the docs.
    import nbformat

    nb.cells.insert(
        0,
        nbformat.v4.new_code_cell(
            "import os as _os, sys as _sys, tempfile as _tf\n"
            f"_sys.path.insert(0, {str(SRC)!r})\n"
            "_os.chdir(_tf.mkdtemp())"
        ),
    )

    t0 = time.perf_counter()
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(SRC)}},
    )
    client.execute()
    nb.cells.pop(0)  # drop the bootstrap cell before rendering
    problems = _rendered_problems(nb)

    exporter = MarkdownExporter()
    body, resources = exporter.from_notebook_node(
        nb,
        resources={"output_files_dir": f"{stem}_files", "unique_key": stem},
    )

    files_dir = OUT / f"{stem}_files"
    if files_dir.exists():
        shutil.rmtree(files_dir)
    outputs: dict[str, bytes] = resources.get("outputs", {})
    for relname, payload in outputs.items():
        dest = OUT / relname
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)

    # Quote the title: chapter headings contain colons ("L1 -> L2 -> L3: three
    # resolutions ..."), and an unquoted YAML scalar with ": " in it is a parse
    # error. json.dumps writes a double-quoted scalar, which YAML reads with the
    # same escapes, so any character in a heading survives the round trip.
    title = json.dumps(_title_of(body, stem), ensure_ascii=False)
    (OUT / f"{stem}.md").write_text(
        f"---\ntitle: {title}\n---\n\n{body}", encoding="utf-8"
    )
    summary = f"{stem}: {len(outputs)} figure(s), {time.perf_counter() - t0:.1f}s"
    return summary, problems


def main(argv: list[str]) -> int:
    prefixes = argv or [""]
    chapters = sorted(
        p
        for p in SRC.glob("[0-9][0-9]_*.py")
        if any(p.name.startswith(pre) for pre in prefixes)
    )
    if not chapters:
        print(f"no chapter sources matching {prefixes} under {SRC}", file=sys.stderr)
        return 1
    failed: dict[str, list[str]] = {}
    for chapter in chapters:
        print(f"building {chapter.relative_to(REPO)} ...", flush=True)
        summary, problems = build_chapter(chapter)
        print(f"  {summary}")
        if problems:
            failed[chapter.name] = problems
    if failed:
        print("\nwarnings or errors rendered into the pages:", file=sys.stderr)
        for name, lines in failed.items():
            for line in lines:
                print(f"  {name} {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
