"""Execute tutorial chapter sources and render them into the docs site.

Chapters live as jupytext py:percent scripts in ``docs/tutorial/src/``
(``NN_slug.py``). This script converts each to a notebook, executes it
top-to-bottom (any error fails the build — the guarantee that the tutorial
can never drift from the API), and renders the result to
``docs/tutorial/NN_slug.md`` plus extracted figure PNGs in
``docs/tutorial/NN_slug_files/``. The generated files are gitignored;
CI runs this before ``zensical build`` (see .github/workflows/docs.yml),
and locally you do the same:

    uv run python scripts/build_tutorial.py            # all chapters
    uv run python scripts/build_tutorial.py 00         # chapters matching a prefix
"""

from __future__ import annotations

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


def _title_of(markdown: str, fallback: str) -> str:
    m = re.search(r"^# (.+)$", markdown, flags=re.MULTILINE)
    return m.group(1).strip() if m else fallback


def build_chapter(path: Path) -> str:
    stem = path.stem
    nb = jupytext.read(path)

    t0 = time.perf_counter()
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(SRC)}},
    )
    client.execute()

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

    title = _title_of(body, stem)
    (OUT / f"{stem}.md").write_text(
        f"---\ntitle: {title}\n---\n\n{body}", encoding="utf-8"
    )
    return f"{stem}: {len(outputs)} figure(s), {time.perf_counter() - t0:.1f}s"


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
    for chapter in chapters:
        print(f"building {chapter.relative_to(REPO)} ...", flush=True)
        print(f"  {build_chapter(chapter)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
