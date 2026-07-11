"""Render the example gallery into the docs site.

Runs every spec in ``docs/examples/gallery_examples.py`` on the bundled
sample, saves each figure, and writes ``docs/gallery/index.md``: a
thumbnail grid grouped by category, then one section per example showing
the full figure, its caption, and the exact function that produced it.
The generated page and images are gitignored; CI runs this before
``zensical build`` (see .github/workflows/docs.yml), and locally:

    uv run python scripts/build_gallery.py
"""

from __future__ import annotations

import inspect
import os
import tempfile
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "gallery"
IMAGES = OUT / "images"


def _source_of(fn) -> str:
    """The function's source with its ``def`` line and docstring stripped —
    just the recipe body, dedented and ready to copy."""
    lines = inspect.getsource(fn).splitlines()
    body = lines[1:]  # drop the `def ...:` line
    text = textwrap.dedent("\n".join(body))
    # drop a leading docstring block if present
    stripped = text.lstrip()
    for q in ('"""', "'''"):
        if stripped.startswith(q):
            end = stripped.index(q, 3) + 3
            text = stripped[end:].lstrip("\n")
            break
    return text.rstrip()


def main() -> int:
    from ob_analytics import Pipeline, sample_csv_path

    import sys

    sys.path.insert(0, str(REPO / "docs" / "examples"))
    from gallery_examples import GALLERY  # type: ignore[import-not-found]

    IMAGES.mkdir(parents=True, exist_ok=True)
    result = Pipeline().run(sample_csv_path())

    # Render each example from a scratch cwd (no stray files in the repo).
    prev_cwd = os.getcwd()
    os.chdir(tempfile.mkdtemp())
    try:
        rendered = []
        for ex in GALLERY:
            fig = ex.render(result)
            # Full figure: enlarge ~40% (preserving aspect) so the per-example
            # render on the page is big and legible, saved at a crisp dpi.
            w, h = fig.get_size_inches()
            fig.set_size_inches(w * 1.4, h * 1.4)
            fig.savefig(IMAGES / f"{ex.name}.png", dpi=130, bbox_inches="tight")
            # Thumbnail: reduce to just the primary graphic — a visual
            # impression, like the seaborn example gallery. Drop colorbars,
            # keep only the largest panel (so small-multiple faces like
            # cancellations show one panel, not two), and strip all chrome.
            for ax in list(fig.axes):
                if ax.get_label() == "<colorbar>":
                    ax.remove()
            data_axes = list(fig.axes)
            if len(data_axes) > 1:
                largest = max(
                    data_axes,
                    key=lambda a: a.get_position().width * a.get_position().height,
                )
                for ax in data_axes:
                    if ax is not largest:
                        ax.remove()
            for ax in fig.axes:
                ax.set_title("")
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.axis("off")
                if (leg := ax.get_legend()) is not None:
                    leg.remove()
            fig.savefig(
                IMAGES / f"{ex.name}_thumb.png",
                dpi=90,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close(fig)
            rendered.append(ex)
            print(f"  {ex.name}: {ex.title}")
    finally:
        os.chdir(prev_cwd)

    # Group by category, preserving first-seen order.
    categories: dict[str, list] = {}
    for ex in rendered:
        categories.setdefault(ex.category, []).append(ex)

    out = ["---", "title: Example gallery", "---", "", "# Example gallery", ""]
    out += [
        "Every figure ob-analytics draws, rendered from the bundled Bitstamp",
        "sample. Click a thumbnail for the full figure and the exact code that",
        "produced it. New to these? The [tutorial](../tutorial/index.md) builds",
        "each one up from first principles.",
        "",
    ]

    # Thumbnail grid.
    out.append('<div style="display:flex;flex-wrap:wrap;gap:1rem;margin:1.5rem 0;">')
    for ex in rendered:
        out.append(
            f'<a href="#{ex.name}" title="{ex.title}" '
            'style="flex:0 0 auto;width:300px;text-align:center;'
            'text-decoration:none;font-size:0.85rem;">'
            f'<img src="images/{ex.name}_thumb.png" alt="{ex.title}" '
            'style="width:300px;height:200px;object-fit:contain;'
            'border:1px solid #ccc;border-radius:4px;background:#fff;padding:6px;"><br>'
            f"{ex.title}</a>"
        )
    out.append("</div>")
    out.append("")

    # Per-category sections with full figure + code.
    for category, exs in categories.items():
        out += [f"## {category}", ""]
        for ex in exs:
            out += [
                f"### {ex.title} {{#{ex.name}}}",
                "",
                f"![{ex.title}](images/{ex.name}.png)",
                "",
                f"*{ex.caption}*",
                "",
                "```python",
                _source_of(ex.render),
                "```",
                "",
            ]

    (OUT / "index.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"gallery: {len(rendered)} examples -> {OUT / 'index.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
