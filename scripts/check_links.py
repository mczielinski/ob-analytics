"""Fail if the built docs site has broken internal links or anchors.

Parses every ``site/**/*.html`` page, resolves each relative link (and its
``#fragment``) against the rendered output, and exits non-zero if any target
page or anchor is missing. External URLs are not fetched (no network
flakiness); root-absolute links (the ``/ob-analytics/`` deploy prefix) and
the generated ``404.html`` are skipped, since those only resolve on the
deployed Pages site, not in the local tree.

    uv run python scripts/check_links.py            # checks ./site
    uv run python scripts/check_links.py path/to/site
"""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urldefrag


class _Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d = dict(attrs)
        for key in ("id", "name"):
            if d.get(key):
                self.ids.add(d[key])  # type: ignore[arg-type]
        if tag == "a" and d.get("href"):
            self.links.append(d["href"])  # type: ignore[arg-type]


def main(argv: list[str]) -> int:
    root = Path(argv[0]) if argv else Path("site")
    if not root.is_dir():
        print(
            f"no site directory at {root!r} — run the docs build first", file=sys.stderr
        )
        return 2

    pages: dict[Path, _Page] = {}
    for html in root.rglob("*.html"):
        p = _Page()
        p.feed(html.read_text(encoding="utf-8"))
        pages[html.resolve()] = p

    broken: list[tuple[Path, str, str]] = []
    for path, page in pages.items():
        if path.name == "404.html":
            continue
        for href in page.links:
            if href.startswith(("http://", "https://", "mailto:", "/")):
                continue
            if href.startswith("#"):
                frag = unquote(href[1:])
                if frag and frag not in page.ids:
                    broken.append((path, href, "in-page anchor"))
                continue
            url, frag = urldefrag(href)
            target = (path.parent / url).resolve() if url else path
            if target.is_dir():
                target = target / "index.html"
            if url and not target.exists():
                broken.append((path, href, "missing page"))
                continue
            if frag:
                tp = pages.get(target)
                if tp is None and target.exists():
                    tp = _Page()
                    tp.feed(target.read_text(encoding="utf-8"))
                if tp is None or unquote(frag) not in tp.ids:
                    broken.append((path, href, "missing anchor"))

    if broken:
        print(f"Found {len(broken)} broken internal link(s):", file=sys.stderr)
        for path, href, why in broken:
            print(
                f"  {path.relative_to(root.resolve())} -> {href}  ({why})",
                file=sys.stderr,
            )
        return 1
    print(f"OK — {len(pages)} pages, no broken internal links or anchors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
