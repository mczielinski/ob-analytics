"""PROTOTYPE - throwaway. Fetch the live #124 graph from GitHub into a JSON cache.

Not production code. Answers issue #171: what does the generated epic body
look like against the real graph?

    uv run --no-project python scripts/prototype_roadmap_fetch.py
"""

import json
import subprocess
import sys

REPO = "mczielinski/ob-analytics"
EPIC = 124
OUT = "scratch-roadmap-graph.json"


def gh(path):
    """--slurp wraps each page in an array, so pages flatten cleanly."""
    proc = subprocess.run(
        ["gh", "api", "--paginate", "--slurp", path],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(f"FAILED {path}: {proc.stderr}", file=sys.stderr)
        return []
    return [item for page in json.loads(proc.stdout) for item in page]


def main():
    subs = gh(f"repos/{REPO}/issues/{EPIC}/sub_issues")
    nodes = {}
    for s in subs:
        n = s["number"]
        nodes[n] = {
            "number": n,
            "title": s["title"],
            "state": s["state"],
            "labels": [lbl["name"] for lbl in s["labels"]],
            "open_blockers": s.get("issue_dependencies_summary", {}).get(
                "blocked_by", 0
            ),
            "blocked_by": [],
        }

    for n in sorted(nodes):
        blockers = gh(f"repos/{REPO}/issues/{n}/dependencies/blocked_by")
        if blockers:
            nodes[n]["blocked_by"] = [
                {"number": b["number"], "state": b["state"]} for b in blockers
            ]
        print(f"  #{n}: {len(nodes[n]['blocked_by'])} blockers", file=sys.stderr)

    with open(OUT, "w") as f:
        json.dump({"epic": EPIC, "nodes": nodes}, f, indent=2)
    print(f"wrote {OUT}: {len(nodes)} nodes", file=sys.stderr)


if __name__ == "__main__":
    main()
