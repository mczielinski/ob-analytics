"""PROTOTYPE - throwaway. Render #124's body from the live graph.

Not production code. Answers issue #171 by making the generated body concrete
enough to react to, with the open design choices exposed as flags:

    --colour   readiness | current   how nodes are coloured
    --closed   keep | prune          whether closed issues stay in a diagram
    --goals-count-as-dependents      do goal edges keep a closed node alive?
    --checklist on | off             keep the by-area checklist

    python3 scripts/prototype_roadmap_render.py --colour readiness --closed prune
"""

import argparse
import json
import tomllib

GRAPH = "scratch-roadmap-graph.json"
CONFIG = "scratch-roadmap-groups.toml"

# Short labels for diagram nodes; the real generator would derive these or read
# them from the config. Hand-written here because issue titles are far too long
# to put in a mermaid box.
SHORT = {
    98: "L2 loader",
    99: "Coinbase",
    100: "Databento",
    101: "Binance",
    102: "Kalshi",
    103: "Polymarket",
    104: "Polars + Narwhals",
    105: "live view",
    106: "CCXT",
    107: "trade signs",
    108: "audit command",
    109: "micro-price",
    110: "spread + impact",
    111: "iceberg detection",
    112: "shared schema",
    113: "export writer",
    114: "fake L3 data",
    115: "conda-forge",
    116: "chunked run",
    117: "replay scrubber",
    118: "plot catalogue",
    119: "flow toxicity",
    120: "docs warnings",
    121: "tutorial",
    122: "Plotly theme",
    123: "Bokeh backend",
    134: "cryptofeed",
    136: "separate engine",
    137: "source plug-in",
    138: "faster engine",
    139: "streaming core",
    140: "metric hook",
    141: "public API",
    142: "license check",
    143: "correctness tests",
    144: "speed test",
    145: "source packaging",
    146: "sequence numbers",
    147: "multi-symbol",
    148: "bars",
    149: "feature table",
    150: "robust capture",
    154: "UTC time model",
    155: "integer ticks",
}

AREA_ORDER = [
    ("area:venues", "Venues & ingestion"),
    ("area:ingestion", "Ingestion"),
    ("area:analytics", "Analytics"),
    ("area:perf", "Performance & scale"),
    ("area:interop", "Interop & distribution"),
    ("area:viz", "Visualization"),
    ("area:docs", "Docs"),
]

# Mermaid's ":::" applies exactly one class, so "foundational" is emitted as a
# combined class (fill + heavy stroke) rather than a second class on the node.
CLASSDEF_READINESS = """  classDef done fill:#1a7f37,color:#fff,stroke:#166534;
  classDef ready fill:#8250df,color:#fff,stroke:#6b21a8;
  classDef blocked fill:#57606a,color:#fff,stroke:#424a53;
  classDef checkable fill:#0969da,color:#fff,stroke:#0a3d91;
  classDef doneF fill:#1a7f37,color:#fff,stroke:#0b1f10,stroke-width:4px;
  classDef readyF fill:#8250df,color:#fff,stroke:#2b0b45,stroke-width:4px;
  classDef blockedF fill:#57606a,color:#fff,stroke:#1c2128,stroke-width:4px;"""

CLASSDEF_CURRENT = """  classDef done fill:#1a7f37,color:#fff,stroke:#166534;
  classDef found fill:#8250df,color:#fff,stroke:#6b21a8;
  classDef later fill:#9a6700,color:#fff,stroke:#7c5300;
  classDef goal fill:#0969da,color:#fff,stroke:#0a3d91;"""


def load():
    with open(GRAPH) as f:
        graph = json.load(f)["nodes"]
    nodes = {int(k): v for k, v in graph.items()}
    with open(CONFIG, "rb") as f:
        config = tomllib.load(f)["group"]
    return nodes, config


def is_goal(n):
    return "goal" in n["labels"]


def readiness(n):
    """done / ready / blocked, plus 'checkable' for a goal with no open blockers.

    A foundational issue gets the same fill with a heavy stroke: readiness is the
    fact, foundational is the judgement, and both belong on the node.
    """
    if n["state"] == "closed":
        base = "done"
    elif n["open_blockers"] > 0:
        base = "blocked"
    else:
        return (
            "checkable"
            if is_goal(n)
            else ("readyF" if "foundational" in n["labels"] else "ready")
        )
    return base + "F" if "foundational" in n["labels"] else base


def current_class(n):
    if n["state"] == "closed":
        return "done"
    if is_goal(n):
        return "goal"
    if "foundational" in n["labels"]:
        return "found"
    return "later" if n["number"] in (138, 139) else None


def label(nodes, num):
    n = nodes.get(num)
    if is_goal(n):
        return n["title"].replace("Users can ", "")
    tick = " OK" if n["state"] == "closed" else ""
    return f"#{num} {SHORT.get(num, n['title'][:24])}{tick}"


def open_dependents(nodes, num, count_goals):
    """Open issues that this one blocks."""
    out = []
    for m in nodes.values():
        if m["state"] != "open":
            continue
        if is_goal(m) and not count_goals:
            continue
        if any(b["number"] == num for b in m["blocked_by"]):
            out.append(m["number"])
    return out


def group_members(nodes, group, args):
    """The nodes a group draws, after expansion and pruning."""
    members = set(group["issues"])
    if group.get("kind") == "goals":
        # a goal group names goals; their prerequisites are pulled in
        for g in group["issues"]:
            members.update(b["number"] for b in nodes[g]["blocked_by"])
    members = {m for m in members if m in nodes}

    if args.closed == "prune" and not group.get("keep_closed"):
        kept = set()
        for m in members:
            n = nodes[m]
            if n["state"] == "open" or open_dependents(
                nodes, m, args.goals_count_as_dependents
            ):
                kept.add(m)
        members = kept
    return members


def diagram(nodes, group, args):
    members = group_members(nodes, group, args)
    if not members:
        return "", 0, 0
    lines = ["```mermaid", "graph LR"]
    lines.append(CLASSDEF_READINESS if args.colour == "readiness" else CLASSDEF_CURRENT)
    for m in sorted(members):
        n = nodes[m]
        cls = readiness(n) if args.colour == "readiness" else current_class(n)
        shape = (
            f'g{m}(["{label(nodes, m)}"])'
            if is_goal(n)
            else f'n{m}["{label(nodes, m)}"]'
        )
        lines.append(
            f"  {shape}{f':::{cls}' if cls else ''};".replace('";:::', '"]:::').rstrip(
                ";"
            )
            + ";"
        )
    edges = 0
    for m in sorted(members):
        for b in nodes[m]["blocked_by"]:
            if b["number"] in members:
                src = (
                    f"g{b['number']}"
                    if is_goal(nodes[b["number"]])
                    else f"n{b['number']}"
                )
                dst = f"g{m}" if is_goal(nodes[m]) else f"n{m}"
                lines.append(f"  {src} --> {dst}")
                edges += 1
    lines.append("```")
    return "\n".join(lines), len(members), edges


def checklist(nodes):
    out = ["## All issues, by area", ""]
    seen = set()
    for lbl, heading in AREA_ORDER:
        rows = [
            n
            for n in sorted(nodes.values(), key=lambda x: x["number"])
            if lbl in n["labels"] and not is_goal(n) and n["number"] not in seen
        ]
        if not rows:
            continue
        out.append(f"### {heading}")
        for n in rows:
            seen.add(n["number"])
            box = "x" if n["state"] == "closed" else " "
            flag = " *(foundational)*" if "foundational" in n["labels"] else ""
            out.append(f"- [{box}] #{n['number']} — {n['title']}{flag}")
        out.append("")
    rest = [
        n
        for n in sorted(nodes.values(), key=lambda x: x["number"])
        if not is_goal(n) and n["number"] not in seen
    ]
    if rest:
        out.append("### Unlabelled")
        for n in rest:
            out.append(
                f"- [{'x' if n['state'] == 'closed' else ' '}] #{n['number']} — {n['title']}"
            )
        out.append("")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--colour", choices=["readiness", "current"], default="readiness")
    p.add_argument("--closed", choices=["keep", "prune"], default="prune")
    p.add_argument("--goals-count-as-dependents", action="store_true")
    p.add_argument("--checklist", choices=["on", "off"], default="on")
    p.add_argument("--goals", choices=["grouped", "per-goal"], default="grouped")
    p.add_argument("--out", default="-")
    args = p.parse_args()

    nodes, config = load()
    marker = (
        "<!-- ROADMAP:BEGIN - generated from GitHub sub-issues and "
        "blocked_by. Do not edit between the markers. -->"
    )
    body = [marker, ""]

    ready = sorted(
        n["number"]
        for n in nodes.values()
        if not is_goal(n) and n["state"] == "open" and n["open_blockers"] == 0
    )
    blocked = sorted(
        n["number"] for n in nodes.values() if not is_goal(n) and n["open_blockers"] > 0
    )
    checkable = sorted(
        n["number"]
        for n in nodes.values()
        if is_goal(n) and n["state"] == "open" and n["open_blockers"] == 0
    )

    body.append("## Where the work stands\n")
    done = sum(1 for n in nodes.values() if not is_goal(n) and n["state"] == "closed")
    total = sum(1 for n in nodes.values() if not is_goal(n))
    body.append(
        f"{done} of {total} work issues closed. {len(ready)} are ready to "
        f"start now; {len(blocked)} are waiting on something.\n"
    )
    if checkable:
        body.append(
            "**Ready to check** — every prerequisite is closed, but nobody has "
            "confirmed the capability yet: "
            + ", ".join(f"#{g}" for g in checkable)
            + "\n"
        )

    stats = []
    for g in config:
        if args.goals == "per-goal" and g.get("kind") == "goals":
            continue
        d, nn, ee = diagram(nodes, g, args)
        if not d:
            continue
        stats.append((g["id"], nn, ee))
        body.append(f"### {g['title']}\n")
        body.append(g["prose"].strip() + "\n")
        body.append(d + "\n")

    if args.goals == "per-goal":
        body.append("## What each capability waits on\n")
        for gnum in sorted(n["number"] for n in nodes.values() if is_goal(n)):
            g = nodes[gnum]
            body.append(f"### {g['title']} (#{gnum})\n")
            if not g["blocked_by"]:
                body.append("Nothing outstanding.\n")
                continue
            fake = {"issues": [gnum], "kind": "goals", "id": f"goal{gnum}"}
            d, nn, ee = diagram(nodes, fake, args)
            stats.append((f"goal#{gnum}", nn, ee))
            body.append(d + "\n")

    # from the config, NOT from post-prune membership: pruning a closed node
    # out of its diagram must not make it look ungrouped.
    grouped = set()
    for g in config:
        grouped.update(g["issues"])
        if g.get("kind") == "goals":
            for gg in g["issues"]:
                grouped.update(b["number"] for b in nodes[gg]["blocked_by"])
    stray = sorted(
        n["number"]
        for n in nodes.values()
        if not is_goal(n) and n["number"] not in grouped
    )
    body.append("### Ungrouped\n")
    body.append(
        "In no diagram. Add them to `roadmap-groups.toml` or leave them here.\n"
    )
    for s in stray:
        n = nodes[s]
        body.append(f"- [{'x' if n['state'] == 'closed' else ' '}] #{s} — {n['title']}")
    body.append("")

    if args.checklist == "on":
        body.append(checklist(nodes))

    body.append("<!-- ROADMAP:END -->")
    text = "\n".join(body)

    if args.out == "-":
        print(text)
    else:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}  ({len(text.splitlines())} lines)")
        for gid, nn, ee in stats:
            print(f"  {gid:<16} {nn:>3} nodes {ee:>3} edges")
        print(f"  stray: {len(stray)}")


if __name__ == "__main__":
    main()
