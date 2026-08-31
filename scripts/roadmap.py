"""Write the roadmap views in epic #124 and the sixteen goal issues.

GitHub's own issue graph is the only store of the roadmap. The sub-issues of
#124 are the nodes and their ``blocked_by`` links are the edges; this script
reads that graph and writes the views of it that people read. It runs one way
only: it never writes to the graph, and it never reads its configuration out of
an issue body, which would mean parsing config from the same surface it writes.

The same graph used to be written out by hand four times over, so closing one
issue took four correct edits and the copies drifted apart.

What it writes
--------------
Into #124: a status count, one diagram per group in ``roadmap-groups.toml``, a
list of the work issues no group names, and one line per goal with its
readiness. Into each goal issue: the list of what that goal waits on and a
diagram of just that goal. Everything lands between ``<!-- ROADMAP:BEGIN -->``
and ``<!-- ROADMAP:END -->``; the prose above the opening marker is written by
people and is never touched. Which issue belongs in which diagram is editorial
judgement and lives in the config; the edges never do.

The rules it applies
--------------------
**Pruning.** A closed issue is dropped from a diagram unless an open non-goal
issue still depends on it, so a diagram shows the work ahead rather than the
whole history. Goal edges must not count: sixteen goals depend on nearly
everything, so counting them keeps every closed issue and the rule does
nothing. A group marked ``keep_closed`` opts out, for a diagram that records a
finished chapter.

**Size.** A diagram runs about 79 px per node and hardly varies with the number
of edges, so node count is the only lever and each goal gets its own small
diagram instead of sharing a large one. Keep any generated diagram under about
12 nodes.

**Colour.** The fill is readiness — closed, open and ready, open and blocked,
or a goal with everything closed and nobody having confirmed it yet. A
foundational issue keeps its readiness fill and gains a heavy stroke. Mermaid's
``:::`` applies exactly one class, so that heavier form is emitted as its own
combined class rather than as a second class on the node.

**Either/or.** ``blocked_by`` has no OR, so a goal that needs one of two issues
has both wired as hard blockers, and the view would overstate what it needs and
count its blockers one too high — which can show a goal as blocked when it is
ready to check. The config declares those choices; each is drawn as one node
and is met as soon as any member closes.

**Writing.** Each body is compared before it is written, so a run triggered by
every issue event does not churn seventeen edit histories. A body whose markers
are missing or malformed is skipped and logged rather than guessed at.

Running it
----------
    uv run --no-project python scripts/roadmap.py --dry-run

Drop ``--dry-run`` to write. In CI the default ``GITHUB_TOKEN`` with
``issues: write`` is enough; see ``.github/workflows/roadmap.yml``.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

LOG = logging.getLogger("roadmap")

# The roadmap epic. Its children are the whole graph.
EPIC = 124


BEGIN = "<!-- ROADMAP:BEGIN -->"
END = "<!-- ROADMAP:END -->"


class MarkerError(Exception):
    """An issue body has no usable pair of roadmap markers."""


@dataclass(frozen=True)
class Node:
    """One issue in the roadmap graph."""

    number: int
    title: str
    state: str
    labels: tuple[str, ...]
    open_blockers: int
    blocked_by: tuple[int, ...]

    @property
    def is_goal(self) -> bool:
        return "goal" in self.labels

    @property
    def is_closed(self) -> bool:
        return self.state == "closed"


@dataclass(frozen=True)
class Graph:
    """Every child of the epic, keyed by issue number."""

    nodes: dict[int, Node]

    @property
    def work(self) -> list[Node]:
        return [n for n in self._ordered if not n.is_goal]

    @property
    def goals(self) -> list[Node]:
        return [n for n in self._ordered if n.is_goal]

    @property
    def _ordered(self) -> list[Node]:
        return [self.nodes[k] for k in sorted(self.nodes)]


@dataclass(frozen=True)
class Group:
    """One work diagram: a set of nodes, never a set of edges."""

    id: str
    title: str
    prose: str
    issues: tuple[int, ...]
    keep_closed: bool = False


@dataclass(frozen=True)
class Alternative:
    """An either/or prerequisite of one goal.

    ``blocked_by`` has no OR, so both members are wired as hard blockers.  This
    says they are a choice: any one of them satisfies the goal.
    """

    goal: int
    any_of: tuple[int, ...]
    label: str


@dataclass(frozen=True)
class Prerequisite:
    """Something a goal waits on: one issue, or a choice between several."""

    members: tuple[int, ...]
    label: str | None = None

    @property
    def key(self) -> int:
        return min(self.members)


@dataclass(frozen=True)
class Config:
    """The editorial half: which issues share a diagram and what to call them."""

    groups: tuple[Group, ...]
    alternatives: tuple[Alternative, ...]
    labels: dict[int, str]


def load_graph(path: Path) -> Graph:
    """Read a graph snapshot written by :func:`fetch_graph`."""
    raw = json.loads(Path(path).read_text())["nodes"]
    nodes = {}
    for value in raw.values():
        nodes[value["number"]] = Node(
            number=value["number"],
            title=value["title"],
            state=value["state"],
            labels=tuple(value["labels"]),
            open_blockers=value["open_blockers"],
            blocked_by=tuple(b["number"] for b in value["blocked_by"]),
        )
    return Graph(nodes=nodes)


def load_config(path: Path) -> Config:
    """Read ``roadmap-groups.toml``."""
    raw = tomllib.loads(Path(path).read_text())
    return Config(
        groups=tuple(
            Group(
                id=g["id"],
                title=g["title"],
                prose=g["prose"].strip(),
                issues=tuple(g["issues"]),
                keep_closed=g.get("keep_closed", False),
            )
            for g in raw.get("group", ())
        ),
        alternatives=tuple(
            Alternative(goal=a["goal"], any_of=tuple(a["any_of"]), label=a["label"])
            for a in raw.get("alternative", ())
        ),
        labels={int(k): v for k, v in raw.get("labels", {}).items()},
    )


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

# Readiness fill, plus a heavy stroke for a foundational issue.  Mermaid's
# ":::" applies exactly one class, so "foundational" cannot be a second class
# on the node; each combined class repeats the fill and adds the stroke.
CLASSDEFS = """  classDef done fill:#1a7f37,color:#fff,stroke:#166534;
  classDef ready fill:#8250df,color:#fff,stroke:#6b21a8;
  classDef blocked fill:#57606a,color:#fff,stroke:#424a53;
  classDef checkable fill:#0969da,color:#fff,stroke:#0a3d91;
  classDef doneF fill:#1a7f37,color:#fff,stroke:#0b1f10,stroke-width:4px;
  classDef readyF fill:#8250df,color:#fff,stroke:#2b0b45,stroke-width:4px;
  classDef blockedF fill:#57606a,color:#fff,stroke:#1c2128,stroke-width:4px;"""


def _readiness(node: Node) -> str:
    """The mermaid class for a node: its readiness, heavy if foundational."""
    if node.is_closed:
        base = "done"
    elif node.open_blockers > 0:
        base = "blocked"
    elif node.is_goal:
        return "checkable"
    else:
        base = "ready"
    return base + "F" if "foundational" in node.labels else base


# How wide a derived label may run before it is cut, in characters.
LABEL_WIDTH = 24


def _cut(title: str) -> str:
    """The opening words of a title, up to :data:`LABEL_WIDTH` characters."""
    kept: list[str] = []
    used = 0
    for word in title.split():
        if kept and used + 1 + len(word) > LABEL_WIDTH:
            return " ".join(kept) + "…"
        used += (1 if kept else 0) + len(word)
        kept.append(word)
    return " ".join(kept)


def _short_label(config: Config, node: Node) -> str:
    """The name for a node's box.

    A short name is editorial: it is a human's abbreviation of a title, not
    something the title contains, so ``[labels]`` in the config wins.  Falling
    back to a cut title keeps a newly opened issue from turning a run red.
    """
    label = config.labels.get(node.number) or _cut(node.title)
    return label.replace('"', "'")


def _node_line(graph: Graph, config: Config, node: Node) -> str:
    """One mermaid node: a box for work, a rounded box for a goal."""
    if node.is_goal:
        label = node.title.removeprefix("Users can ").replace('"', "'")
        return f'  g{node.number}(["{label}"]):::{_goal_class(graph, config, node)}'
    label = _short_label(config, node)
    return f'  n{node.number}["#{node.number} {label}"]:::{_readiness(node)}'


def _alt_line(graph: Graph, prereq: Prerequisite) -> str:
    """One mermaid node standing for a choice between several issues."""
    names = " or ".join(f"#{m}" for m in prereq.members)
    if _satisfied(graph, prereq):
        cls = "done"
    elif any(
        not graph.nodes[m].is_closed and graph.nodes[m].open_blockers == 0
        for m in prereq.members
        if m in graph.nodes
    ):
        cls = "ready"
    else:
        cls = "blocked"
    return f'  alt{prereq.key}{{{{"{names} — {prereq.label}"}}}}:::{cls}'


def _render_mermaid(node_lines: list[str], edges: list[tuple[str, str]]) -> str:
    """Wrap node lines and edges in a ``graph LR`` block."""
    lines = ["```mermaid", "graph LR", CLASSDEFS, *node_lines]
    lines += [f"  {src} --> {dst}" for src, dst in edges]
    lines.append("```")
    return "\n".join(lines)


def _ref(graph: Graph, number: int) -> str:
    """The mermaid id of a node: ``g`` for a goal, ``n`` for work."""
    return f"g{number}" if graph.nodes[number].is_goal else f"n{number}"


def _open_dependents(graph: Graph, number: int) -> list[Node]:
    """Open **non-goal** issues waiting on this one.

    Goals are excluded on purpose: they depend on nearly everything, so
    counting them would keep every closed issue alive and make pruning a no-op.
    """
    return [n for n in graph.work if not n.is_closed and number in n.blocked_by]


def _members(graph: Graph, group: Group) -> list[int]:
    """The issues a group actually draws, after dropping stale closed work."""
    present = [i for i in sorted(group.issues) if i in graph.nodes]
    if group.keep_closed:
        return present
    return [
        i for i in present if not graph.nodes[i].is_closed or _open_dependents(graph, i)
    ]


def _edges_between(graph: Graph, members: list[int]) -> list[tuple[str, str]]:
    """Every ``blocked_by`` edge with both ends among the drawn nodes."""
    drawn = set(members)
    return [
        (_ref(graph, blocker), _ref(graph, m))
        for m in members
        for blocker in sorted(graph.nodes[m].blocked_by)
        if blocker in drawn
    ]


def _diagram(graph: Graph, config: Config, members: list[int]) -> str:
    """A mermaid ``graph LR`` over the given nodes and the edges between them."""
    return _render_mermaid(
        [_node_line(graph, config, graph.nodes[m]) for m in members],
        _edges_between(graph, members),
    )


def _prerequisites(graph: Graph, config: Config, goal: Node) -> list[Prerequisite]:
    """What a goal waits on, with declared alternatives collapsed into one.

    Every blocker the goal names, except that the members of an alternative
    become a single entry.  Blockers outside the graph are dropped: they cannot
    be drawn and their state is unknown here.
    """
    alternatives = [a for a in config.alternatives if a.goal == goal.number]
    grouped = {m for a in alternatives for m in a.any_of}
    out = [
        Prerequisite(members=tuple(sorted(a.any_of)), label=a.label)
        for a in alternatives
    ]
    out += [
        Prerequisite(members=(b,))
        for b in goal.blocked_by
        if b not in grouped and b in graph.nodes
    ]
    return sorted(out, key=lambda p: p.key)


def _satisfied(graph: Graph, prereq: Prerequisite) -> bool:
    """A prerequisite is met once **any** of its members is closed."""
    return any(graph.nodes[m].is_closed for m in prereq.members if m in graph.nodes)


def _outstanding(graph: Graph, config: Config, goal: Node) -> list[Prerequisite]:
    """The goal's prerequisites that are not met yet."""
    return [p for p in _prerequisites(graph, config, goal) if not _satisfied(graph, p)]


def _goal_class(graph: Graph, config: Config, goal: Node) -> str:
    """A goal's colour, counting a declared choice as one prerequisite."""
    if goal.is_closed:
        return "done"
    return "blocked" if _outstanding(graph, config, goal) else "checkable"


def _ungrouped(graph: Graph, config: Config) -> list[Node]:
    """Work issues no group names.

    Read from the config, never from what survived pruning: a closed issue
    dropped from its diagram is still grouped.  Goals are not groups either —
    every work issue is a prerequisite of some goal, so counting those would
    leave this list permanently empty.
    """
    named = {i for group in config.groups for i in group.issues}
    return [n for n in graph.work if n.number not in named]


def _goal_summary(graph: Graph, config: Config, goal: Node) -> str:
    """One line naming a goal, its state and how much of it is outstanding."""
    box = "x" if goal.is_closed else " "
    prereqs = _prerequisites(graph, config, goal)
    outstanding = _outstanding(graph, config, goal)
    if goal.is_closed:
        state = "done"
    elif not outstanding:
        state = "**ready to check** — every prerequisite is closed"
    else:
        state = (
            f"blocked, waiting on {len(outstanding)} of {len(prereqs)} prerequisites"
        )
    return f"- [{box}] #{goal.number} {goal.title} — {state}"


# ---------------------------------------------------------------------------
# The generated block for epic #124
# ---------------------------------------------------------------------------


def render_epic_body(graph: Graph, config: Config) -> str:
    """Render the generated block for epic #124."""
    work = graph.work
    closed = [n for n in work if n.is_closed]
    ready = [n for n in work if not n.is_closed and n.open_blockers == 0]
    blocked = [n for n in work if not n.is_closed and n.open_blockers > 0]

    out = [
        "## Where the work stands",
        "",
        (
            f"{len(closed)} of {len(work)} work issues closed. "
            f"{len(ready)} are ready to start now; "
            f"{len(blocked)} are waiting on something."
        ),
        "",
    ]

    for group in config.groups:
        members = _members(graph, group)
        if not members:
            continue
        out += [
            f"### {group.title}",
            "",
            group.prose,
            "",
            _diagram(graph, config, members),
            "",
        ]

    out += ["## What each capability waits on", ""]
    out += [_goal_summary(graph, config, goal) for goal in graph.goals]
    out.append("")

    ungrouped = _ungrouped(graph, config)
    if ungrouped:
        out += [
            "## Ungrouped",
            "",
            (
                "In no diagram above. Add them to `roadmap-groups.toml`, or "
                "leave them here."
            ),
            "",
        ]
        out += [
            f"- [{'x' if n.is_closed else ' '}] #{n.number} {n.title}"
            for n in ungrouped
        ]
        out.append("")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# The generated block for a goal issue
# ---------------------------------------------------------------------------


def _prereq_line(graph: Graph, prereq: Prerequisite) -> str:
    """One checklist row: the issue and its title, or the choice and its name."""
    box = "x" if _satisfied(graph, prereq) else " "
    if prereq.label:
        names = " or ".join(f"#{m}" for m in prereq.members)
        return f"- [{box}] {names} — {prereq.label}"
    node = graph.nodes[prereq.members[0]]
    return f"- [{box}] #{node.number} — {node.title}"


def render_goal_body(graph: Graph, config: Config, number: int) -> str:
    """Render the generated block for one goal issue."""
    goal = graph.nodes[number]
    prereqs = _prerequisites(graph, config, goal)

    out = ["## What it needs", ""]
    if not prereqs:
        out += ["Nothing outstanding.", ""]
        return "\n".join(out)

    out += [_prereq_line(graph, p) for p in prereqs]
    out.append("")

    # The list is complete; the diagram drops closed work nothing open still
    # waits on, so it shows what is left plus whatever finished work still
    # matters to it.
    drawn = [
        p
        for p in prereqs
        if not _satisfied(graph, p)
        or (not p.label and _open_dependents(graph, p.members[0]))
    ]
    members = [p.members[0] for p in drawn if not p.label]
    node_lines = [_node_line(graph, config, goal)]
    node_lines += [
        _alt_line(graph, p)
        if p.label
        else _node_line(graph, config, graph.nodes[p.members[0]])
        for p in drawn
    ]
    edges = _edges_between(graph, members)
    edges += [
        (f"alt{p.key}" if p.label else _ref(graph, p.members[0]), f"g{number}")
        for p in drawn
    ]
    out += [_render_mermaid(node_lines, edges), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Writing a block into an issue body
# ---------------------------------------------------------------------------


def splice_generated_block(body: str, block: str) -> str:
    """Return ``body`` with the text between the markers replaced by ``block``.

    Everything outside the markers is left exactly as it was, which is what
    keeps the hand-written prose in #124 and in each goal issue safe.  A body
    whose markers are missing, unpaired or the wrong way round raises
    :class:`MarkerError`; the caller skips that issue and logs it, because
    guessing where the block belongs would overwrite someone's writing.
    """
    start = body.find(BEGIN)
    end = body.find(END)
    if start < 0 or end < 0 or end < start:
        raise MarkerError(
            f"need {BEGIN} then {END}; found begin at {start}, end at {end}"
        )
    head = body[:start]
    tail = body[end + len(END) :]
    return f"{head}{BEGIN}\n{block.strip()}\n{END}{tail}"


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


class Issues(Protocol):
    """The slice of the GitHub issues API the generator uses."""

    def sub_issues(self, epic: int) -> list[dict]: ...

    def blocked_by(self, number: int) -> list[dict]: ...

    def get_body(self, number: int) -> str: ...

    def update_body(self, number: int, body: str) -> None: ...


@dataclass
class Report:
    """What one run did, for the log and for the tests."""

    written: list[int] = field(default_factory=list)
    unchanged: list[int] = field(default_factory=list)
    skipped: list[int] = field(default_factory=list)


def build_graph(client: Issues, epic: int) -> Graph:
    """Read the epic's children and the edges between them.

    Edges come from ``blocked_by`` per issue, not from the sub-issue payload:
    the payload's ``issue_dependencies_summary.blocked_by`` counts open
    blockers only, which is what readiness needs and useless for drawing, since
    a diagram has to show the closed ones too.
    """
    nodes = {}
    for payload in client.sub_issues(epic):
        number = payload["number"]
        nodes[number] = Node(
            number=number,
            title=payload["title"],
            state=payload["state"],
            labels=tuple(label["name"] for label in payload["labels"]),
            open_blockers=payload.get("issue_dependencies_summary", {}).get(
                "blocked_by", 0
            ),
            blocked_by=(),
        )
    for number, node in nodes.items():
        edges = tuple(
            b["number"] for b in client.blocked_by(number) if b["number"] in nodes
        )
        nodes[number] = replace(node, blocked_by=edges)
    return Graph(nodes=nodes)


def _write(client: Issues, number: int, block: str, report: Report) -> None:
    """Write one generated block, unless the body already says the same thing."""
    try:
        updated = splice_generated_block(client.get_body(number), block)
    except MarkerError as exc:
        LOG.warning("skipping #%s: %s", number, exc)
        report.skipped.append(number)
        return
    if updated == client.get_body(number):
        report.unchanged.append(number)
        return
    client.update_body(number, updated)
    report.written.append(number)


def run(client: Issues, config: Config, epic: int = EPIC) -> Report:
    """Read the graph and write the generated block into the epic and goals."""
    graph = build_graph(client, epic)
    report = Report()
    _write(client, epic, render_epic_body(graph, config), report)
    for goal in graph.goals:
        _write(
            client, goal.number, render_goal_body(graph, config, goal.number), report
        )
    return report


# ---------------------------------------------------------------------------
# Talking to GitHub, and the command line
# ---------------------------------------------------------------------------


class GhClient:
    """The GitHub issues API, over the ``gh`` CLI.

    Deliberately thin: it shells out, parses JSON, and holds no roadmap logic.
    The workflow's default ``GITHUB_TOKEN`` with ``issues: write`` is enough for
    every call here, and staying on that token is a requirement rather than a
    default — GitHub does not re-trigger workflows from its own events, which is
    the only reason this needs no loop guard.
    """

    def __init__(self, repo: str):
        self.repo = repo

    def _api(self, path: str) -> list[dict]:
        """A paginated GET, flattened.

        ``--slurp`` wraps each page in its own array so the pages can be
        flattened here.  Stitching raw ``--paginate`` output by scanning for
        brackets breaks silently, because issue titles contain ``[`` and ``]``.
        """
        out = subprocess.run(
            ["gh", "api", "--paginate", "--slurp", path],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return [item for page in json.loads(out) for item in page]

    def sub_issues(self, epic: int) -> list[dict]:
        return self._api(f"repos/{self.repo}/issues/{epic}/sub_issues")

    def blocked_by(self, number: int) -> list[dict]:
        return self._api(f"repos/{self.repo}/issues/{number}/dependencies/blocked_by")

    def get_body(self, number: int) -> str:
        out = subprocess.run(
            ["gh", "api", f"repos/{self.repo}/issues/{number}", "--jq", ".body"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return out.rstrip("\n")

    def update_body(self, number: int, body: str) -> None:
        subprocess.run(
            [
                "gh",
                "api",
                "-X",
                "PATCH",
                f"repos/{self.repo}/issues/{number}",
                "--input",
                "-",
            ],
            input=json.dumps({"body": body}),
            capture_output=True,
            text=True,
            check=True,
        )


class ReadOnly:
    """A client that reads for real and logs what it would have written."""

    def __init__(self, inner: Issues):
        self._inner = inner

    def sub_issues(self, epic: int) -> list[dict]:
        return self._inner.sub_issues(epic)

    def blocked_by(self, number: int) -> list[dict]:
        return self._inner.blocked_by(number)

    def get_body(self, number: int) -> str:
        return self._inner.get_body(number)

    def update_body(self, number: int, body: str) -> None:
        LOG.info("would update #%s (%s lines)", number, len(body.splitlines()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default="mczielinski/ob-analytics")
    parser.add_argument("--epic", type=int, default=EPIC)
    parser.add_argument("--config", type=Path, default=Path("roadmap-groups.toml"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="read the graph and render, but write nothing back",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    client: Issues = GhClient(args.repo)
    if args.dry_run:
        client = ReadOnly(client)

    report = run(client, load_config(args.config), epic=args.epic)
    LOG.info(
        "%d written, %d unchanged, %d skipped",
        len(report.written),
        len(report.unchanged),
        len(report.skipped),
    )
    for number in report.written:
        LOG.info("  updated #%s", number)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
