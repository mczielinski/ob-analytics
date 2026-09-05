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
Into #124: a status count, what to pick up next, one diagram per group in
``roadmap-groups.toml``, a list of the work issues no group names, one line per
goal with its readiness, and any blocker that points outside the epic. Into each
goal issue: the list of what that goal waits on and a diagram of just that goal.
Everything lands between ``<!-- ROADMAP:BEGIN -->`` and ``<!-- ROADMAP:END -->``;
the prose above the opening marker is written by people and is never touched.
Which issue belongs in which diagram is editorial judgement and lives in the
config; the edges never do.

The rules it applies
--------------------
**Prose.** Nothing written by hand may say where the work stands, because that
is the one thing that changes without anyone touching the words. So the epic's
own advice on what to do next is derived here — which goal is one issue from
done, which issue frees other work, how much is free to take in any order — and
the hand-written half above the marker is checked for issue numbers, each of
which is a claim about another issue that the graph can quietly outrun. A
caption is checked the same way against its own diagram: a caption naming a node
the pruner has dropped is the same fault. Both fail the run. The single
judgement the graph cannot make, that ready work should still wait, is a
``[[hold]]`` in the config, listed only while its issues are open.

**Pruning.** A closed issue is dropped from a diagram unless an open non-goal
issue still depends on it, so a diagram shows the work ahead rather than the
whole history. Goal edges must not count: sixteen goals depend on nearly
everything, so counting them keeps every closed issue and the rule does
nothing. A group marked ``keep_closed`` opts out, for a diagram that records
completed work.

**Size.** A diagram runs about 79 px per node and hardly varies with the number
of edges, so node count is the only lever and each goal gets its own small
diagram instead of sharing a large one. Keep any generated diagram under about
12 nodes.

**Colour.** The fill is readiness — closed, open and ready, open and blocked,
or a goal with everything closed and nobody having confirmed it yet. Closed is
purple and ready is green, which is what those two colours mean on GitHub
itself; the diagrams sit a few lines under GitHub's own state chips, so
reversing them made the roadmap contradict the page around it. A foundational
issue keeps its readiness fill and gains a heavy stroke. Mermaid's ``:::``
applies exactly one class, so that heavier form is emitted as its own combined
class rather than as a second class on the node.

**Either/or.** ``blocked_by`` has no OR, so a goal that needs one of two issues
has both wired as hard blockers, and the view would overstate what it needs and
count its blockers one too high — which can show a goal as blocked when it is
ready to check. The config declares those choices; each is drawn as one node
and is met as soon as any member closes.

**Edges off the roadmap.** A ``blocked_by`` link whose blocker is not a child
of #124 cannot be drawn, so it is left out. It is named in the run log and in
#124 rather than dropped in silence, and the blocker is never pulled into the
node set to save it: what belongs on the roadmap stays a deliberate choice.

**Writing.** Each body is compared before it is written, so a run triggered by
every issue event does not churn seventeen edit histories. A body whose markers
are missing or malformed is skipped and logged rather than guessed at, and a run
that skipped anything exits non-zero, because a skipped issue is a stale view.

Running it
----------
    uv run --no-project python scripts/roadmap.py --dry-run

Drop ``--dry-run`` to write. ``.github/workflows/roadmap.yml`` runs it on issue
events, once a week and on demand; the default ``GITHUB_TOKEN`` with
``issues: write`` is enough for every call it makes.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
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
    """Every child of the epic, keyed by issue number.

    ``dropped_edges`` holds the ``blocked_by`` links the graph could not keep,
    as (issue, missing blocker) pairs: the blocker is not a child of the epic,
    so no view can draw it or say what state it is in.  They are kept here so
    a run can name them instead of losing them.
    """

    nodes: dict[int, Node]
    dropped_edges: tuple[tuple[int, int], ...] = ()

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
class Hold:
    """Work that is ready but should wait, and the reason it waits.

    The graph cannot hold an opinion. "Do this only once a measurement asks for
    it" is a judgement about work nothing blocks, and it used to be written into
    the epic by hand, where it outlived the issues it was about. Here it is
    attached to the issues themselves, so it is listed only while they are open
    and disappears when they close.
    """

    issues: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class Config:
    """The editorial half: which issues share a diagram and what to call them."""

    groups: tuple[Group, ...]
    alternatives: tuple[Alternative, ...]
    labels: dict[int, str]
    holds: tuple[Hold, ...] = ()


def _split_edges(
    number: int, blockers: list[int], nodes: dict[int, Node]
) -> tuple[tuple[int, ...], list[tuple[int, int]]]:
    """Split one issue's blockers into the ones in the node set and the rest.

    A blocker that is not a child of the epic cannot go in the graph: nothing
    here knows its title or its state, so no list and no diagram can show it.
    It is returned rather than thrown away, because a real blocker leaving a
    goal's list without a word makes that goal look easier than it is, while
    GitHub's own dependency panel still names it on the same page.
    """
    kept = tuple(b for b in blockers if b in nodes)
    dropped = [(number, b) for b in blockers if b not in nodes]
    return kept, dropped


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
            blocked_by=(),
        )
    dropped: list[tuple[int, int]] = []
    for value in raw.values():
        number = value["number"]
        kept, missing = _split_edges(
            number, [b["number"] for b in value["blocked_by"]], nodes
        )
        nodes[number] = replace(nodes[number], blocked_by=kept)
        dropped += missing
    return Graph(nodes=nodes, dropped_edges=tuple(sorted(dropped)))


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
        holds=tuple(
            Hold(issues=tuple(h["issues"]), reason=h["reason"])
            for h in raw.get("hold", ())
        ),
    )


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

# Readiness fill, plus a heavy stroke for a foundational issue.  Mermaid's
# ":::" applies exactly one class, so "foundational" cannot be a second class
# on the node; each combined class repeats the fill and adds the stroke.
# The key to those fills. It is generated with the diagrams because it belongs
# to them: a hand-written key drifts from the colours the moment they change.
COLOUR_KEY = (
    "Colour: purple = done, green = ready to start, "
    "grey = waiting on something, blue = a goal whose prerequisites are all "
    "closed and which nobody has confirmed yet. A heavy outline marks a "
    "foundational issue. An arrow points from a task to the work it unblocks."
)

# Green and purple are GitHub's own state colours, and these diagrams render
# inside issue bodies a few lines under GitHub's own chips: green on an open
# issue, purple on one closed as completed. So they have to carry the meanings
# GitHub gives them. Do not swap them back.
CLASSDEFS = """  classDef done fill:#8250df,color:#fff,stroke:#6b21a8;
  classDef ready fill:#1a7f37,color:#fff,stroke:#166534;
  classDef blocked fill:#57606a,color:#fff,stroke:#424a53;
  classDef checkable fill:#0969da,color:#fff,stroke:#0a3d91;
  classDef doneF fill:#8250df,color:#fff,stroke:#2b0b45,stroke-width:4px;
  classDef readyF fill:#1a7f37,color:#fff,stroke:#0b1f10,stroke-width:4px;
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
# What to pick up next
# ---------------------------------------------------------------------------


def _held(config: Config, number: int) -> str | None:
    """The reason this issue is held back, if it is."""
    for hold in config.holds:
        if number in hold.issues:
            return hold.reason
    return None


def _name(graph: Graph, config: Config, number: int) -> str:
    """``#123 short label``, the way every list here names an issue."""
    return f"#{number} {_short_label(config, graph.nodes[number])}"


def _goal_name(node: Node) -> str:
    """A goal named as the thing a user can do, without the stock opening."""
    return f"#{node.number} ({node.title.removeprefix('Users can ')})"


def _prereq_name(graph: Graph, config: Config, prereq: Prerequisite) -> str:
    """One prerequisite named: an issue, or a choice between several."""
    if prereq.label:
        return f"{' or '.join(f'#{m}' for m in prereq.members)} ({prereq.label})"
    return _name(graph, config, prereq.members[0])


def _one_away(graph: Graph, config: Config) -> list[tuple[str, tuple[int, ...]]]:
    """Goals with a single prerequisite left, and the issue that would close it.

    This is the strongest thing the graph can say about what to do next: every
    other issue moves a goal along, and these finish one.
    """
    rows = []
    for goal in graph.goals:
        if goal.is_closed:
            continue
        outstanding = _outstanding(graph, config, goal)
        if len(outstanding) != 1:
            continue
        prereq = outstanding[0]
        held = any(_held(config, m) for m in prereq.members)
        rows.append(
            (
                f"- {_prereq_name(graph, config, prereq)} — the last thing "
                f"{_goal_name(goal)} waits on"
                + (" (held back, see below)" if held else ""),
                prereq.members,
            )
        )
    return rows


def _frees_work(graph: Graph, config: Config) -> list[tuple[str, tuple[int, ...]]]:
    """Ready issues that something already open is waiting on.

    Sorted by how much each one frees, because that is the only ordering the
    graph justifies: an issue two others wait on unblocks more than one nobody
    waits on.
    """
    rows = []
    for node in graph.work:
        if node.is_closed or node.open_blockers:
            continue
        waiting = _open_dependents(graph, node.number)
        if not waiting:
            continue
        rows.append((len(waiting), node.number, waiting))
    out = []
    for _, number, waiting in sorted(rows, key=lambda r: (-r[0], r[1])):
        names = " and ".join(f"#{n.number}" for n in waiting)
        verb = "waits" if len(waiting) == 1 else "wait"
        held = " (held back, see below)" if _held(config, number) else ""
        out.append(
            (
                f"- {_name(graph, config, number)} — {names} {verb} on it{held}",
                (number,),
            )
        )
    return out


def _holds(graph: Graph, config: Config) -> list[str]:
    """Every hold that still has an open issue under it."""
    lines = []
    for hold in config.holds:
        open_issues = [
            i for i in hold.issues if i in graph.nodes and not graph.nodes[i].is_closed
        ]
        if not open_issues:
            continue
        names = ", ".join(_name(graph, config, i) for i in open_issues)
        lines.append(f"- {names} — {hold.reason}")
    return lines


def _free_choice(graph: Graph, config: Config, named: set[int]) -> int:
    """How many ready issues are left once the lists above have had their say.

    Named issues are subtracted so the four groups partition the open work
    instead of counting the same issue twice, which is the arithmetic a reader
    checks first.
    """
    return sum(
        1
        for n in graph.work
        if not n.is_closed
        and not n.open_blockers
        and n.number not in named
        and not _open_dependents(graph, n.number)
        and not _held(config, n.number)
    )


def render_next_up(graph: Graph, config: Config) -> list[str]:
    """The section that says what to pick up, worked out from the graph.

    #124 used to answer this in hand-written prose, which meant every close
    re-dated a paragraph nobody remembered to edit. Everything here is derived:
    which goal is one issue from done, which issue frees other work, and how
    much is free to take in any order. The one thing the graph cannot know —
    that a piece of ready work should still wait — comes from the config's
    holds, which expire with the issues they name.
    """
    out = [
        "## What to pick up next",
        "",
        "Worked out from the graph on each run, so none of it needs an edit.",
        "",
    ]
    named: set[int] = set()
    sections = [
        (
            (
                "**One issue away from a goal.** Closing any of these finishes "
                "something a user can do."
            ),
            _one_away(graph, config),
        ),
        ("**Frees other work.**", _frees_work(graph, config)),
    ]
    for heading, rows in sections:
        if not rows:
            continue
        out += [heading, "", *[line for line, _ in rows], ""]
        named |= {number for _, members in rows for number in members}

    holds = _holds(graph, config)
    if holds:
        out += ["**Held back on purpose.**", "", *holds, ""]

    free = _free_choice(graph, config, named)
    if free:
        out += [
            (
                f"**Free to take in any order.** {free} other issues are open "
                "with nothing in their way and nothing waiting on them, so the "
                "order is yours."
            ),
            "",
        ]
    return out


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
        COLOUR_KEY,
        "",
    ]

    out += render_next_up(graph, config)

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

    if graph.dropped_edges:
        out += [
            "## Blockers outside the roadmap",
            "",
            (
                "These issues wait on issues that are not in this epic, so "
                "the link is missing from the lists and diagrams above. To "
                "fix one, add the blocker to this epic, or remove the "
                "dependency."
            ),
            "",
        ]
        out += [
            f"- #{number} is blocked by #{blocker}"
            for number, blocker in graph.dropped_edges
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
# Guards against prose that goes stale
# ---------------------------------------------------------------------------

# An issue number in the hand-written half of a body. Written as a pattern
# rather than a search for "#" so that a heading, a hex colour or a plain hash
# is not mistaken for a reference.
MENTION = re.compile(r"#(\d+)")


def _hand_written(body: str) -> str:
    """Everything in a body the generator does not write.

    Both sides of the block count: #124 carries prose after the closing marker
    as well as before it, and prose there goes stale exactly as fast.
    """
    head, _, rest = body.partition(BEGIN)
    _, _, tail = rest.partition(END)
    return head + tail


def named_issues(body: str) -> list[int]:
    """Every issue number in the hand-written parts of a body."""
    return [int(m) for m in MENTION.findall(_hand_written(body))]


def stale_mentions(body: str, tracked: set[int]) -> list[int]:
    """Roadmap issues named in prose the generator does not rewrite.

    Each is a claim about another issue — what is done, what is next, what
    waits on what — that the graph moves under without a word. That is how the
    epic came to carry sentences describing boxes no reader could see. The
    generated block names every issue a reader needs, with its state as it is
    now, so hand-written prose says why the work is shaped this way and never
    where it stands.

    Only issues the roadmap tracks can go stale this way, so only they are
    flagged. A goal's footer says which epic it belongs to and which discussion
    decided it: neither is a claim about progress, and neither can be
    contradicted by a graph that does not contain it.
    """
    return [n for n in named_issues(body) if n in tracked]


def stale_captions(graph: Graph, config: Config) -> list[str]:
    """Group captions that name an issue their diagram no longer draws.

    A closed issue nothing waits on is pruned from its diagram, and a caption
    written when it was there goes on describing a box that is gone. The short
    labels are exactly the words a caption uses for a node, so a label that
    appears in a caption whose node is not drawn is that mistake, found without
    anyone re-reading the epic.
    """
    out = []
    for group in config.groups:
        drawn = set(_members(graph, group))
        for number in group.issues:
            if number in drawn or number not in graph.nodes:
                continue
            label = config.labels.get(number)
            if not label:
                continue
            if re.search(rf"\b{re.escape(label)}\b", group.prose, re.IGNORECASE):
                out.append(
                    f"{group.id}: the caption says {label!r}, but #{number} is "
                    "not drawn in that diagram any more"
                )
    return out


def unknown_holds(config: Config) -> list[str]:
    """Holds that name an issue no group draws.

    Checked against the config rather than the graph, so a mistyped number is
    caught the same way whatever state the roadmap is in. A hold on an issue
    the roadmap never draws would otherwise sit in the file printing nothing,
    which is the silent kind of wrong this file exists to avoid.
    """
    drawn = {i for group in config.groups for i in group.issues}
    return [
        f"a hold names #{number}, which no diagram draws"
        for hold in config.holds
        for number in hold.issues
        if number not in drawn
    ]


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
    dropped_edges: list[tuple[int, int]] = field(default_factory=list)
    stale_prose: list[str] = field(default_factory=list)


def build_graph(client: Issues, epic: int) -> Graph:
    """Read the epic's children and the edges between them.

    Edges come from ``blocked_by`` per issue, not from the sub-issue payload:
    the payload's ``issue_dependencies_summary.blocked_by`` counts open
    blockers only, which is what readiness needs and useless for drawing, since
    a diagram has to show the closed ones too.

    An edge whose blocker is not a child of the epic is left out and recorded
    in :attr:`Graph.dropped_edges`, for the run to report.  Nothing is added to
    the node set to save such an edge: membership of #124 is a deliberate act,
    and it is what keeps one-off issues and dependabot pull requests out of the
    roadmap.
    """
    nodes = {}
    for payload in client.sub_issues(epic):
        number = payload["number"]
        nodes[number] = Node(
            number=number,
            title=payload["title"],
            state=payload["state"],
            labels=tuple(label["name"] for label in payload["labels"]),
            # This count is unfiltered: it includes open blockers whose edges
            # are dropped below, so an issue can draw as blocked with no
            # incoming arrow. Whether readiness should use the filtered count
            # instead is a separate decision.
            open_blockers=payload.get("issue_dependencies_summary", {}).get(
                "blocked_by", 0
            ),
            blocked_by=(),
        )
    dropped: list[tuple[int, int]] = []
    for number, node in nodes.items():
        kept, missing = _split_edges(
            number, [b["number"] for b in client.blocked_by(number)], nodes
        )
        nodes[number] = replace(node, blocked_by=kept)
        dropped += missing
    return Graph(nodes=nodes, dropped_edges=tuple(sorted(dropped)))


def _write(
    client: Issues, number: int, block: str, report: Report, tracked: set[int]
) -> None:
    """Write one generated block, unless the body already says the same thing.

    A run that changes nothing reads each body **once**: comparing before
    writing is what keeps a run on every issue event from churning seventeen
    edit histories, and one read halves the API calls, 17 rather than 34.

    A run that does have something to write reads that one body again, and
    splices into the second copy. The gap between reading a body and writing it
    is a gap someone can edit the prose in, and the runs overlap in exactly the
    way that makes it likely: an event arrives while a run is in flight, and
    the write that follows carries whatever the prose was when the run started.
    That is not theoretical — it silently reverted a rewrite of #124's prose
    while this was being built. GitHub has no conditional write for an issue
    body, so the gap cannot be closed altogether; splicing into the freshest
    copy narrows it from a whole run to a single request.
    """
    current = client.get_body(number)
    for named in stale_mentions(current, tracked):
        report.stale_prose.append(
            f"#{number} names #{named} outside the generated block, where "
            "nothing keeps a claim about a roadmap issue up to date"
        )
    try:
        updated = splice_generated_block(current, block)
    except MarkerError as exc:
        LOG.warning("skipping #%s: %s", number, exc)
        report.skipped.append(number)
        return
    if updated == current:
        report.unchanged.append(number)
        return
    fresh = client.get_body(number)
    try:
        updated = splice_generated_block(fresh, block)
    except MarkerError as exc:
        LOG.warning("skipping #%s: %s", number, exc)
        report.skipped.append(number)
        return
    if updated == fresh:
        report.unchanged.append(number)
        return
    client.update_body(number, updated)
    report.written.append(number)


def run(client: Issues, config: Config, epic: int = EPIC) -> Report:
    """Read the graph and write the generated block into the epic and goals."""
    graph = build_graph(client, epic)
    report = Report(dropped_edges=list(graph.dropped_edges))
    for number, blocker in report.dropped_edges:
        LOG.warning(
            "#%s is blocked by #%s, which is not a child of #%s, "
            "so the link is left out of every view",
            number,
            blocker,
            epic,
        )
    report.stale_prose += stale_captions(graph, config)
    report.stale_prose += unknown_holds(config)
    # The epic is the one number prose may name: a goal belongs to it whatever
    # the graph does, and the generator would not be running without it.
    tracked = set(graph.nodes) - {epic}
    _write(client, epic, render_epic_body(graph, config), report, tracked)
    for goal in graph.goals:
        _write(
            client,
            goal.number,
            render_goal_body(graph, config, goal.number),
            report,
            tracked,
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


def exit_code(report: Report) -> int:
    """Fail the run if it skipped an issue, or found prose that will go stale.

    A skip is never harmless. The generator carries on past a body whose
    markers are missing or malformed, so that one broken body cannot stop the
    other sixteen, but that issue now shows whatever the graph said the last
    time anyone could write to it. Returning success there would report a
    healthy roadmap while a view had quietly stopped updating, which is the
    drift this exists to end. Writing nothing because nothing changed is the
    steady state and succeeds.

    Prose that names an issue fails for the same reason one step earlier: the
    view is right and the writing around it is not, which is harder to notice
    than a view that stopped moving. The run still writes everything it can, so
    the failure is a report, not a refusal.
    """
    return 1 if report.skipped or report.stale_prose else 0


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
    for number in report.skipped:
        LOG.error("  #%s has no usable markers, so its view is stale", number)
    for complaint in report.stale_prose:
        LOG.error("  %s", complaint)
    return exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
