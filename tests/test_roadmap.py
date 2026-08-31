"""Tests for the roadmap generator, ``scripts/roadmap.py``.

The generator reads GitHub's issue graph and writes the generated blocks in
epic #124 and the sixteen goal issues.  Tests run against
``tests/fixtures/roadmap_graph.json``, a snapshot of the real 60-node graph
taken on 2026-08-30, so nothing here touches the network.

Expected values come from issue #192 and from that snapshot, never from
re-running the generator's own arithmetic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts.roadmap import (
    MarkerError,
    exit_code,
    load_config,
    load_graph,
    render_epic_body,
    render_goal_body,
    run,
    splice_generated_block,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
GRAPH_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "roadmap_graph.json"


@pytest.fixture(scope="module")
def graph():
    """The real 60-node graph as it stood on 2026-08-30."""
    return load_graph(GRAPH_FIXTURE)


@pytest.fixture(scope="module")
def config():
    """The repo's own ``roadmap-groups.toml``."""
    return load_config(REPO_ROOT / "roadmap-groups.toml")


def test_epic_counts_the_work(graph, config):
    """The status line counts work issues, not the 16 goals.

    From the snapshot: 60 children = 44 work issues + 16 goals; 11 work issues
    are closed, 28 are open with no open blocker, and 5 are open behind one.
    """
    body = render_epic_body(graph, config)

    assert "11 of 44 work issues closed" in body
    assert "28 are ready to start now" in body
    assert "5 are waiting on something" in body


# ---------------------------------------------------------------------------
# Helpers: build a config from TOML text, and read a rendered diagram back.
# ---------------------------------------------------------------------------


def write_config(tmp_path: Path, toml_text: str):
    """Load a config from TOML written for one test."""
    path = tmp_path / "roadmap-groups.toml"
    path.write_text(toml_text)
    return load_config(path)


def work_nodes(body: str) -> set[int]:
    """Issue numbers drawn as work boxes anywhere in a rendered body."""
    return {int(m) for m in re.findall(r"^\s*n(\d+)\[", body, flags=re.MULTILINE)}


# Every closed issue in the snapshot, work issues only (#175 and #176 are
# closed goals and are not work).
CLOSED_WORK = [98, 106, 107, 109, 112, 114, 137, 143, 146, 154, 155]


def test_diagram_drops_closed_work_nothing_open_waits_on(graph, tmp_path):
    """#192: pruning keeps #106, #112, #114, #137, #143 and #146, and drops
    #98, #107, #109, #154 and #155.

    A closed issue stays only while an **open non-goal** issue still depends on
    it.  #154 and #155 survive only if goal edges count, and because 16 goals
    depend on nearly everything, counting them turns the rule into a no-op.
    """
    config = write_config(
        tmp_path,
        f"""
[[group]]
id = "closed"
title = "Every closed issue"
prose = "One diagram holding all of them."
issues = {CLOSED_WORK}
""",
    )

    drawn = work_nodes(render_epic_body(graph, config))

    assert drawn == {106, 112, 114, 137, 143, 146}


def test_keep_closed_group_prunes_nothing(graph, tmp_path):
    """``keep_closed`` opts a diagram out, for one recording finished work."""
    config = write_config(
        tmp_path,
        f"""
[[group]]
id = "closed"
title = "Every closed issue"
prose = "One diagram holding all of them."
issues = {CLOSED_WORK}
keep_closed = true
""",
    )

    drawn = work_nodes(render_epic_body(graph, config))

    assert drawn == set(CLOSED_WORK)


def test_arrow_runs_from_the_blocker_to_the_work_it_unblocks(graph, config):
    """#136 is ``blocked_by`` #112, so the arrow is ``n112 --> n136``.

    The arrow points at the work a task unblocks, which is the opposite of the
    way ``blocked_by`` reads.
    """
    body = render_epic_body(graph, config)

    assert "n112 --> n136" in body
    assert "n136 --> n112" not in body


def test_edge_needs_both_ends_in_the_diagram(graph, tmp_path):
    """#112 blocks both #146 and #155, but #155 is pruned out, so only the
    edge with both ends still drawn survives."""
    config = write_config(
        tmp_path,
        f"""
[[group]]
id = "closed"
title = "Every closed issue"
prose = "One diagram holding all of them."
issues = {CLOSED_WORK}
""",
    )

    body = render_epic_body(graph, config)

    assert "n112 --> n146" in body
    assert "n155" not in body


def test_node_colour_shows_readiness_and_marks_foundational(graph, config):
    """The fill is readiness; a foundational issue gets the same fill with a
    heavy stroke, as a combined class.

    From the snapshot: #112 is closed and foundational, #136 is open with no
    open blocker and foundational, #113 is open and unblocked, #140 is open
    behind a blocker.  Mermaid's ``:::`` takes exactly one class, so
    ``foundational`` cannot ride along as a second one.
    """
    body = render_epic_body(graph, config)

    assert ":::doneF" in body.split("n112[")[1].split("\n")[0]
    assert ":::readyF" in body.split("n136[")[1].split("\n")[0]
    assert ":::ready\n" in body.split("n113[")[1]
    assert ":::blocked\n" in body.split("n140[")[1]
    assert "classDef blockedF" in body and "stroke-width:4px" in body


def test_epic_summarises_every_goal(graph, config):
    """One line per goal with its readiness, and no goal diagram in #124.

    Each goal draws its own diagram in its own issue, so the epic carries only
    the summary.  From the snapshot: 16 goals, #175 and #176 closed, #177 open
    with 1 of its 4 prerequisites still open.
    """
    body = render_epic_body(graph, config)
    lines = [ln for ln in body.splitlines() if re.match(r"- \[[ x]\] #1[78]\d ", ln)]

    assert len(lines) == 16
    assert "- [x] #176 Users can add their own plot — done" in lines
    assert (
        "- [ ] #177 Users can export to backtesting engines — blocked, "
        "waiting on 1 of 4 prerequisites" in lines
    )


def test_either_or_prerequisite_counts_as_one(graph, config):
    """A declared alternative is one prerequisite, not two.

    #179 needs #102 **or** #103, and #173 needs #134 **or** #99, but
    ``blocked_by`` has no OR so both members are wired as hard blockers.  Left
    alone the count overstates what the goal needs, which can show a goal as
    blocked when it is ready to check.

    #179 has 3 raw blockers (#102, #103 open, #106 closed) and so 2 real ones,
    1 outstanding.  #173 has 8 raw blockers and so 7 real ones, 3 outstanding.
    """
    body = render_epic_body(graph, config)

    assert (
        "- [ ] #179 Users can analyze prediction markets — blocked, "
        "waiting on 1 of 2 prerequisites" in body
    )
    assert (
        "- [ ] #173 Users can see live data on screen — blocked, "
        "waiting on 3 of 7 prerequisites" in body
    )


def write_graph(tmp_path: Path, nodes: list[dict]):
    """Load a hand-built graph in the shape the fetcher writes."""
    path = tmp_path / "graph.json"
    path.write_text(
        json.dumps(
            {
                "epic": 124,
                "nodes": {
                    str(n["number"]): {
                        "number": n["number"],
                        "title": n["title"],
                        "state": n.get("state", "open"),
                        "labels": n.get("labels", []),
                        "open_blockers": n.get("open_blockers", 0),
                        "blocked_by": [
                            {"number": b, "state": "open"}
                            for b in n.get("blocked_by", [])
                        ],
                    }
                    for n in nodes
                },
            }
        )
    )
    return load_graph(path)


def test_goal_is_ready_to_check_once_either_side_closes(tmp_path):
    """A goal is ready to check when nothing is outstanding, even though the
    unused half of an either/or is still open.

    This is the state the goal view exists to get right and the snapshot has no
    instance of it, so it is built by hand: a goal needing #102 **or** #103,
    with #102 closed.  Counted as ALL the goal looks blocked; counted as a
    choice it is ready for someone to confirm.
    """
    graph = write_graph(
        tmp_path,
        [
            {"number": 102, "title": "Kalshi", "state": "closed"},
            {"number": 103, "title": "Polymarket", "state": "open"},
            {
                "number": 179,
                "title": "Users can analyze prediction markets",
                "labels": ["goal"],
                "open_blockers": 1,
                "blocked_by": [102, 103],
            },
        ],
    )
    config = write_config(
        tmp_path,
        """
[[alternative]]
goal = 179
any_of = [102, 103]
label = "a prediction market"
""",
    )

    body = render_epic_body(graph, config)

    assert (
        "- [ ] #179 Users can analyze prediction markets — **ready to check**" in body
    )


def test_goal_body_lists_prerequisites_and_draws_the_goal(graph, config):
    """A goal issue gets the list of what it needs and its own diagram.

    #173 names 8 blockers; #134 and #99 are a declared choice, so the list has
    7 entries and the diagram draws the choice as one node.  The diagram prunes
    closed work nothing open still waits on, so #154 goes and #112 stays.
    """
    body = render_goal_body(graph, config, 173)
    checkboxes = [ln for ln in body.splitlines() if ln.startswith("- [")]

    assert len(checkboxes) == 7
    assert "- [ ] #99 or #134 — a live source" in checkboxes
    assert (
        "- [x] #112 — Stabilize + document a canonical Parquet microstructure "
        "schema (Arrow interop)" in checkboxes
    )
    assert 'g173(["see live data on screen"])' in body
    assert "a live source" in body.split("```mermaid")[1]
    assert work_nodes(body) == {105, 112, 137, 139}


def test_goal_diagram_stays_small(graph, config):
    """Every generated diagram stays under 12 nodes.

    Height runs about 79 px per node and barely depends on edge count, so node
    count is the only lever, and per-goal diagrams are why the goals were split
    out of #124.
    """
    for goal in graph.goals:
        body = render_goal_body(graph, config, goal.number)
        drawn = re.findall(r"^\s{2}(?:n|g|alt)\d+[\[\(\{]", body, flags=re.MULTILINE)
        assert len(drawn) <= 12, f"#{goal.number} draws {len(drawn)} nodes"


def test_ungrouped_means_in_no_work_group(graph, tmp_path):
    """An issue in no work diagram is listed, and the run stays green.

    "Ungrouped" has to mean "in no **work** group".  Every work issue is a
    prerequisite of some goal — #109 feeds #183 — so counting goals as groups
    would empty this section and hide the issues it exists to surface.  The
    config here names #110 and not #109, and #109 is what gets reported.
    """
    config = write_config(
        tmp_path,
        """
[[group]]
id = "analytics"
title = "Measuring the book"
prose = "Part of it."
issues = [110]
""",
    )

    body = render_epic_body(graph, config)
    listed = [
        int(m) for m in re.findall(r"^- \[[ x]\] #(\d+) ", body, flags=re.MULTILINE)
    ]

    assert 109 in listed
    assert 110 not in listed
    # and the section comes last, after the goal summaries
    assert body.index("## What each capability waits on") < body.index("## Ungrouped")


def test_pruned_closed_issue_is_not_reported_as_ungrouped(graph, tmp_path):
    """Membership is read from the config, not from what survived pruning.

    #154 is closed and nothing open still waits on it, so a diagram naming it
    draws nothing.  It is still grouped, and reporting it as ungrouped would
    invite someone to add an issue that is already there.
    """
    config = write_config(
        tmp_path,
        """
[[group]]
id = "data-model"
title = "Data-model decisions"
prose = "Settled."
issues = [154, 155]
""",
    )

    body = render_epic_body(graph, config)
    listed = [
        int(m) for m in re.findall(r"^- \[[ x]\] #(\d+) ", body, flags=re.MULTILINE)
    ]

    assert "n154" not in body
    assert 154 not in listed


def test_unnamed_issue_gets_a_label_cut_from_its_title(tmp_path):
    """An issue with no entry under ``[labels]`` still draws.

    Opening an issue must never turn the workflow run red, so a missing short
    name is filled in from the title rather than raising.  The cut keeps whole
    words and marks that it cut.
    """
    graph = write_graph(
        tmp_path,
        [
            {
                "number": 999,
                "title": (
                    "Stabilize + document a canonical Parquet microstructure "
                    "schema (Arrow interop)"
                ),
            }
        ],
    )
    config = write_config(
        tmp_path,
        """
[[group]]
id = "new"
title = "Newly opened"
prose = "Whatever just arrived."
issues = [999]
""",
    )

    body = render_epic_body(graph, config)

    assert 'n999["#999 Stabilize + document a…"]' in body


def test_label_cannot_break_the_mermaid_quoting(tmp_path):
    """A double quote in a title would end the mermaid label early."""
    graph = write_graph(
        tmp_path, [{"number": 999, "title": 'Handle the "unknown" order type'}]
    )
    config = write_config(
        tmp_path,
        """
[[group]]
id = "new"
title = "Newly opened"
prose = "Whatever just arrived."
issues = [999]
""",
    )

    body = render_epic_body(graph, config)
    line = next(ln for ln in body.splitlines() if "n999[" in ln)

    assert '"' not in line.split('["')[1].rsplit('"]', 1)[0]


HAND_WRITTEN = """# The roadmap

This paragraph is written by a person and the generator must not touch it.

## What to build next

Ranking is judgement and is not in the graph.

<!-- ROADMAP:BEGIN -->
stale generated content
<!-- ROADMAP:END -->
"""


def test_splice_replaces_only_what_is_between_the_markers():
    """Hand-written prose above the opening marker survives every run."""
    spliced = splice_generated_block(HAND_WRITTEN, "fresh content")

    assert "This paragraph is written by a person" in spliced
    assert "Ranking is judgement" in spliced
    assert "stale generated content" not in spliced
    assert "fresh content" in spliced
    assert spliced.count("<!-- ROADMAP:BEGIN -->") == 1
    assert spliced.count("<!-- ROADMAP:END -->") == 1


def test_splice_is_idempotent():
    """Writing the same block twice leaves the body byte for byte the same."""
    once = splice_generated_block(HAND_WRITTEN, "fresh content")
    twice = splice_generated_block(once, "fresh content")

    assert twice == once


@pytest.mark.parametrize(
    "body",
    [
        "no markers at all",
        "<!-- ROADMAP:BEGIN -->\nnever closed",
        "orphan end\n<!-- ROADMAP:END -->",
        "<!-- ROADMAP:END -->\nbackwards\n<!-- ROADMAP:BEGIN -->",
    ],
)
def test_splice_refuses_to_guess_where_a_broken_block_goes(body):
    """A missing or malformed marker pair is skipped and logged, never guessed
    at: writing outside the markers would destroy someone's prose."""
    with pytest.raises(MarkerError):
        splice_generated_block(body, "fresh content")


class FakeGitHub:
    """An in-memory stand-in for the GitHub issues API.

    Serves the snapshot in the payload shape the real client returns, and holds
    the 17 issue bodies the generator writes, so a run needs no network.
    """

    def __init__(self, raw: dict, bodies: dict[int, str]):
        self._raw = raw
        self.bodies = dict(bodies)
        self.writes: list[int] = []

    def sub_issues(self, epic: int) -> list[dict]:
        return [
            {
                "number": v["number"],
                "title": v["title"],
                "state": v["state"],
                "labels": [{"name": name} for name in v["labels"]],
                "issue_dependencies_summary": {"blocked_by": v["open_blockers"]},
            }
            for v in self._raw.values()
        ]

    def blocked_by(self, number: int) -> list[dict]:
        return list(self._raw[str(number)]["blocked_by"])

    def get_body(self, number: int) -> str:
        return self.bodies[number]

    def update_body(self, number: int, body: str) -> None:
        self.bodies[number] = body
        self.writes.append(number)


EMPTY_BLOCK = "Hand-written prose.\n\n<!-- ROADMAP:BEGIN -->\n<!-- ROADMAP:END -->\n"


@pytest.fixture
def client():
    raw = json.loads(GRAPH_FIXTURE.read_text())["nodes"]
    goals = [int(k) for k, v in raw.items() if "goal" in v["labels"]]
    bodies = {n: EMPTY_BLOCK for n in [124, *goals]}
    return FakeGitHub(raw, bodies)


def test_second_run_changes_nothing(client, config):
    """The whole point: run it twice and the second run writes nothing.

    A run on every ``issues`` event would otherwise churn 17 edit histories, so
    each body is compared before it is written.
    """
    first = run(client, config, epic=124)
    after_first = dict(client.bodies)

    second = run(client, config, epic=124)

    assert sorted(first.written) == sorted([124, *range(173, 189)])
    assert second.written == []
    assert sorted(second.unchanged) == sorted([124, *range(173, 189)])
    assert client.bodies == after_first


def test_run_keeps_the_hand_written_half_of_every_body(client, config):
    """Nothing outside the markers is touched, in any of the 17 issues."""
    run(client, config, epic=124)

    for number, body in client.bodies.items():
        assert body.startswith("Hand-written prose.\n\n"), f"#{number}"


def test_run_skips_an_issue_whose_markers_are_missing(client, config):
    """One broken body does not stop the other sixteen, and does not raise."""
    client.bodies[180] = "someone deleted the markers"

    report = run(client, config, epic=124)

    assert report.skipped == [180]
    assert client.bodies[180] == "someone deleted the markers"
    assert 124 in report.written


def test_a_skipped_issue_fails_the_run(client, config):
    """A skip means a view nobody is maintaining is now stale, so the run fails.

    The generator carries on past a broken body, because one bad marker pair
    must not stop the other sixteen.  But the run as a whole has not done its
    job: the skipped issue still shows whatever the graph said the last time
    anyone could write to it.  A green run there would report success for a
    roadmap that had quietly stopped updating, which is the drift this whole
    thing exists to end.
    """
    client.bodies[180] = "someone deleted the markers"

    report = run(client, config, epic=124)

    assert exit_code(report) == 1


def test_a_run_that_writes_every_issue_succeeds(client, config):
    report = run(client, config, epic=124)

    assert report.skipped == []
    assert exit_code(report) == 0


def test_a_run_that_changes_nothing_succeeds(client, config):
    """Nothing to write is the steady state, not a failure."""
    run(client, config, epic=124)

    report = run(client, config, epic=124)

    assert report.written == []
    assert exit_code(report) == 0


def test_epic_sections_run_in_the_order_the_spec_gives(graph, config):
    """Status count, then the work diagrams, then the goals.

    Every work issue is in a diagram today, so there is no Ungrouped section to
    order; where that section falls is checked where it is made to appear.
    """
    body = render_epic_body(graph, config)

    assert [line for line in body.splitlines() if line.startswith("## ")] == [
        "## Where the work stands",
        "## What each capability waits on",
    ]
    assert body.index("### Part 1 - the foundation") < body.index(
        "## What each capability waits on"
    )


def test_every_generated_diagram_stays_small(graph, config):
    """The size rule covers the work diagrams too, not just the goal ones."""
    bodies = [render_epic_body(graph, config)]
    bodies += [render_goal_body(graph, config, g.number) for g in graph.goals]

    for body in bodies:
        for block in body.split("```mermaid")[1:]:
            drawn = re.findall(
                r"^ {2}(?:n|g|alt)\d+[\[\(\{]",
                block.split("```")[0],
                flags=re.MULTILINE,
            )
            assert len(drawn) <= 12


def test_epic_explains_what_the_colours_mean(graph, config):
    """The key is generated too.

    #124's key is hand-written today and describes the older scheme.  It sits
    with the diagrams it explains, so it goes when they do, and the fills mean
    nothing without it.
    """
    body = render_epic_body(graph, config)
    key = body.split("```mermaid")[0]

    assert "green = done" in key
    assert "purple = ready to start" in key
    assert "grey = waiting on something" in key
    assert "blue = a goal" in key
    assert "heavy outline" in key
    assert "arrow points from a task to the work it unblocks" in key


class EditedMidRun(FakeGitHub):
    """A GitHub where a person edits one body while the run is reading it.

    ``bodies`` holds the truth, including the edit.  The first read of the
    target returns what was there beforehand, so a generator that reads a body
    twice sees one version and writes another.
    """

    def __init__(self, raw: dict, bodies: dict[int, str], target: int, stale: str):
        super().__init__(raw, bodies)
        self.target = target
        self.stale = stale
        self.reads: dict[int, int] = {}

    def get_body(self, number: int) -> str:
        self.reads[number] = self.reads.get(number, 0) + 1
        if number == self.target and self.reads[number] == 1:
            return self.stale
        return self.bodies[number]


def test_a_body_edited_during_a_run_keeps_the_edit(client, config):
    """Each body is read once, so a run cannot write back what it did not read.

    Reading twice means splicing one version and comparing against another: a
    prose edit landing between the two reads makes the comparison differ, and
    the write then carries the prose from before the edit.  The edit is lost
    with nothing logged.
    """
    run(client, config, epic=124)
    stale = client.bodies[180]
    edited = stale.replace("Hand-written prose.", "Hand-written prose, revised.")
    assert edited != stale

    flaky = EditedMidRun(
        json.loads(GRAPH_FIXTURE.read_text())["nodes"],
        {**client.bodies, 180: edited},
        target=180,
        stale=stale,
    )
    report = run(flaky, config, epic=124)

    assert flaky.reads[180] == 1
    assert "revised" in flaky.bodies[180]
    assert report.written == []
