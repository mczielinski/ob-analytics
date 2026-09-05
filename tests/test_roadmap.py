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
import logging
import re
from pathlib import Path

import pytest

from scripts.roadmap import (
    MarkerError,
    exit_code,
    load_config,
    load_graph,
    named_issues,
    render_epic_body,
    render_goal_body,
    run,
    splice_generated_block,
    stale_captions,
    stale_mentions,
    unknown_holds,
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


def test_a_blocker_outside_the_roadmap_is_reported_not_dropped(config, caplog):
    """An edge the graph cannot keep is named, in the log and in the epic.

    A blocker that is not a child of #124 has no title and no state here, so
    the edge cannot be drawn and is left out.  Losing it in silence makes a
    goal look easier than it is, while GitHub's own dependency panel still
    names the blocker a few lines above the generated block.  The snapshot has
    no such edge, so one is added: #206 blocking goal #178, which is the case
    that found this.
    """
    raw = json.loads(GRAPH_FIXTURE.read_text())["nodes"]
    raw["178"]["blocked_by"] = [
        *raw["178"]["blocked_by"],
        {"number": 206, "state": "open"},
    ]
    goals = [int(k) for k, v in raw.items() if "goal" in v["labels"]]
    client = FakeGitHub(raw, {n: EMPTY_BLOCK for n in [124, *goals]})

    with caplog.at_level(logging.WARNING, logger="roadmap"):
        report = run(client, config, epic=124)

    assert report.dropped_edges == [(178, 206)]
    assert "#178 is blocked by #206" in caplog.text
    assert "## Blockers outside the roadmap" in client.bodies[124]
    assert "- #178 is blocked by #206" in client.bodies[124]
    # and #206 is not made a node to save its edge
    assert "#206" not in client.bodies[178]


def test_the_snapshot_has_no_blocker_outside_the_roadmap(client, config):
    """Nothing is reported for the real graph, so its views are unchanged."""
    report = run(client, config, epic=124)

    assert report.dropped_edges == []
    assert "## Blockers outside the roadmap" not in client.bodies[124]


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
        "## What to pick up next",
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


def test_green_and_purple_mean_what_they_mean_on_github(graph, config):
    """These are GitHub's own two state colours, so they must agree with it.

    The diagrams render inside issue bodies, directly under GitHub's own state
    chips: green on an open issue, purple on one closed as completed. Assigning
    the same two colours to the opposite states made the roadmap contradict the
    page it was sitting on. The class names are the readiness, so only the
    fills move.
    """
    body = render_epic_body(graph, config)

    assert "classDef done fill:#8250df" in body
    assert "classDef doneF fill:#8250df" in body
    assert "classDef ready fill:#1a7f37" in body
    assert "classDef readyF fill:#1a7f37" in body


def test_epic_explains_what_the_colours_mean(graph, config):
    """The key is generated too.

    #124's key is hand-written today and describes the older scheme.  It sits
    with the diagrams it explains, so it goes when they do, and the fills mean
    nothing without it.
    """
    body = render_epic_body(graph, config)
    key = body.split("```mermaid")[0]

    assert "purple = done" in key
    assert "green = ready to start" in key
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
    """A run with nothing to write reads once, and cannot write back a stale copy.

    The comparison is what stops a run on every issue event from churning
    seventeen edit histories, and a run that finds nothing to change never
    reaches a write at all.
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


def test_a_write_carries_the_prose_as_it_is_at_the_write(config):
    """Runs overlap, so the body a run started with is not the body it writes.

    This is a real revert, not a hypothesis: a run already in flight held #124's
    prose from before a rewrite and wrote it back afterwards, undoing the
    rewrite with nothing logged. The write splices into a second read, so the
    prose that lands is the prose that was there when the write happened.
    """
    raw = json.loads(GRAPH_FIXTURE.read_text())["nodes"]
    goals = [int(k) for k, v in raw.items() if "goal" in v["labels"]]
    before = EMPTY_BLOCK
    after = before.replace("Hand-written prose.", "Hand-written prose, revised.")
    flaky = EditedMidRun(
        raw,
        {n: EMPTY_BLOCK for n in [124, *goals]} | {124: after},
        target=124,
        stale=before,
    )

    report = run(flaky, config, epic=124)

    assert 124 in report.written
    assert "revised" in flaky.bodies[124]
    assert "## Where the work stands" in flaky.bodies[124]


# ---------------------------------------------------------------------------
# What to pick up next: derived, so that no one has to write it down
# ---------------------------------------------------------------------------


def next_up(body: str) -> str:
    """The "what to pick up next" section of a rendered epic body."""
    return (
        body.split("## What to pick up next")[1]
        .split("## Where")[0]
        .split("### Part")[0]
    )


def test_the_goal_one_issue_from_done_is_named_with_that_issue(graph, config):
    """The strongest thing the graph can say: this issue finishes that goal.

    From the snapshot, four open goals have a single prerequisite left: #177
    needs #113, #182 needs #150, #184 needs #108, and #179 needs either #102 or
    #103, which the config declares a choice and which therefore counts as one.
    """
    section = next_up(render_epic_body(graph, config))

    assert "#113" in section and "#177" in section
    assert "#150" in section and "#182" in section
    assert "#108" in section and "#184" in section
    assert "#102 or #103" in section and "#179" in section


def test_a_goal_several_issues_away_is_not_listed_as_one_away(graph, config):
    """#173 waits on two things in the snapshot, so it is not in that list."""
    one_away = next_up(render_epic_body(graph, config)).split("**Frees")[0]

    assert "#173" not in one_away


def test_work_that_frees_other_work_is_ranked_by_how_much_it_frees(graph, config):
    """The one ordering the graph justifies, so it is the one that is printed.

    In the snapshot #136 has three open issues waiting on it, #100 has two, and
    #105, #144 and #148 have one each.
    """
    section = next_up(render_epic_body(graph, config))
    frees = section.split("**Frees other work.**")[1].split("**")[0]
    order = [line.split()[1] for line in frees.strip().splitlines()]

    assert order == ["#136", "#100", "#105", "#144", "#148"]


def test_a_ready_issue_nothing_waits_on_is_counted_not_listed(graph, config):
    """Naming twenty-odd independent issues would bury the three that matter."""
    section = next_up(render_epic_body(graph, config))

    assert "Free to take in any order." in section
    assert "#117" not in section


def test_the_lists_do_not_count_the_same_issue_twice(graph, config):
    """The four groups partition the open work, which is the sum a reader checks."""
    section = next_up(render_epic_body(graph, config))
    free = int(re.search(r"any order\.\*\* (\d+) other issues", section).group(1))
    named = set(re.findall(r"^- (.+)$", section, re.MULTILINE))
    listed = {int(n) for line in named for n in re.findall(r"#(\d+)", line)}
    open_work = {n.number for n in graph.work if not n.is_closed}

    assert free == len(
        open_work
        - listed
        - {n.number for n in graph.work if not n.is_closed and n.open_blockers}
    )


def test_a_hold_is_printed_while_its_issues_are_open(graph, tmp_path):
    """The judgement the graph cannot make, kept next to the issues it is about."""
    config = write_config(
        tmp_path,
        """
        [[group]]
        id = "g"
        title = "Part 1 - g"
        prose = "A group."
        issues = [138, 139]

        [[hold]]
        issues = [138, 139]
        reason = "wait for a measurement that asks for them"
        """,
    )

    section = next_up(render_epic_body(graph, config))

    assert "wait for a measurement that asks for them" in section
    assert "#138" in section and "#139" in section


def test_a_hold_disappears_when_its_issues_close(graph, tmp_path):
    """This is why the judgement moved out of the epic's prose.

    #154 and #155 are closed in the snapshot, so a hold on them has nothing
    left to say and is not printed. The same sentence written into #124 by hand
    stayed there until someone noticed.
    """
    config = write_config(
        tmp_path,
        """
        [[group]]
        id = "g"
        title = "Part 1 - g"
        prose = "A group."
        issues = [154, 155]

        [[hold]]
        issues = [154, 155]
        reason = "no longer true of anything"
        """,
    )

    assert "no longer true of anything" not in render_epic_body(graph, config)


def test_a_hold_on_an_issue_no_diagram_draws_is_reported(tmp_path):
    """A hold nothing draws prints nothing, which is the silent kind of wrong."""
    config = write_config(
        tmp_path,
        """
        [[group]]
        id = "g"
        title = "Part 1 - g"
        prose = "A group."
        issues = [112]

        [[hold]]
        issues = [999]
        reason = "a number nobody checked"
        """,
    )

    assert unknown_holds(config) == ["a hold names #999, which no diagram draws"]


# ---------------------------------------------------------------------------
# Guards: prose that says where the work stands goes stale, so it is refused
# ---------------------------------------------------------------------------


def test_an_issue_number_in_hand_written_prose_is_found(graph):
    """Every such mention is a claim the graph can outrun without a word.

    Prose runs on both sides of the block in #124, so both sides are read.
    """
    body = (
        "The schema (#112) is done.\n\n<!-- ROADMAP:BEGIN -->\n- #999 waits\n"
        "<!-- ROADMAP:END -->\n\nSee #136 for the split.\n"
    )

    assert named_issues(body) == [112, 136]


def test_a_number_inside_the_block_is_the_generator_own_writing(graph):
    """The generated block is rewritten on every run, so it cannot go stale."""
    body = (
        "Prose with no numbers.\n\n<!-- ROADMAP:BEGIN -->\n- #113 waits\n"
        "<!-- ROADMAP:END -->\n"
    )

    assert named_issues(body) == []


def test_a_pointer_to_the_epic_or_to_something_off_the_roadmap_is_allowed():
    """A goal's footer says where it came from, which no graph move can falsify.

    Every goal ends "Part of #124. Decided in #170." The epic is the one fixed
    point in the whole thing, and #170 is a discussion the roadmap does not
    track, so neither can be contradicted by it.
    """
    body = "Part of #124. Decided in #170.\n"

    # What the run passes: every child of the epic except the epic itself.
    # #170 was never one of them.
    assert stale_mentions(body, tracked={112, 113}) == []


def test_prose_that_names_an_issue_fails_the_run(client, config):
    """The run still writes everything it can; the failure is the report."""
    client.bodies[124] = "The schema (#112) is done.\n\n" + EMPTY_BLOCK

    report = run(client, config, epic=124)

    assert 124 in report.written
    assert exit_code(report) == 1
    assert any("#124 names #112" in complaint for complaint in report.stale_prose)


def test_a_caption_naming_an_issue_its_diagram_dropped_is_reported(graph, tmp_path):
    """The mistake that put four stale captions in #124, found without reading it.

    #107 is closed in the snapshot with nothing open waiting on it, so pruning
    drops it from the diagram and the caption is left describing a box that is
    not there. That is exactly the sentence Part 4 carried: "trade signs and the
    first signals are done", about two boxes no reader could see.
    """
    config = write_config(
        tmp_path,
        """
        [[group]]
        id = "analytics"
        title = "Part 4 - metrics"
        prose = "Trade signs are done."
        issues = [107, 110]

        [labels]
        107 = "trade signs"
        """,
    )

    assert stale_captions(graph, config) == [
        (
            "analytics: the caption says 'trade signs', but #107 is not drawn "
            "in that diagram any more"
        )
    ]


def test_a_caption_may_name_what_its_diagram_still_draws(graph, tmp_path):
    """The check is about what is drawn, not about naming a thing at all.

    A group that keeps its closed issues, like the sources and the data-quality
    checks, can describe them freely: they are still on screen.
    """
    config = write_config(
        tmp_path,
        """
        [[group]]
        id = "analytics"
        title = "Part 4 - metrics"
        prose = "Trade signs are done."
        issues = [107, 110]
        keep_closed = true

        [labels]
        107 = "trade signs"
        """,
    )

    assert stale_captions(graph, config) == []
