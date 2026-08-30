# Issue tracker: GitHub

Issues and specs for this repo live as GitHub issues on `mczielinski/ob-analytics`.
Use the `gh` CLI for all operations.

## Which repo

This clone is a fork of `phil8192/ob-analytics`, but all issues live on the fork,
not on the parent. Two settings keep `gh` pointed at the fork:

- `origin` is the only remote (the unused `upstream` remote was removed).
- `gh repo set-default mczielinski/ob-analytics` is recorded in `.git/config`.

With both in place, plain `gh issue ...` commands resolve to the fork and need no
`--repo` flag. If `gh` ever asks which repository to use, or an issue number that
should exist comes back "not found", the default has been lost: re-run
`gh repo set-default mczielinski/ob-analytics`. Never let a write land on the
parent repo.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --comments`, filtering comments by `jq` and also fetching labels.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."`

## Labels in use

This repo has its own label taxonomy beyond the triage labels in
`.agents/triage-labels.md`. Label new issues in the existing vocabulary
rather than inventing new terms:

- `area:` — `ingestion`, `analytics`, `viz`, `venues`, `interop`, `perf`, `docs`
- `level:` — `L2` (price-level aggregated feed), `L3` (per-order feed)
- `effort:` — `S` (hours to ~1 day), `M` (a few days), `L` (~1-2 weeks), `XL` (multi-PR)
- `foundational` — a dependency root that unblocks several other issues
- `epic` — a tracking / roadmap issue

Run `gh label list` before adding a label that is not in this list.

## Roadmap structure

Issue #124 is the roadmap epic. Its children are wired as **native GitHub
sub-issues**, with **native issue dependencies** (`blocked_by`) recording the
order they must be built in. Both are the real, UI-visible structures, not a
task list in the body. Read the epic before proposing where a new issue fits,
and attach new roadmap work to it the same way.

## Pull requests as a triage surface

**PRs as a request surface: no.** _(Set to `yes` if this repo treats external PRs as feature requests; `/triage` reads this flag.)_

When set to `yes`, PRs run through the same labels and states as issues, using the `gh pr` equivalents:

- **Read a PR**: `gh pr view <number> --comments` and `gh pr diff <number>` for the diff.
- **List external PRs for triage**: `gh pr list --state open --json number,title,body,labels,author,authorAssociation,comments` then keep only `authorAssociation` of `CONTRIBUTOR`, `FIRST_TIME_CONTRIBUTOR`, or `NONE` (drop `OWNER`/`MEMBER`/`COLLABORATOR`).
- **Comment / label / close**: `gh pr comment`, `gh pr edit --add-label`/`--remove-label`, `gh pr close`.

GitHub shares one number space across issues and PRs, so a bare `#42` may be either: resolve with `gh pr view 42` and fall back to `gh issue view 42`.

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments`.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single issue with **child** issues as tickets.

- **Map**: a single issue labelled `wayfinder:map`, holding the Notes / Decisions-so-far / Fog body. `gh issue create --label wayfinder:map`.
- **Child ticket**: an issue linked to the map as a GitHub sub-issue (`gh api` on the sub-issues endpoint). Where sub-issues aren't enabled, add the child to a task list in the map body and put `Part of #<map>` at the top of the child body. Labels: `wayfinder:<type>` (`research`/`prototype`/`grilling`/`task`). Once claimed, the ticket is assigned to the driving dev.
- **Blocking**: GitHub's **native issue dependencies**, the canonical, UI-visible representation. Add an edge with `gh api --method POST repos/<owner>/<repo>/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`, where `<blocker-db-id>` is the blocker's numeric **database id** (`gh api repos/<owner>/<repo>/issues/<n> --jq .id`, _not_ the `#number` or `node_id`). GitHub reports `issue_dependencies_summary.blocked_by` (open blockers only, the live gate). Where dependencies aren't available, fall back to a `Blocked by: #<n>, #<n>` line at the top of the child body. A ticket is unblocked when every blocker is closed.
- **Frontier query**: list the map's open children (`gh issue list --state open`, scoped to the map's sub-issues / task list), drop any with an open blocker (`issue_dependencies_summary.blocked_by > 0`, or an open issue in the `Blocked by` line) or an assignee; first in map order wins.
- **Claim**: `gh issue edit <n> --add-assignee @me`, the session's first write.
- **Resolve**: `gh issue comment <n> --body "<answer>"`, then `gh issue close <n>`, then append a context pointer (gist + link) to the map's Decisions-so-far.
