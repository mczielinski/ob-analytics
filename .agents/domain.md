# Domain Docs

How the engineering skills should consume this repo's domain documentation when
exploring the codebase.

## Where these files live

Anything placed under `docs/` is published to the public site
(https://mczielinski.github.io/ob-analytics/) — the site builder renders every
`**/*.md` under `docs/`, whether or not it appears in the `nav` in
`zensical.toml`. Internal agent-facing documents therefore live **outside**
`docs/`:

- Agent skill configuration: `.agents/` (this file and its siblings).
- Architecture decision records: `adr/` at the repo root, **not** `docs/adr/`.

`CONTEXT.md` sits at the repo root, which is also outside the published tree.

If you later want an ADR to be public, add it to `docs/` *and* to the `nav` in
`zensical.toml` deliberately, rather than letting it land there by default.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root.
- **`adr/`**: read ADRs that touch the area you're about to work in.
- **`ARCHITECTURE.md`** at the repo root, and **`docs/architecture.md`** — these
  already exist and describe the pipeline structure. Read them before proposing
  structural change.

If any of these files don't exist, **proceed silently**. Don't flag their
absence; don't suggest creating them upfront. The `/domain-modeling` skill
(reached via `/grill-with-docs` and `/improve-codebase-architecture`) creates
them lazily when terms or decisions actually get resolved.

## File structure

This is a single-context repo: one package, `ob_analytics/`, one glossary.

```
/
├── CONTEXT.md
├── adr/                 ← empty today; ADRs are created lazily
└── ob_analytics/
```

A multi-context layout (a root `CONTEXT-MAP.md` pointing at one `CONTEXT.md` per
context, plus per-context ADR directories) is what you would move to if this repo
ever split into several packages. It has not, so don't build it.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal,
a hypothesis, a test name), use the term as defined in `CONTEXT.md`.

This repo also has a published, human-facing glossary at `docs/glossary.md`,
covering order book terminology (level, resolution, event, lifecycle, and so on).
Read it as well, and don't drift to synonyms either glossary avoids. Where the
two disagree, `docs/glossary.md` is the one users read: fix `CONTEXT.md` to match
rather than inventing a third term.

If the concept you need isn't in either glossary yet, that's a signal: either
you're inventing language the project doesn't use (reconsider) or there's a real
gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than
silently overriding:

> _Contradicts ADR-0007 (event-sourced orders), but worth reopening because…_
