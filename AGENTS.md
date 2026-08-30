# AGENTS.md

## Agent skills

Configuration for the engineering skills lives in **`.agents/`**, not
`docs/agents/`. Everything under `docs/` is published to the public
documentation site, and these files are internal. A skill that looks for
`docs/agents/<name>.md` and doesn't find it should read `.agents/<name>.md`
instead — the setup is present, only the directory differs.

### Issue tracker

GitHub Issues on `mczielinski/ob-analytics`. See `.agents/issue-tracker.md`.

### Triage labels

The five canonical triage labels, unchanged. See `.agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` and `adr/` at the repo root. See `.agents/domain.md`.
