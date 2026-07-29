# Agent IDX — working notes

Read [`CONTEXT.md`](./CONTEXT.md) before touching this repo. It is the shared vocabulary: name things the way it names them, and check its "Flagged ambiguities" section before introducing a term that already has three meanings here.

Constraints that are easy to get wrong:

- **One browser, one thread.** All Stockbit scraping shares a single Playwright singleton pinned to one worker thread. Never open a second one, and never close it to "clean up" — the login session lives in it.
- **Mutations are staged.** Anything that writes to disk, posts to Stockbit, or farms in bulk must be staged and confirmed with `ya`, not executed directly.
- **A missing number is `GAP_DATA`.** Never substitute an estimate for a figure the source did not provide.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues for `hrizriz/agent_idx`. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the five canonical triage labels without aliases. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repo with `CONTEXT.md` at the root and ADRs under `docs/adr/`. See `docs/agents/domain.md`.
