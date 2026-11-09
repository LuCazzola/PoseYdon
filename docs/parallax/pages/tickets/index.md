# Tickets

A ticket is a bug, loose end or idea that was noticed but not acted on. Tickets are
how a mismatch between the docs, the code and what someone expected gets kept.

- **Issues** (bugs and loose ends) live in GitHub Issues, labelled by priority
  tier and `needs-triage` until reproduced:
    - `Prio:CRITICAL`: breaks a run, or may harm people, systems or data.
    - `Prio:WANTED`: relevant features worth doing, but not urgent.
    - `Prio:MINOR`: small, cheap fixes: stale references, noisy logs, cosmetic gaps.
- **Ideas** (proposed changes and things worth investigating; nothing is broken
  yet) live in the [Ideas](ideas.md) table, in the repo, so they are reviewed in the
  same PR as the code they concern.

Open, edit and close tickets only with these scripts, run where `gh` is logged in:

```bash
python3 docs/parallax/tools/open_ticket.py <critical|wanted|minor|idea> "<issue>" <path> [<path> ...] \
    [--title "<title>"] [--repro "<command>"] [--cause "<guess>"]
python3 docs/parallax/tools/edit_ticket.py <ID|#N> [--tier <tier>] [--issue "<issue>"] [--where <path> ...] \
    [--repro "<command>"] [--cause "<guess>"] [--verified "<evidence>"]
python3 docs/parallax/tools/close_ticket.py <ID|#N> [...] [--reason "not planned"] [--comment "<why>"]  # --comment required for issues
```

- An idea gets a 4-character ID (`grep <ID> docs/parallax/pages/tickets/ideas.md`), an
  issue a number (`gh issue view <N>`).
- An issue keeps apart what was **Observed** (`<issue>`: facts only), how to
  **Reproduce** it (`--repro`: a command) and the **Suspected cause** (`--cause`),
  because the guessed cause is the part most often wrong.
- Every new issue carries `needs-triage`. Drop it with `--verified` only on
  evidence: a failing test, or a command and its output.
- A fix closes its issue through `Fixes #N: <what fixed it>` in the commit message
  or PR, one line per issue. `close_ticket.py` is for issues that turn out wrong or
  not planned, and needs `--comment` saying why.
- Moving a ticket between an idea and an issue (`--tier`) opens it in the new
  place and closes it in the old one.

!!! warning "Leads, not facts"
    Each ticket is what was believed when it was written. It may be stale, already
    fixed, or wrong about the cause. Before acting on one, check it against the code,
    and fix the ticket if it disagrees.
