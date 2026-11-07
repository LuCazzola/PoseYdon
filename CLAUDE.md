# PoseYdon

Research infrastructure for neural motion (`poseydon` package): generation,
in-betweening, blending and cross-topology retargeting on characters of any
shape, from the vendor's rig back onto that same rig.

## Start here: the docs home

The repo is documented in `docs/` (MkDocs Material, config in `mkdocs.yml`,
published at https://lucazzola.github.io/PoseYdon/, served locally by
`docker compose up -d docs` on http://127.0.0.1:8019). Start every task at
`docs/index.md`: its map table says which page covers what, so follow it to the
pages your task needs before reading or changing code. The Index map
(`docs/reference/map.md`) tells you where the code for each part lives; the
Design rationale (`docs/user-guide/index.md`) explains the structuring
philosophy behind it, so read it before adding or moving code. Adding a rig,
dataset, feature block, model, process, loss or operation? Follow its recipe in
`docs/how-to/adding.md`; if a step there is wrong or missing, fix the page.

The repo has two sides: the **library** (`src/poseydon/`, owns every contract)
and the **tooling** around it (`scripts/`, `tools/`, `configs/`, `tests/`,
`data/`). Keep them apart in code and in docs.

## Keeping the docs true

- **The docs must match the code.** When a change alters anything a page
  states (a path, command, flag, default, config key, file format, behaviour),
  update that page in the same commit. If you find a page that disagrees with
  the code, fix the page toward the code, never the other way round, and say
  so in the commit message.
- **A change under `src/poseydon/<pkg>/`** to behaviour, structure or a contract
  updates that package's API page, `docs/api/<pkg>/index.md` (or the module page
  in its dropdown): the intro under the title, the **Limitations** (what is
  bug-prone and not guarded by code) and the class diagram.
- **Always update the Index**: `docs/reference/map.md` when a package, module,
  script or tool is added, moved or removed; `docs/reference/tooling.md` when a
  script, tool, config group or compose service changes.
- **Tickets**: a bug you notice but don't fix becomes a GitHub issue, an idea
  worth keeping a row in `docs/reference/ideas.md`; open, edit and close both only
  with the scripts that page's header lists. An issue keeps what was observed apart
  from its repro command (`--repro`) and suspected cause (`--cause`). Issues start
  `needs-triage` (leads, not facts): drop it only with evidence (`--verified`). A
  fix closes its issue via `Fixes #N: <what fixed it>` in the commit, one line per
  issue. A user may name a ticket by issue number (`gh issue view <N>`) or by an
  idea's 4-character ID (`grep <ID> docs/reference/ideas.md`). Read the tickets
  touching your work when you start, and check each against the code: they go
  stale, and a recorded cause may be wrong.
- **Fixing something a page lists as open** (under **Limitations**, or a guide's
  **Open issues**, e.g. `docs/user-guide/roundtrip.md`) means updating that
  bullet in the same commit.
- **A new package needs an API page**: `docs/api/<package/as/path>/index.md` with a
  `::: poseydon.<pkg>` block and one `:::` line per module, listed in the nested
  API section of the `mkdocs.yml` nav. `tests/test_docs_index.py` fails otherwise.
- **Class diagrams** (Mermaid, top of the API pages) are hand-written: update
  them when a change touches a package's classes, bases or ownership. Give every
  class a colour role at the end of the diagram: `cssClass "A,B" iface|impl|data|ext`
  (legend: `docs/api/index.md`, palette: `docs/javascripts/diagrams.js`).
- **The docs build must stay strict-clean**: `mkdocs build --strict` (pages
  missing from the nav, broken links, broken anchors and unresolved
  `[name][poseydon.x.y]` API links all fail it). The command is under
  [Running things](#running-things); CI runs it on every PR.
- `tests/test_docs_index.py` catches missing and stale paths and missing pages.
  It cannot catch stale prose: that part is on you.
- Write code references in docs as inline code paths
  (`src/poseydon/core/spec.py`), not links: links leaving `docs/` fail the strict
  build. Never invent a rationale; write **rationale not recorded** instead.
- Design specs and implementation plans from planning workflows (e.g. the
  superpowers skills) are working notes, not documentation: write them under
  `.superpowers/` (gitignored), never into `docs/`, and never cite them from the
  docs. Reasoning that matters goes into docstrings, commit messages and the
  docs pages.
- **The site is public.** Nothing private goes in `docs/`: no host names, no
  credentials, no licensed vendor data.

## Tests

- **Write macro tests, and few of them.** A test drives a real path end to end
  (the round trip is the model: raw source in, file out, compared against the
  source) and covers several functionalities in one run. A red macro test tells
  you a bug exists; tracking down which function causes it happens on the fly
  while debugging, not in a pre-written test per fix, per edge case or per helper.
  Before adding a test, check whether an existing macro test can assert it in a
  line or two; if a new one is unavoidable, keep it one function.
- **Write the test first.** For a feature or fix, write the macro test before the
  code, watch it go red, then change the code until it is green. The code changes
  to satisfy the test; the test changes only when the requirement does, and you
  say so to the user.
- **Distrust every test, green ones included.** A pass proves only what the test
  actually asserts. Before relying on one, read its assertions and break the code
  it covers to watch it go red; a test that cannot go red gets deleted. Report each
  result with what ran (which container, which dataset, which rigs).
- The same goes for scripts: don't bloat `scripts/`. Extend the existing entry
  point rather than adding a new script for a one-off check (probes go in
  gitignored `.superpowers/`).

## Agent Behavior

- **Verify end-to-end before calling something done**; run it and report what you
  actually observed, not just that it compiles.
- **Stop and ask when a requirement**, scope or design choice isn't clearly specified.
- **Question whether new code is necessary at all before writing it**; prefer
  reusing what exists or the smallest change that solves the actual problem.
- **Keep comments small and effective**; only where the WHY genuinely isn't obvious
  from the code (a hidden constraint, a subtle invariant, a non-obvious tradeoff). No
  comment beats a meaningless one: don't restate what the code already says. When
  one's warranted, keep it to 1-3 lines stating the constraint itself, not a
  step-by-step narration of the mechanism and its consequences: a text wall gets
  skipped by the next reader just as fast as no comment at all.
- **Generalizability is key**; this repo is meant to be foundational work: new rigs,
  datasets, representations, models and processes will keep arriving. Keep the design
  modular, and keep what a character or dataset needs in data and configs, not code.
- `external/neural_motion_blending/` is a read-only checkout of the published
  implementation, the parity reference; don't edit it.

## Running things

The library, tests and scripts never run on host Python; they go through
docker compose. Two things stay on the host: the docs site, which needs nothing
from `poseydon` (it runs as the `docs` service or with `uvx`), and the ticket
scripts, which use only the standard library and the host's `gh` login.

- Python dependencies live in `pyproject.toml` + `uv.lock`, never in the
  Dockerfile: `uv add <pkg> [--optional <extra>]`, then `docker compose build`.
  Docker carries only OS and system packages (and Blender's distro Python).
- Tests: `docker compose run --rm -T test python -m pytest -q`
- Docs site: `docker compose up -d docs` (http://127.0.0.1:8019, stays up
  across reboots until `docker compose stop docs`), or `uvx --from 'mkdocs<2' --with mkdocs-material --with 'mkdocstrings[python]' mkdocs serve` (http://127.0.0.1:8000)
- Docs check: `uvx --from 'mkdocs<2' --with mkdocs-material --with 'mkdocstrings[python]' mkdocs build --strict`
- The site deploys from `main` through `.github/workflows/docs.yml`.
