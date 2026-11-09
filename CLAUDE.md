# PoseYdon

Research infrastructure for neural motion (`poseydon` package): generation,
in-betweening, blending and cross-topology retargeting on characters of any
shape, from the vendor's rig back onto that same rig.

<!-- parallax:start -->
## Docs

Docs live in `docs/parallax/`; start every task at `docs/parallax/pages/index.md`. The code is the oracle, docs
follow, and neither implies correctness: treat a doc claim as a lead and check it in the
code. Update the docs in the same change as the code, and never change code to match a doc.
<!-- parallax:end -->

## Repo notes

The repo has two sides: the **library** (`src/poseydon/`, owns every contract)
and the **tooling** around it (`scripts/`, `tools/`, `configs/`, `tests/`,
`data/`). Keep them apart in code and in docs.

- Tickets (bugs left unfixed, ideas worth keeping) are opened, edited and closed
  only with the scripts in `docs/parallax/tools/`; how is in
  `docs/parallax/pages/tickets/index.md`.
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
from `poseydon` (it runs as the `docs` service or with `docs/parallax/tools/serve.py`), and the ticket
scripts, which use only the standard library and the host's `gh` login.

- Python dependencies live in `pyproject.toml` + `uv.lock`, never in the
  Dockerfile: `uv add <pkg> [--optional <extra>]`, then `docker compose build`.
  Docker carries only OS and system packages (and Blender's distro Python).
- Tests: `docker compose run --rm -T test python -m pytest -q`
- Docs site: `docker compose up -d docs` (http://127.0.0.1:8019, stays up
  across reboots until `docker compose stop docs`), or `python3 docs/parallax/tools/serve.py` (http://127.0.0.1:8000)
- Docs check: `python3 docs/parallax/tools/check_docs.py` and `python3 docs/parallax/tools/serve.py --build` (strict)
- The site deploys from `main` through `.github/workflows/docs.yml`.
