# PoseYdon

Start at `docs/index/INDEX.md`. It says where everything lives and which page
explains it. The repo has two sides: the **library** (`src/poseydon/`, owns
every contract) and the **tooling** around it (`scripts/`, `tools/`,
`configs/`, `tests/`, `data/`). Keep them apart in code and in docs.

## docs/index/ is part of every change

- A change to behaviour, structure or a contract under `src/poseydon/<pkg>/`
  updates that package's API page, `docs/api/<pkg>/index.md` (or the module
  page in its dropdown), in the same commit: the intro under the title, the
  **Limitations** (what is bug-prone and not guarded by code) and the class
  diagram. Open bugs and leftovers go to `docs/index/todo.md`, not the page.
- Adding, moving or removing a package, module, script or tool updates
  `docs/index/INDEX.md` in the same commit; for a new package, add its page too,
  plus its API page `docs/api/<pkg/as/path>/index.md` (`::: poseydon.<pkg>`)
  and nav line in `mkdocs.yml`. A change to a package's classes, bases or
  ownership updates that page's class diagram (legend: `docs/api/index.md`).
- A change to a script, tool, config group or compose service updates
  `docs/index/tooling.md`.
- Fixing something a page lists as open (under **Limitations**, or a guide's
  **Open issues**, e.g. `docs/guides/concepts/data/roundtrip.md`) means updating that bullet
  in the same commit.

## docs/index/todo.md

`docs/index/todo.md` lists what is known to be open: suspected bugs,
leftovers and pending decisions. Read it when starting work, but treat each
row as a lead to verify against the code, not as fact: rows go stale, and a
cause recorded there may be wrong. Keep it current. Add a row for anything
you find and don't fix now. Remove or correct a row in the same commit that
fixes or disproves it, and fix a row you find to be wrong.
- A page found disagreeing with the code is fixed toward the code, and the
  commit message says so.

`tests/test_docs_index.py` catches missing and stale paths and missing pages.
It cannot catch stale prose: that part is on you.

Write code references in docs as inline code paths (`src/poseydon/core/spec.py`),
not links: links leaving `docs/` fail the strict site build. Never invent a
rationale; write **rationale not recorded** instead.

Design specs and implementation plans from planning workflows (e.g. the
superpowers skills) are working notes, not documentation: write them under
`.superpowers/` (gitignored), never into `docs/`, and never cite them from the
docs. Reasoning that matters goes into docstrings, commit messages and the
`docs/index/` pages.

## Tests stay few and macro

Don't bloat `tests/` or `scripts/`. A test earns its place by covering a whole
path end to end (the round trip is the model: raw source in, file out,
compared against the source), not one function or one detail. A bug should
surface as a macro test failing; don't add a test per fix, per edge case or
per helper. Before adding a test, check whether an existing macro test can
assert it in a line or two; if a new one is unavoidable, keep it one function.
The same goes for scripts: extend the existing entry point rather than adding
a new script for a one-off check (probes go in gitignored `.superpowers/`).

## Running things

The library, tests and scripts never run on host Python; they go through
docker compose. The docs site needs nothing from `poseydon`: it runs as the
`docs` service or straight from the host with `uvx`.

- Python dependencies live in `pyproject.toml` + `uv.lock`, never in the
  Dockerfile: `uv add <pkg> [--optional <extra>]`, then `docker compose build`.
  Docker carries only OS and system packages (and Blender's distro Python).
- Tests: `docker compose run --rm -T test python -m pytest -q`
- Docs site: `docker compose up -d docs` (http://127.0.0.1:8019, stays up
  across reboots until `docker compose stop docs`), or `uvx --from 'mkdocs<2' --with mkdocs-material --with 'mkdocstrings[python]' mkdocs serve` (http://127.0.0.1:8000)
- Docs check: `uvx --from 'mkdocs<2' --with mkdocs-material --with 'mkdocstrings[python]' mkdocs build --strict`
