# PoseYdon

Start at `docs/index/INDEX.md`. It says where everything lives and which page
explains it. The repo has two sides: the **library** (`src/poseydon/`, owns
every contract) and the **tooling** around it (`scripts/`, `tools/`,
`configs/`, `tests/`, `data/`). Keep them apart in code and in docs.

## docs/index/ is part of every change

- A change to behaviour, structure or a contract under `src/poseydon/<pkg>/`
  updates `docs/index/library/<pkg>.md` in the same commit.
- Adding, moving or removing a package, module, script or tool updates
  `docs/index/INDEX.md` in the same commit; for a new package, add its page too.
- A change to a script, tool, config group or compose service updates
  `docs/index/tooling.md`.
- Fixing something a page lists as open (under **Gotchas**, or a guide's
  **Open issues**, e.g. `docs/guides/roundtrip.md`) means updating that bullet
  in the same commit, and removing its row from **Open work** in
  `docs/index/INDEX.md`, which lists everything known and not fixed.
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

## Running things

The library, tests and scripts never run on host Python; they go through
docker compose. The docs site (`uvx`) is the one exception: it needs nothing
from `poseydon`.

- Python dependencies live in `pyproject.toml` + `uv.lock`, never in the
  Dockerfile: `uv add <pkg> [--optional <extra>]`, then `docker compose build`.
  Docker carries only OS and system packages (and Blender's distro Python).
- Tests: `docker compose run --rm -T test python -m pytest -q`
- Docs site: `uvx --from 'mkdocs<2' --with mkdocs-material mkdocs serve` (http://127.0.0.1:8000)
- Docs check: `uvx --from 'mkdocs<2' --with mkdocs-material mkdocs build --strict`
