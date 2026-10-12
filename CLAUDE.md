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
  in the same commit.
- A page found disagreeing with the code is fixed toward the code, and the
  commit message says so.

`tests/test_docs_index.py` catches missing and stale paths and missing pages.
It cannot catch stale prose: that part is on you.

Write code references in docs as inline code paths (`src/poseydon/core/spec.py`),
not links: links leaving `docs/` fail the strict site build. Cite specs and
plans the same way (`docs/superpowers/specs/...`): `docs/superpowers/` is left
out of the site, so a link into it would be dead there. Never invent a
rationale; write **rationale not recorded** instead.

## Running things

The library, tests and scripts never run on host Python; they go through
docker compose. The docs site (`uvx`) is the one exception: it needs nothing
from `poseydon`.

- Tests: `docker compose run --rm -T test python -m pytest -q`
- Docs site: `uvx --from 'mkdocs<2' --with mkdocs-material mkdocs serve` (http://127.0.0.1:8000)
- Docs check: `uvx --from 'mkdocs<2' --with mkdocs-material mkdocs build --strict`
