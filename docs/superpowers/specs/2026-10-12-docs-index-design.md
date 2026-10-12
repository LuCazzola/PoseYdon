# Documentation index — design

Date: 2026-10-12

## Problem

`docs/` explains how to run the project (`guides/`) and argues individual
designs (`superpowers/specs/`, `superpowers/plans/`), but nothing answers the
two questions asked first by anyone — human or agent — arriving cold:

1. **Where does X live?** There are 16 subpackages under `src/poseydon/`, 10
   scripts and 17 tools/probes. Finding the owner of a concept means grepping.
2. **How does it fit together, and why?** The rationale exists, but scattered
   across docstrings, four specs, `guides/architecture.md` and commit messages.
   There is no per-area entry point that collects it.

And nothing keeps any such document current: `docs/README.md` already admits
specs drift from code.

## Goals

- `docs/index/INDEX.md`: a map from path to responsibility, written for agents.
- One page per `src/poseydon/` subpackage, plus an overview, explaining what each
  part does, how it connects, and why it is shaped that way, for humans and agents.
- A browsable site (search, nav, diagrams) over all of `docs/`.
- Drift caught mechanically where it can be, and by an explicit rule where it cannot.

## Non-goals

- API reference. `docs/README.md`'s position stands: contracts live in
  docstrings, and a generated API doc would be a second source of truth.
- Moving `guides/`, `specs/` or `plans/`. Existing links in commits and specs
  keep working; the site's nav surfaces them in place.
- Inventing rationale. Where no reason is recorded in code, docstrings, specs or
  commit messages, the page says **rationale not recorded** rather than guessing.

## Layout

```
docs/index/
  INDEX.md        path → what lives there → page that explains it
  overview.md     pipeline as a whole (Mermaid flowchart) and cross-cutting rules
  <pkg>.md        one per src/poseydon/<pkg>: core, build, io, features, ...
mkdocs.yml        site config at repo root, docs_dir: docs
CLAUDE.md         points agents at INDEX.md; states the upkeep rule
```

### INDEX.md

Plain tables, grouped: `src/poseydon/` (every subpackage and top-level module),
`scripts/`, `tools/`, `configs/`, `tests/`, `data/` layout, `docs/`. Each row:
path, one-line responsibility, link to the explaining page (package page, guide
or spec). Paths are written as inline code so the check below can find them.

### Package pages

Fixed headings so agents can jump straight to a section:

- **Purpose**: what the package owns and what it explicitly does not own.
- **Key files and types**: the entry points worth reading first.
- **How it connects**: what it consumes, what it produces, and who calls it.
- **Why it is shaped this way**: decisions, each linked to the spec, docstring
  or commit that records it.
- **Gotchas**: the invariants that bite.

### overview.md

A Mermaid flowchart of ingest → build → train → sample → export with the
package that owns each step, then the cross-cutting rules (skeleton knowledge
is data, features composed at read time, only declared conditioning is loaded,
...). It links to `guides/architecture.md` for the worked examples rather than
copying them.

## Keeping it current

**Mechanical check:** `tests/test_docs_index.py`. Stdlib only, so it runs in the
existing pure suite (`docker compose run --rm -T test python -m pytest -q`). It
fails when:

1. a `src/poseydon/` subpackage or top-level module, `scripts/*.py` or
   `tools/*.py` is not mentioned in `INDEX.md`;
2. `INDEX.md` mentions a repo path (inline code containing `/` or ending in
   `.py`) that does not exist; or
3. a `src/poseydon/` subpackage has no `docs/index/<pkg>.md`.

`__init__.py` and `__pycache__` are excluded.

**Rule for prose:** a test cannot tell whether a paragraph is still true.
`CLAUDE.md` states that any commit changing behaviour, structure or a contract
under `src/` updates the matching `docs/index/` page in the same commit, and
that a page found disagreeing with the code is fixed toward the code.

A per-page "last verified at commit" stamp was considered and deferred. Add it
if prose drift shows up in practice.

## Viewing

Served on the host with `uvx`, which runs MkDocs Material in a throwaway
environment:

```
uvx --with mkdocs-material mkdocs serve    # http://127.0.0.1:8000, live reload
```

This adds no `pyproject.toml` dependency and changes no image: the docs need
nothing from `poseydon` itself, so they have no reason to go through compose.
The command is recorded in `docs/README.md` and `CLAUDE.md`. Material renders
Mermaid via `pymdownx.superfences`, which it ships with. Nav: Index (INDEX,
overview, package pages), Guides, Specs, Plans.

## Writing the content

Each page is drafted by reading that package's modules and docstrings, the
specs that touch it and `git log` for its files. Rationale is quoted or linked,
never paraphrased into stronger claims than the source makes.

## Testing

- `tests/test_docs_index.py` passes on the finished tree, and fails when a
  subpackage is added without an index entry (checked once by hand by adding a
  temporary empty package).
- `uvx --with mkdocs-material mkdocs build --strict` succeeds, so there are no
  broken nav entries or links.
