# Documents

**Start at [the index](index/INDEX.md)**: where everything lives, split into
the library (`src/poseydon/`) and the tooling around it, with one design page
per package. Browse it all as a site with
`uvx --from 'mkdocs<2' --with mkdocs-material mkdocs serve`.

**Guides** (`guides/`) are how to use the thing. Start here if you want to run
something.

| | |
|---|---|
| [Quick start](guides/quickstart.md) | run the tests, train, sample, export |
| [Architecture](guides/architecture.md) | how the pieces fit, and why they are separate |
| [Dataset layout](guides/truebones-layout.md) | what each prepared file holds |
| [Data round trip](guides/roundtrip.md) | a character in, generated motion back out on that same character |
| [Generating motion](guides/sampling.md) | samplers, controls, rebuilding a skeleton |
| [Retargeting](guides/retargeting.md) | cross-topology transfer and its validation |
| [Testing](guides/testing.md) | what is covered, and how coverage is checked |

## Where the reasoning lives

Why the code looks the way it does is recorded in three places, closest to the
code first:

- **docstrings and comments**, at the point of use, including the
  measurements that decided between alternatives;
- **commit messages**, which record what was measured and what was rejected,
  not just what changed;
- **the index pages**, whose *Why it is shaped this way* sections collect both
  per package and cite them.

If a page disagrees with the code, the code is right. Fix the page in the same
change and say so.

## What is deliberately not here

API documentation. The contracts live in the code — `core/spec.py`,
`core/batch.py`, `models/base.py`, `sampling/base.py` — with docstrings that
explain the reasoning at the point of use. A separate API document would be a
second source of truth with no mechanism keeping it honest.

Nor is there a changelog: `git log` carries it.
