# Documents

Three kinds.

**Guides** (`guides/`) are how to use the thing. Start here if you want to run
something.

| | |
|---|---|
| [Quick start](guides/quickstart.md) | ingest a corpus, train, sample |
| [Architecture](guides/architecture.md) | how the pieces fit, and why they are separate |
| [Sampling](guides/sampling.md) | samplers, controls, rebuilding a skeleton |
| [Retargeting](guides/retargeting.md) | cross-topology transfer and its validation |
| [Testing](guides/testing.md) | what is covered, and how coverage is checked |

The other two are why it looks the way it does, and the difference between them
matters.

**Specs** (`superpowers/specs/`) argue a design. They state the problem, the options
considered, the measurements that decided between them, and the resulting
contract. They are written before the code and kept afterwards, because the
measurement is the expensive part — anyone can read the code to see *what* it
does, but not *why* the alternative was rejected.

**Plans** (`superpowers/plans/`) break one spec into tasks small enough to review
independently. Each task names the files it touches, the interfaces it produces
for later tasks, and the tests that prove it. They are a record of how the work
was actually sequenced.

## Reading them

Start with the spec for the area you are changing. If it disagrees with the
code, the code is right and the spec has drifted — say so in your change rather
than quietly matching one to the other, because a spec nobody trusts is worse
than no spec.

| area | spec |
|---|---|
| features, normalization, parity with the reference | [2026-09-06-data-pipeline-and-training-parity-design](superpowers/specs/2026-09-06-data-pipeline-and-training-parity-design.md) |
| the training recipe, retarget validation | [2026-09-08-training-run-and-retarget-validation-design](superpowers/specs/2026-09-08-training-run-and-retarget-validation-design.md) |

## What is deliberately not here

API documentation. The contracts live in the code — `core/spec.py`,
`core/batch.py`, `models/base.py`, `sampling/base.py` — with docstrings that
explain the reasoning at the point of use. A separate API document would be a
second source of truth with no mechanism keeping it honest.

Nor is there a changelog. `git log` carries the reasoning: commit messages here
record what was measured and what was rejected, not just what changed.
