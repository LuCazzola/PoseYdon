# Design rationale

This guide is for a person supervising the system. It explains how the system is
structured and why, so you can compare it with your own picture. Noticing "that is not
how I imagined it" is the point: it may be a stale page, a bug, a design flaw, or a
misunderstanding, and each of those is worth finding.

Most reasons here are quoted from the module docstrings that record them, and each
page names its code. A reason nobody has written down is marked as inferred.

| Page | The question it answers |
|---|---|
| [The round trip](round-trip.md) | How does a user's rig become training data, and how does generated motion get back onto that rig? |
| [Motion representation](representation.md) | What does the model see of a clip, and how is it turned back into a skeleton? |
| [Generation](generation.md) | How are processes, models, samplers, controls and operations kept apart? |
| [Configuration](configuration.md) | Why is every component named in config, and where do per-character facts live? |
| [Verification](verification.md) | How do the checks avoid testing the pipeline against itself? |

## Two sides: library and tooling

`src/poseydon/` is the library and owns every contract: file formats, containers,
the dataset-to-model batch and the registries. `scripts/`, `tools/`, `configs/`,
`tests/` and `data/` are tooling that drives the library and must not define contracts
of their own. This split was stated by the project owner (in `CLAUDE.md`).

The repo is meant to be foundational: new rigs, datasets, representations, models and
processes keep arriving. So what a character or dataset needs lives in data and configs,
not in code. That is also the owner's stated reason (in `CLAUDE.md`).
