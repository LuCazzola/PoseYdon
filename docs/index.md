# PoseYdon

<div align="center" markdown>
![PoseYdon logo](assets/logo_poseydon.png){ width="560" }
</div>

PoseYdon learns to generate and edit motion for animated characters of any
shape: a dog, a scorpion, a flamingo. You hand it a character as its vendor
authored it, and generated motion comes back out **on that same character**,
ready to open in Blender or a game engine.

New here? Read the [Glossary](reference/glossary.md), then do the three
[tutorials](how-to/tutorials/01-setup.md).

## Map of these docs

| Section | Read it when you want to... |
|---|---|
| **User Guide** | understand how the repo is organised and why. |
| [Design rationale](user-guide/index.md) | see the few rules the whole codebase follows, and the problem each one solves. |
| [Overview](user-guide/overview.md) | follow the pipeline through the library, package by package. |
| [The round trip](user-guide/roundtrip.md) | see how a character goes in and comes back out unchanged. |
| **How-to** | get something done; these pages are mostly commands to type. |
| [Tutorials](how-to/tutorials/01-setup.md) | go from a fresh clone to generated motion on a character, step by step. |
| [Tasks](how-to/tasks/index.md) | use a trained model: [generate](how-to/tasks/generate.md), [in-between](how-to/tasks/inbetween.md), [blend](how-to/tasks/blend.md), [retarget](how-to/tasks/retarget.md). |
| [Adding to PoseYdon](how-to/adding.md) | add something: a rig, a dataset, a feature block, a model, a process, a loss, an operation. One recipe each. |
| [Testing](how-to/testing.md) | run the tests, and learn what they can't see. |
| **Index** | look something up. |
| [Map](reference/map.md) | find where a piece of code lives, split into the library (`src/poseydon/`) and the tooling around it. |
| [Glossary and conventions](reference/glossary.md) | look up a term, an array shape or an axis. |
| [Data formats](reference/data-formats.md) | read or write `data/`: what each file on disk holds. |
| [Tooling](reference/tooling.md) | run a script, a probe or a container, or find a config group. |
| [Issues & Ideas](reference/ideas.md) | see bugs noticed but not yet fixed (GitHub Issues) and ideas worth keeping. |
| **API** | read one page per library package: what it owns, its limitations, a class diagram, and the reference generated from the docstrings. [The API index](api/index.md) says how to read them. |

## Where the reasoning lives

Why the code looks the way it does is recorded in three places, closest to the
code first:

- **docstrings and comments**, at the point of use, including the
  measurements that decided between alternatives;
- **commit messages**, which record what was measured and what was rejected,
  not just what changed;
- **the API package pages**, whose introductions say what each package owns
  and what it connects to, and whose *Limitations* list what is bug-prone and
  not guarded by code. What is known to be broken or unfinished is in
  [Issues & Ideas](reference/ideas.md).

If a page disagrees with the code, the code is right. Fix the page in the same
change and say so.

## What is deliberately not here

A changelog: `git log` carries it.

To browse these docs from a clone, run `docker compose up -d docs` and open
port 8019 on your machine, or without Docker run
`uvx --from 'mkdocs<2' --with mkdocs-material --with 'mkdocstrings[python]' mkdocs serve`.
