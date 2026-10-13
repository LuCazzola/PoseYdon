# PoseYdon

<p align="center">
  <img src="assets/logo_poseydon.png" alt="PoseYdon" width="560">
</p>

PoseYdon learns to generate and edit motion for animated characters of any
shape: a dog, a scorpion, a flamingo. You hand it a character as its vendor
authored it, and generated motion comes back out **on that same character**,
ready to open in Blender or a game engine.

Read whichever part matches what you're after.

## Concepts: how it fits together

For reading, not running: how the repository is organised, and why.

| | |
|---|---|
| **Design rationale** | [Design rationale](guides/concepts/rationale.md): the few rules the whole codebase follows, and the problem each one solves |
| **Data** | [Notation and conventions](guides/concepts/data/notation.md): the words, shapes and axes everything else uses |
| | [Dataset layout](guides/concepts/data/layout.md): what each file on disk holds |
| | [The round trip](guides/concepts/data/roundtrip.md): how a character goes in and comes back out unchanged |

## How-to: doing things

Commands to type, in the order you'll need them.

| | |
|---|---|
| **Tutorials** | [1. Setting up](guides/how-to/tutorials/setup.md), [2. Building the dataset](guides/how-to/tutorials/build-a-corpus.md), [3. Train, generate, export](guides/how-to/tutorials/train-and-generate.md) |
| **Tasks** | [what a trained model can do](guides/how-to/tasks/overview.md): [generate](guides/how-to/tasks/generate.md), [in-between](guides/how-to/tasks/inbetween.md), [blend](guides/how-to/tasks/blend.md), [retarget](guides/how-to/tasks/retarget.md) |
| **Testing** | [what the tests check, how to run them, and what they can't see](guides/how-to/testing.md) |

New here? Read [Notation](guides/concepts/data/notation.md), then do the
three tutorials.

## Index: where everything lives

[The index](index/INDEX.md) maps every path in the repository, split into the
library (`src/poseydon/`) and the tooling around it. Each library package is
explained on its page in the **API** section.

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
  [open work](index/todo.md).

If a page disagrees with the code, the code is right. Fix the page in the same
change and say so.

## API reference

One page per library package and subpackage. Each opens with a short
introduction and its limitations, then a class diagram where the classes form
a design ([the API index](api/index.md) says how to read them), then the
reference, generated from the docstrings each time the site builds so it can't
drift from the code. Names starting with an underscore are left out.
A guide that names a class links to its entry there.

## What is deliberately not here

A changelog: `git log` carries it.

Browse these docs as a site with `docker compose up -d docs`
(http://127.0.0.1:8019; it stays up until `docker compose stop docs`), or
without Docker with
`uvx --from 'mkdocs<2' --with mkdocs-material --with 'mkdocstrings[python]' mkdocs serve`.
