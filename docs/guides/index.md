# Guides

PoseYdon learns to generate and edit motion for animated characters of any
shape: a dog, a scorpion, a flamingo. You hand it a character as its vendor
authored it, and generated motion comes back out **on that same character**,
ready to open in Blender or a game engine.

These guides are written to be read, in roughly this order:

| section | read it when | pages |
|---|---|---|
| **Tutorials** | you are new and want something running | [Setting up](tutorials/setup.md), [Building the dataset](tutorials/build-a-corpus.md), [Train, generate, export](tutorials/train-and-generate.md) |
| **Data** | you want to know what a file, an array or a word means | [Notation and conventions](data/notation.md), [Dataset layout](data/layout.md), [The round trip](data/roundtrip.md) |
| **Using a model** | you have a checkpoint and want more control over its output | [Generating motion](using/sampling.md), [Retargeting](using/retargeting.md) |
| **Design** | you want to know why the code looks the way it does | [Design rationale](design/rationale.md), [Testing](design/testing.md) |

The guides explain. They don't list every file: for "where does X live", go
to the [index](../index/INDEX.md), which has one page per library package.
If a guide disagrees with the code, the code is right.
