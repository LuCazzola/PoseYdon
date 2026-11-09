# Agent Guide

**Hard rule: the code is the oracle, docs follow, and neither implies correctness.**
A claim here is a lead. Check it in the code before relying on it, and if the two
disagree, the page is what changes.

This section is operative: it holds what the code cannot tell you quickly.

| Page | Use it to... |
|---|---|
| [Map](map.md) | find where each piece of code lives. |
| [Recipes](recipes.md) | add something, one recipe per kind of thing. |
| [Testing](testing.md) | run the checks. |
| [Glossary](glossary.md) | look up a term or convention. |

## Keeping the docs true

Update docs in the same change as the code, and the map when files move. Point to code
rather than paraphrasing it: if reading the code would be as fast, the text should not
exist.
