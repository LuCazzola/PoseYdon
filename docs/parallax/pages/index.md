# PoseYdon

Research infrastructure for neural motion (the `poseydon` package): generation,
in-betweening, blending and cross-topology retargeting on characters of any shape,
taking a vendor's rig in and giving motion back on that same rig. It is written for
the people who build, train and supervise it, and for the agents working on it.

The repo has two sides. The **library** (`src/poseydon/`) owns every contract. The
**tooling** around it (`scripts/`, `tools/`, `configs/`, `tests/`, `data/`) prepares
the data, runs the library and checks its output.

**The code is the oracle, docs follow, and neither implies correctness.** These docs
describe what the code does. They make wrong behaviour easier to spot; they do not
prove anything is right.

## Map of these docs

| Section | Read it when you want to... |
|---|---|
| [User Guide](user-guide/index.md) | supervise the system: understand how it is built and why, and check that against your own picture of it. |
| [Agent Guide](agent-guide/index.md) | get something done in the repo: find code, run things, add things. |
| [Tickets](tickets/index.md) | keep a bug or idea for later, or see what was kept. |
| [API](api/poseydon/index.md) | read a package's docstrings, one page per package. |
