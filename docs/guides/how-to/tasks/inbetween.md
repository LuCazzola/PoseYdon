# In-between

Give the model the start and the end of a motion; it generates the frames in
between.

It needs no special model. At every sampling step the frames you gave are put
back, noised to the current step's level, so the generated middle is shaped
to meet them. Any model trained for [generate](generate.md) can do it, as long
as it was trained with the diffusion process (`process=gaussian`).

## Running it

There is no command for this yet: `poseydon sample` builds its task from the
config, and in-betweening takes a motion as input, which a config can't carry.
From Python:

```python
from poseydon.ops import Inbetween

task = Inbetween.from_endpoints(reference=clip, process=process, head=10, tail=10)
motion = task.run(model, process, sampler, shape=clip.shape, cond=cond)
```

- `clip` is a batch of motion in the model's features, shaped `(B, J, D, T)`
  (see [Notation](../../concepts/data/notation.md#array-shapes)). Only its
  first `head` and last `tail` frames are used.
- `model`, `process`, `sampler` and `cond` are built the same way
  `poseydon sample` builds them; `src/poseydon/cli.py` (`_sample`) is the
  working example.

To keep frames other than the two ends, build
[`Inbetween`][poseydon.ops.Inbetween] directly with a `known` mask of shape
`(B, T)`, `True` for every frame to keep.
