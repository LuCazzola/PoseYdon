# Blend

Start as one motion and turn smoothly into another.

The model describes *what* a clip is doing with a **semantic latent**, a
summary that leaves out which skeleton is doing it. Blending moves that latent
from the first clip's to the second's over the length of the output, so the
motion changes in meaning, not just by averaging poses. It needs a model that
has such a latent (`modiffae`).

How fast it turns is a **schedule**:

| schedule | the mix over time |
|---|---|
| `ease` (default) | slow at both ends, fastest in the middle; `slope=` sharpens or softens it |
| `linear` | at a constant rate |
| `static` | one fixed mix throughout (`alpha=0.5` is half and half) |

## Running it

There is no command for this yet: `poseydon sample` builds its task from the
config, and blending takes two motions as input, which a config can't carry.
From Python:

```python
from poseydon.ops import Blend

z_from, _ = model.encode(clip_from, cond_from)   # (B, T, C)
z_to, _ = model.encode(clip_to, cond_to)
task = Blend(reference=z_from, target=z_to, schedule="ease", length=z_from.shape[-2])
motion = task.run(model, process, sampler, shape=shape, cond=cond)
```

- `z_from` and `z_to` are the two clips' semantic latents, one per frame, and
  must have the same shape.
- Pass `length=` as the number of frames: without it, the schedule is sized
  from the batch axis and fails unless the batch size equals the frame count
  (listed in [Open work](../../../index/todo.md)).
- `model`, `process`, `sampler` and `cond` are built the same way
  `poseydon sample` builds them; `src/poseydon/cli.py` (`_sample`) is the
  working example.

See [`Blend`][poseydon.ops.Blend] for the full signature.
