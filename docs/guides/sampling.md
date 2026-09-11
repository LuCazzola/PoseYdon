# Sampling, and rebuilding a skeleton

Everything after the flags is a Hydra override into `configs/sample.yaml`,
which selects the operation, the sampler, and how the skeleton is rebuilt
from the features a model produces.

```bash
poseydon sample \
    --checkpoint runs/**/last.ckpt \
    --render --zoom 2.0 --out samples \
    skeleton=Scorpion n_samples=4 n_frames=40 \
    sampler.steps=50 reconstruct.name=positions_ik
```

Everything after the flags is a Hydra override into `configs/sample.yaml`, which
selects the operation (`generate`, `inbetween`, `blend`), the sampler (`ddpm`,
`ddim`, `euler`), and how the skeleton is rebuilt.

The code path, in order:

| step | where |
|---|---|
| parse flags, compose config | `src/poseydon/cli.py::_sample` |
| build dataset, model, process | `src/poseydon/training/build.py` |
| pick the verb | `src/poseydon/ops/` (`generate`, `inbetween`, `blend`) |
| solve the trajectory | `src/poseydon/sampling/` (`ddpm`, `ddim`, `euler`) |
| per-step edits | `src/poseydon/sampling/controls/` |
| features back to a skeleton | `src/poseydon/features/reconstruct.py` |
| write BVH / MP4 | `src/poseydon/io/bvh.py`, `src/poseydon/io/render.py` |

One real batch supplies the skeleton's conditioning -- topology, rest pose,
normalization statistics all describe the rig rather than the clip -- so
`skeleton=` picks which character to animate.

### Choosing how the skeleton is rebuilt

The representation is redundant: it carries rotations AND positions, and a
generated clip need not keep them consistent. On real data they agree to 1e-9;
on model output they differ by more than three bone lengths. So this is a real
choice, not a formality:

```yaml
reconstruct:
  name: fk | positions | positions_ik
  iterations: 150     # positions_ik only
  smoothness: 0.05
```

| method | bone lengths | notes |
|---|---|---|
| `fk` | exact | forward kinematics from rotations. Free, rigid |
| `positions` | **not enforced** | the prediction itself; limbs stretch. No BVH, since a BVH stores rotations |
| `positions_ik` | exact | positions fitted back onto a rigid skeleton by IK |

`positions_ik` is approximate by nature: rotation about a bone's own axis does
not move its children, so twist is invisible to a position-fitting solver. It
plateaus near 1% of a bone length mean error rather than reaching zero.

### Rendering

`--render` writes an MP4 per sample; `--zoom` frames it (values above 1 move the
camera closer). Rendering is an optional extra: `pip install poseydon[render]`.

