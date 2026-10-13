# Motion tasks

A trained model is not used for one thing. PoseYdon is built to solve four
**tasks** with the same model, and in the code each one is an `Operation`
registered by name in `src/poseydon/ops/`:

| task | what it does | needs | run it with |
|---|---|---|---|
| [**generate**](generate.md) | new motion for a character, from noise | any model | `poseydon sample` |
| [**in-between**](inbetween.md) | fills the gap between frames you give it | the diffusion process (`process=gaussian`) | Python |
| [**blend**](blend.md) | cross-fades from one clip into another | a model with a semantic latent (`modiffae`) | Python |
| [**retarget**](retarget.md) | performs one character's clip on a different skeleton | a model with a semantic latent (`modiffae`) | `poseydon retarget` |

None of them needs a model trained specially for it. A task is a small object
that says how to steer the same sampling loop, which is why one checkpoint can
do all four (see
[Design rationale](../../concepts/rationale.md#sampler-control-and-operation-are-three-separate-things)).

## What every task shares

### The sampler

Every task walks from noise to motion with a **sampler**, chosen in
`configs/sample.yaml` under `sampler:`:

| sampler | for |
|---|---|
| [`DDPM`][poseydon.sampling.DDPM] | the diffusion process; random at every step, and slow |
| [`DDIM`][poseydon.sampling.DDIM] | the diffusion process; deterministic, and can skip steps (`sampler.steps=50`). The default |
| [`Euler`][poseydon.sampling.Euler] | the flow-matching process (`process=flow`) |

### The guidance scale

`generate` and `retarget` take a `guidance_scale` (classifier-free guidance),
in the standard form `uncond + w * (cond - uncond)`:

- `1.0` is the model as trained, and the default. It costs nothing extra.
- `0.0` is the unconditional branch: the conditioning is ignored. It is not a
  way of saying "off".
- between 0 and 1 weakens the conditioning; above 1 exaggerates it.

Any value but `1.0` runs the model twice per step and needs a checkpoint
trained with `model.cond_drop > 0`.

### Rebuilding the skeleton

The model's output describes each joint twice, by its rotation and by its
position. On real data the two agree; on generated motion they can disagree
by more than three bone lengths. So turning the output back into a skeleton is
a real choice, set under `reconstruct:`:

| method | bone lengths | what you get |
|---|---|---|
| `fk` (default) | exact | the rotations, applied to the rig. Rigid and free |
| `positions` | **not kept** | the predicted positions as they are; limbs can stretch, and there is no BVH (a BVH stores rotations) |
| `positions_ik` | exact | the predicted positions, fitted back onto the rigid rig by inverse kinematics |

`positions_ik` can't recover twist (spinning a bone about its own axis moves
nothing, so a position fit can't see it). It levels off around 1% of a bone
length on average rather than reaching zero.

### The output

Each result is written as a `.anim.npz` (the motion exactly as generated),
a `.bvh`, and with `--render` an `.mp4` stick-figure video. To put it on the
real character, export the `.anim.npz` as in
[Train, generate, export](../tutorials/train-and-generate.md#export-onto-the-real-character).
