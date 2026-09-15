# Conditioning masking, classifier-free guidance and multi-head modulation — design

Date: 2026-09-15

## Problem

Three gaps, one of them already half-built.

**Classifier-free guidance does not exist.** `sampling/controls/guidance.py`
registers a `cfg` control whose `before_step` writes `cond["uncond"] = scale`
and returns. No model reads that key, so `poseydon sample
operation.guidance_scale=2.5` composes cleanly, runs, and changes nothing. The
reference is no better: `--cond_mask_prob` is parsed
(`utils/parser_util.py:99`), stored into `AnyTop.__init__`
(`model/anytop.py:45`), and never read again. There is nothing to port.

The control cannot be completed as written. Guidance needs two model
evaluations per step, and `Control.before_step(z_t, t, cond)` is never given
the model.

**The semantic latent has no bottleneck.** `SemanticEncoder` sees the whole
clean clip, every frame and every joint, and `MoDiffAE` is trained to denoise
that same clip. Nothing stops the latent carrying a near-copy of its input.
For denoising that is harmless; for retargeting — the task — it is the failure
mode, because a latent that encodes *this rig doing this motion* has nothing
rig-independent left to transfer. `joints_names_dropout` (p=0.1, on the T5
rows) is the only corruption anywhere on the conditioning path.

**Semantic modulation is single-headed.** `ConditionAttention`
(`models/modules/graph_backbone.py:117`) maps the latent to a value and gates
it by a projection of `[joint, latent]`. It is a faithful port of the
reference's `semantic_modulation` strategy, parameter names included, so
published checkpoints load into it. The reference's two other strategies
(`mha`, `film`) were not ported. One gated transform is the whole conditioning
capacity of each decoder layer.

## Scope

**In.** Two sibling modules, `MotionMasker` and `ConditionMasker`, owning the
null embeddings and every masking decision; frame, per-frame-joint and
kinematic-chain masking of the semantic encoder's input; learned null tokens
rather than zeros; CFG training
by latent drop and CFG sampling by a wrapper denoiser; multi-head semantic
modulation with mean or sum aggregation; a `chain_ids` helper in
`core/topology.py`; the recipe escape all of the above forces; tests for the
`models/` package, which has none.

**Out.** Masking on the decoder side — the request is explicitly encoder-side,
and corrupting the denoiser's own input is a different technique with a
different purpose. Text or per-rig conditioning drop; `joint_names` and `tpose`
stay in both CFG branches (see "What CFG drops"). Guidance for `AnyTop`, which
has no semantic latent to drop. Learned or attention-weighted head
aggregation: the aggregation is mean or sum, as requested. Changing any
default, so the next run is bit-identical to the measured one until a knob is
set.

## What CFG drops

The unconditional branch replaces `z_sem` and nothing else. `topology`,
`tpose` and `joint_names` stay in both branches because they say *which rig* is
being animated rather than what it is doing. Dropping them would make the
unconditional branch predict motion for no particular skeleton, and
extrapolating away from that pushes the sample off the target rig's manifold
instead of sharpening its content.

## Architecture

### Two maskers, as siblings

New, `src/poseydon/models/modules/masking.py`. Two small `nn.Module`s, each
owning one null embedding and one job:

```python
class MotionMasker(nn.Module):          # the semantic encoder's INPUT
    null_token: nn.Parameter            # (d_model,)
    frames: float; joints: float; chains: float

    def mask(self, joint_valid, parents, n_frames) -> Tensor | None: ...
    def apply(self, motion, mask) -> Tensor: ...


class ConditionMasker(nn.Module):       # the WHOLE conditioning signal (CFG)
    null_latent: nn.Parameter           # (d_model,)
    cond_drop: float

    def drop(self, z_sem, cond) -> Tensor: ...
```

`MoDiffAE` holds both.

**Why siblings rather than `SkeletonMasker(ConditionMasker)`.** The two do not
specialize each other; they act on different objects. Token masking replaces
`(T, B, J, C)` input tokens and needs `parents` and `joint_valid`; latent drop
replaces a pooled `(T, B, 1, C)` vector and knows nothing about skeletons —
it would apply identically to a text embedding or a style vector. Inheritance
here would be composition in inheritance's clothing, and because `MoDiffAE`
needs both behaviours at once the subclass would always be the one
instantiated, leaving the base an abstraction with one real implementation.

Splitting also simplifies the plumbing rather than adding to it. Only the
motion masker threads down through `SemanticEncoder.forward` →
`InputProcess.forward`; `InputProcess` has no business knowing that CFG
exists. The latent drop stays at the top of `MoDiffAE.forward`, where `z_sem`
is. A single class would have to be passed into `InputProcess` to reach its
token half, carrying the CFG logic with it.

No shared base class. What the two would share is a learned null parameter, a
`self.training` gate and an off-is-a-no-op contract — three lines each. It
extracts itself if a third masker ever appears.

Two null parameters, not one, and not four. `null_token` is shared by all
three masking schemes: a distinct token per scheme would tell the encoder
*why* a slot is absent, which is exactly the information a bottleneck is
supposed to withhold. `null_latent` is separate from it because `z_sem` lives
after `projection_head`, in a different distribution from an input token.

`MotionMasker.mask` returns `None` when nothing is enabled or when
`self.training` is false, so sampling and retargeting never mask and a
fully-off masker is a true no-op rather than a mask of all-false.

The three schemes are three floats and three short methods composed by OR,
not a registry of strategy classes mirroring `augment/`. All three take the
same inputs and return the same `(T, B, J)` bool, so a class boundary buys
little; the registry is the upgrade path if masking schemes multiply.

### Masking the encoder's input

`MotionMasker.mask` composes three independent schemes by OR, matching how
`AugmentPipeline` already treats independent entries:

| scheme | draw |
|---|---|
| `frames` | per-sample rate `~ U(0, v)`, Bernoulli over `(B, T)`, broadcast over joints |
| `joints` | per-sample rate `~ U(0, v)`, Bernoulli over `(B, T, J)` |
| `chains` | per-sample probability `v`; one chain, masked across every frame |

Rates are drawn per sample from `U(0, v)` rather than applied flat. That is the
idiom `DropEndEffector` already uses (`rate: [0.1, 0.2, 0.3]`), and it means
some samples in every batch are seen clean — so an always-on mask does not open
a gap between training and a sampling path that never masks.

A chain is masked across the whole window rather than per frame. Masking a
limb in some frames and not others is per-frame joint masking with extra
steps; the scheme earns its name only if the limb is absent from the
conditioning entirely.

Substitution happens inside `InputProcess.forward`, which gains an optional
`masking=None` taking the `MotionMasker`, and calls its `apply`. The null
token replaces the **motion embedding**, and names and frame positions are
then added on top, BERT-style: the encoder reads "joint 7
at frame 12 is hidden" rather than finding a hole where a token should be. The
decoder passes nothing and is unchanged.

Three invariants:

- **The rest frame is never masked.** Index 0 of the token sequence is the
  T-pose, which describes the rig. It is not motion and masking it would
  corrupt the one thing both branches must agree on.
- **The mask is intersected with `joint_valid`.** Padded joints are already
  excluded from attention; giving them null tokens would be meaningless work
  and would make the count of masked slots depend on padding.
- **The attention masks are untouched.** A masked token is still attended.
  That is the whole reason to learn a token instead of excluding the slot, and
  it is what lets the encoder reason about *where* the hole is.

### Kinematic chains

`core/topology.py` gains `chain_ids(parents) -> (B, J) int64`, beside
`edge_relations` and `hop_distances`. A chain is a maximal run of single-child
joints — a tail, a neck, one toe — cut at the next branch point. Because
`core/kinematics.py:10` guarantees parents are topologically ordered, this is a
forward scan rather than a traversal:

```python
chain_ids[j] = chain_ids[p] if child_count[p] == 1 and p > 0 else fresh_id
```

Chains are then picked uniformly over chains, not over joints.

This shape was chosen by measurement, over the two alternatives, across all 73
rigs (15–183 joints, median 57). The fraction of a rig one firing removes:

| scheme | median | p90 | p99 | max | >half the rig |
|---|---|---|---|---|---|
| subtree below a joint | 4.1% | 20.0% | **97.9%** | **99.5%** | **4.9%** |
| single-child run | 3.9% | 10.9% | 16.2% | 48.3% | 0.0% |
| root-to-leaf path | 13.2% | 21.9% | 29.0% | 51.7% | 0.2% |

Subtree masking removes more than half the skeleton on one draw in twenty,
because any joint just below the root takes everything with it. Single-child
runs are bounded by construction: they cannot express "the whole front-left
leg", and they cannot delete a torso either.

The scan is `J` cheap steps, `J <= 183`, against a measured 0.431 s/step. It
carries a `ponytail:` comment naming a conditioner as the upgrade path if it
ever profiles.

### CFG at training time

`ConditionMasker.drop` draws a per-sample Bernoulli at `cond_drop` and applies
`torch.where(drop, null_latent, z_sem)` after encoding. Dropped samples still
pay for the encode; splitting the batch to skip it is the upgrade path and gets
a `ponytail:` comment.

`drop` also honours the `COND_DROP` conditioning key, and **the key
takes precedence over a supplied `Z_SEM`**. `Retarget` pins its latent through
`LatentPin` → `cond[Z_SEM]`, so a guided retarget's unconditional branch has to
override that pin rather than defer to it. Getting this backwards yields a
"guided" retarget whose two branches are identical and whose guidance is
silently zero.

Known limitation, recorded rather than fixed: with `kl_bottleneck: true` a
dropped sample still contributes `mu`/`logvar` to the KL term. The shipped
config sets it `false`.

### CFG at sampling time

`COND_DROP = "cond_drop"` joins `CLEAN_MOTION` and `TEMPORAL_VALID` in
`models/base.py` as a contract key: a bool `(B,)` tensor or scalar. A model
that ignores it makes guidance a silent no-op, which is what happens today.

`src/poseydon/sampling/guidance.py` replaces
`sampling/controls/guidance.py`:

```python
class GuidedDenoiser(Denoiser):
    def forward(self, z_t, t, cond, masks):
        c = self.inner(z_t, t, cond, masks)
        u = self.inner(z_t, t, Cond({**cond.payloads, COND_DROP: True}), masks)
        return Prediction(u.out + self.scale * (c.out - u.out), c.aux)
```

`prepare`, `restore` and `encode` delegate to the inner model, so `Retarget`'s
pre-sampling `encode` and any latent model's `prepare` still work through the
wrapper.

A wrapper rather than the two alternatives. Putting the combination inside
`MoDiffAE.forward` is the shape of the mistake this codebase's docstrings
already call out — the reference's blending lived in `forward`, which is why
in-betweening was unreachable there — and every future model would
reimplement it. Extending `Control` with a model hook adds a method to an ABC
that both samplers and every control must respect, for one user.

`Operation` gains `wrap_model(model)`, identity by default, applied on one line
in `Operation.run` where the model already arrives. `Generate.wrap_model`
returns a `GuidedDenoiser` when `guidance_scale > 0`; `Retarget` inherits it.
`Control` and `Sampler` are untouched. `configs/sample.yaml`'s existing
`guidance_scale: 0.0` becomes functional for the first time.

### Multi-head semantic modulation

`ConditionAttention(d_model, dropout, n_heads=1, aggregate="mean")`.

At `n_heads=1` it builds `query_projector`, `gate_gen`, `value_map` and
`out_proj` exactly as today — same names, same shapes — so the existing 150k
checkpoint and the reference's published ones still load. Above 1 it builds an
`nn.ModuleList` of H × (`query_projector`, `value_map`) over a shared
`out_proj`. Each head reads the full `x` and the full latent and produces a
full-width `d_model` gated value; heads are reduced by `mean` or `sum`.

Heads are independent full-width transforms of one latent, not channel
subspaces. Channel-slice heads would have to be concatenated to be shape-valid,
and mean-or-sum aggregation was the request; the existing gate is also already
elementwise per channel, so subspace heads would restrict what each gate sees
rather than add capacity.

`mean` is the default: `sum` scales the residual branch by H at initialization,
`mean` leaves it comparable to `n_heads=1`.

Cost is `4 * d_model**2` per head, about 65k parameters at `d_model=128`. At
`n_heads=4` that is **+786k on a 4.1M model, +19%**, because the modulation
sits in all four decoder layers. The encoder is unaffected: it builds its
layers with `cross=False` and has no `ConditionAttention` at all.

## Configuration

`configs/model/modiffae.yaml`, mirroring the class:

```yaml
masking:           # MotionMasker
  frames: 0.0      # per-sample rate ~ U(0, v) of frames replaced by the null token
  joints: 0.0      # per-sample rate ~ U(0, v) of (frame, joint) slots
  chains: 0.0      # per-sample probability that one kinematic chain is masked
cond_drop: 0.0     # ConditionMasker: probability the whole latent becomes null
n_modulation_heads: 1
modulation_aggregate: mean
```

The config mirrors the two classes: a `masking:` block for the three
input-masking rates, and `cond_drop` as one float beside it because the whole
of `ConditionMasker` is that one number.

Plain floats with no `_target_`: `MoDiffAE.__init__` takes `masking: dict |
None = None` and builds `MotionMasker(d_model, **(masking or {}))`. Hydra's
recursive `instantiate` would otherwise construct the masker before
`MoDiffAE`, without a `d_model` to size its parameters, forcing the width to
be duplicated in the config.

Everything defaults off. The next run is bit-identical to the run the gradient
measurements were taken on until a knob is set, and the existing checkpoint
still loads.

## The recipe migration this forces

`"model"` is in `CHECKED`, and `_rows` (`training/recipe.py:148`) compares the
block key by key with `ABSENT` standing in for a key one side lacks. So adding
any key to `configs/model/modiffae.yaml` makes **every existing recipe** fail
`check_recipe`:

```
model.cond_drop:
  trained with: '<absent>'
  configured:   0.0
```

That would break `poseydon sample` and `RetargetValidation` for every
checkpoint already on disk.

The fix is the escape `training/recipe.py:51` explicitly anticipates: in
`check_recipe`, skip a row whose **recorded** side is `ABSENT`. A recipe
written before a key existed has nothing to say about that key. Scoped to the
recorded side only — a key the *active* config lacks still raises, because that
is a config that has drifted from the one the checkpoint needs.

What it costs: an old checkpoint sampled with `n_modulation_heads=4` now passes
the recipe check and fails in `load_state_dict` with a shape mismatch instead.
Loud either way, from torch rather than from the guard.

A nested `masking:` block also degrades one error message: `_rows` recurses a
single level, so a disagreement inside the block prints the whole dict rather
than `model.masking.frames: 0.1 vs 0.0`. Accepted in exchange for a config that
mirrors the class that owns it.

## Testing

`tests/models/` is new. The `models/` package has no tests at all today, and
reference-checkpoint loading is exercised only by `tools/reference/*` probes.

**Masking.** The null token lands at masked slots and nowhere else; name and
position embeddings still arrive at a masked slot; the rest frame is never
masked; padded joints are never masked; `model.eval()` masks nothing; an
all-zero masker leaves the tokens bit-identical.

**`chain_ids`.** On a hand-built quadruped, the tail is one id and the spine is
cut at the shoulder branch. Plus a corpus-gated check that no chain exceeds 50%
of its rig across the 73 real rigs — the measurement this design rests on, so
it is the measurement that should fail if a rig is added that breaks it.

**CFG.** `cond_drop=1.0` makes the output independent of the clean motion;
`GuidedDenoiser` at scale 0 reproduces the conditional prediction exactly;
`COND_DROP` beats a supplied `Z_SEM`; the wrapper delegates `encode` to the
inner model.

**Modulation.** `n_heads=1` is numerically identical to the current module and
the current state dict loads into it; `sum` over H identical heads equals H ×
one head; `mean` over H identical heads equals one head.

**Recipe.** A recipe missing a key the active config has passes; an active
config missing a key the recipe has still raises.

## Files

| file | change |
|---|---|
| `src/poseydon/models/modules/masking.py` | new — `MotionMasker`, `ConditionMasker` |
| `src/poseydon/core/topology.py` | `chain_ids` |
| `src/poseydon/models/base.py` | `COND_DROP` contract key |
| `src/poseydon/models/modiffae.py` | owns both maskers; `masking`/head config; `InputProcess` and `SemanticEncoder` take `masking=None` |
| `src/poseydon/models/modules/graph_backbone.py` | multi-head `ConditionAttention`, layer plumbing |
| `src/poseydon/sampling/guidance.py` | new — `GuidedDenoiser` |
| `src/poseydon/sampling/controls/guidance.py` | deleted |
| `src/poseydon/sampling/controls/__init__.py` | drop the export |
| `src/poseydon/ops/base.py` | `wrap_model`, applied in `run` |
| `src/poseydon/ops/generate.py` | `wrap_model` instead of a control |
| `src/poseydon/training/recipe.py` | recorded-side `ABSENT` escape |
| `configs/model/modiffae.yaml` | six keys, all off |
| `tests/models/` | new — masking, CFG, modulation |
| `tests/core/test_topology.py` | `chain_ids` |
| `tests/training/` | the recipe escape |
| `tests/sampling/` | `GuidedDenoiser` |
