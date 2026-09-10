# Retarget Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retarget a clip from one rig onto another, and watch it happen during training — three fixed pairs every 25 000 steps, emitting `.npz`, `.bvh` and `.mp4` to wandb as artifacts, with five scalars per pair.

**Architecture:** The mechanism is already latent in the codebase, so this is four thin pieces rather than a subsystem. `MoDiffAE`'s semantic latent is pooled over joints, so a 40-joint Flamingo latent is shape-compatible with a 63-joint Scorpion decode by construction — only the frame count must match. A `LatentPin` control injects that latent on every sampling step; a `Retarget` operation encodes once and delegates to `Operation.run`; one `run_retarget()` serves both a new CLI verb and the training callback, so what is watched is what ships.

**Tech Stack:** PyTorch 2.14+cu130, Lightning 2.6, Hydra, wandb, numpy.

**Spec:** `docs/superpowers/specs/2026-09-08-training-run-and-retarget-validation-design.md` — §6 (`Retarget`), §7 (`RetargetValidation` and its metrics), §8 tests 11–15.

**Context:** Plan A is merged. A 600 000-step run is live (`poseydon-train-600k`, wandb `8eb9dd7z`) with no validation callback; it will be restarted onto this once the plan lands.

## Global Constraints

- **Everything runs and is tested in Docker.** Never `pytest` or `python` on the host. CPU suite: `docker compose run --rm test bash -c "ruff check . && pytest -q"`. GPU: `docker compose run --rm train ...`.
- **`bash -c`, never `bash -lc`.** A login shell drops `/opt/venv/bin` from `PATH`.
- **`ruff check` is the lint gate.** `ruff format` is NOT this project's convention — it would reformat most of the tree. Do not run it.
- **NEVER use `git stash` (including `-u`).** The tree holds an untracked 1145-file corpus a stash would sweep. Use `git worktree add` if you need a clean tree.
- **Do not touch `data/truebones/source/`** (paid, irreplaceable). Do not rebuild the corpus; `--stats-only` (10 s) is allowed if a normalization change ever demands it, but nothing here should.
- **A training job may be running from this working tree via a bind mount.** Do not check out another branch and do not stop the container. Adding new files is safe; if you must mutate existing source to test something, use a worktree.
- **Measured baseline:** 371 passed / 8 skipped / 12 xfailed, ruff clean.
- **Validation pairs, exact:** Flamingo/`onelegbent` → Scorpion, Coyote/`attack3` → Crab, Goat/`headbutt` → Raptor. All six rigs and all three content clips exist in the built corpus (verified).
- **Sampling is DDPM**, output length is the content clip's own length, reconstruction is `positions_ik`.
- **A validation failure must never kill a training run.** Losing one observation is cheap; losing 72 hours is not.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/poseydon/sampling/controls/latent_pin.py` (create) | `LatentPin` — inject a fixed semantic latent every step |
| `src/poseydon/ops/retarget.py` (create) | `Retarget` — encode the content clip once, pin it, delegate |
| `src/poseydon/ops/retarget_run.py` (create) | `run_retarget()` — the one code path the CLI and the callback share |
| `src/poseydon/training/metrics.py` (create) | the five retarget scalars |
| `src/poseydon/training/validation.py` (create) | `RetargetValidation` callback and its wandb artifact |
| `src/poseydon/cli.py` (modify) | `poseydon retarget` |
| `configs/train.yaml` (modify) | the `validation:` block |

---

## Task 1: `LatentPin`

**Files:**
- Create: `src/poseydon/sampling/controls/latent_pin.py`
- Test: `tests/sampling/test_latent_pin.py`

**Interfaces:**
- Consumes: `poseydon.sampling.base.Control`, `CONTROLS`, `poseydon.core.batch.Cond`.
- Produces: `LatentPin(latent: torch.Tensor, key: str = "z_sem")`, registered as `"latent_pin"`.

**Context the brief cannot know:** this is `LatentMix`'s sibling minus the schedule, and deliberately **without** `LatentMix`'s `reference.shape == target.shape` guard. That guard is exactly what makes `Blend` unable to express cross-rig transfer today — the whole point here is that the latent comes from a rig with a different joint count. Read `sampling/controls/latent_mix.py` and mirror its shape, not its constraints.

- [ ] **Step 1: Write the failing test**

```python
"""Pinning a semantic latent, so a decode follows an instruction from elsewhere."""

from __future__ import annotations

import torch

from poseydon.core.batch import Cond
from poseydon.sampling.base import CONTROLS
from poseydon.sampling.controls.latent_pin import LatentPin


def test_the_pinned_latent_reaches_cond_every_step():
    latent = torch.randn(1, 1, 16)
    control = LatentPin(latent)

    z_t = torch.zeros(1, 4, 3, 8)
    for step in (999, 500, 0):
        _, cond = control.before_step(z_t, torch.tensor([step]), Cond({}))
        assert torch.equal(cond["z_sem"], latent), f"latent not pinned at step {step}"


def test_it_does_not_mutate_the_cond_it_was_given():
    """Controls compose: one that edits its input in place corrupts the next."""
    latent = torch.randn(1, 1, 16)
    original = Cond({"topology": "untouched"})
    _, returned = LatentPin(latent).before_step(torch.zeros(1, 1, 1, 1), torch.tensor([0]), original)

    assert "z_sem" not in original.payloads, "the caller's Cond was mutated"
    assert returned["topology"] == "untouched", "existing payloads must survive"


def test_it_accepts_a_latent_from_a_different_joint_count():
    """The reason this is not `LatentMix`.

    `LatentMix` guards `reference.shape == target.shape`, which is exactly what
    makes cross-rig transfer inexpressible. The semantic latent is pooled over
    joints, so a 40-joint encode is shape-compatible with a 63-joint decode by
    construction -- refusing that would refuse the entire feature.
    """
    latent = torch.randn(1, 1, 16)          # from a 40-joint rig
    z_t = torch.zeros(1, 63, 13, 40)        # decoding onto 63 joints
    _, cond = LatentPin(latent).before_step(z_t, torch.tensor([10]), Cond({}))
    assert cond["z_sem"].shape == latent.shape


def test_it_is_registered_under_its_name():
    assert CONTROLS.get("latent_pin") is LatentPin
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `docker compose run --rm test bash -c "pytest -q tests/sampling/test_latent_pin.py"`
Expected: `ModuleNotFoundError: No module named 'poseydon.sampling.controls.latent_pin'`

- [ ] **Step 3: Implement**

```python
"""Pinning a semantic latent for the whole trajectory."""

from __future__ import annotations

import torch

from poseydon.core.batch import Cond
from poseydon.sampling.base import CONTROLS, Control


@CONTROLS.register("latent_pin")
class LatentPin(Control):
    """Hold one semantic latent fixed across every sampling step.

    `LatentMix`'s sibling minus the schedule, and deliberately without its
    `reference.shape == target.shape` guard. That guard is what makes `Blend`
    unable to express cross-rig transfer: the semantic latent is pooled over
    joints (`SemanticEncoder.forward` returns `(T, B, 1, C)`), so a 40-joint
    encode is shape-compatible with a 63-joint decode by construction, and
    only the frame count has to match. Refusing a shape mismatch here would
    refuse retargeting itself.
    """

    def __init__(self, latent: torch.Tensor, key: str = "z_sem") -> None:
        self.latent = latent
        self.key = key

    def before_step(
        self, z_t: torch.Tensor, t: torch.Tensor, cond: Cond
    ) -> tuple[torch.Tensor, Cond]:
        # A new Cond, never an edit of the caller's: controls compose, and one
        # that mutates its input corrupts every control after it.
        payloads = dict(cond.payloads)
        payloads[self.key] = self.latent.to(z_t.device)
        return z_t, Cond(payloads)
```

- [ ] **Step 4: Run the tests, then the suite**

Run: `docker compose run --rm test bash -c "ruff check . && pytest -q"`
Expected: baseline + 4.

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/sampling/controls/latent_pin.py tests/sampling/test_latent_pin.py
git commit -m "feat(sampling): LatentPin, a fixed semantic latent for cross-rig decode"
```

---

## Task 2: The `Retarget` operation

**Files:**
- Create: `src/poseydon/ops/retarget.py`
- Test: `tests/ops/test_retarget.py`

**Interfaces:**
- Consumes: `Operation`, `OPERATIONS`, `LatentPin` from Task 1, `Denoiser.encode`.
- Produces: `Retarget(content: torch.Tensor, content_cond: Cond)`, registered `"retarget"`, with `build_controls(cond)` and an overridden `run()`.

**Context the brief cannot know:** `Retarget.run` must call `model.encode(content, content_cond)` **once**, before sampling, then build the `LatentPin` from that latent. It must raise a clear error on a model with no `encode` — that is, `AnyTop` — rather than silently producing unconditional motion, which would look like a working retarget that ignored its content entirely. `MoDiffAE.encode(clean, cond, masks=None, temporal_valid=None)` returns `(z_sem, aux)`.

- [ ] **Step 1: Write the failing test**

```python
"""Cross-topology retarget: the test the whole feature rests on."""

from __future__ import annotations

import pytest
import torch

from poseydon.core.batch import Cond
from poseydon.ops.base import OPERATIONS
from poseydon.ops.retarget import Retarget


class _Latent:
    """A model that encodes to a joint-independent latent, like MoDiffAE."""

    requires = ()

    def __init__(self):
        self.encode_calls = 0
        self.seen_latents = []

    def encode(self, clean, cond, masks=None, temporal_valid=None):
        self.encode_calls += 1
        # Pooled over joints: (T, B, 1, C), independent of the joint count.
        return torch.full((clean.shape[-1], 1, 1, 8), 0.5), {}


def test_the_content_is_encoded_exactly_once(...):
    """Encoding per step would be both wrong and 1000x the cost."""


def test_the_output_carries_the_TARGET_joint_count():
    """Encode a 40-joint clip, decode on a 63-joint rig.

    This is the assertion the feature exists for: the semantic latent is pooled
    over joints, so the decode is free to have a different topology.
    """


def test_a_model_without_encode_is_refused():
    """AnyTop has no semantic encoder. Sampling anyway would produce
    unconditional motion that LOOKS like a retarget while ignoring its content
    entirely -- a silent wrong answer, which is worse than a crash.
    """
    with pytest.raises(TypeError, match="encode"):
        Retarget(torch.zeros(1, 4, 3, 8), Cond({})).run(
            model=object(), process=None, sampler=None, shape=(1, 4, 3, 8), cond=Cond({})
        )


def test_it_is_registered_under_its_name():
    assert OPERATIONS.get("retarget") is Retarget
```

Fill the two elided bodies with a stub sampler that records the `controls` it was handed and returns `torch.zeros(shape)`; assert `model.encode_calls == 1` in the first, and that the returned tensor's joint axis is the target's in the second.

- [ ] **Step 2: Run it to confirm it fails**

Expected: `ModuleNotFoundError: No module named 'poseydon.ops.retarget'`

- [ ] **Step 3: Implement**

```python
"""Retargeting: decode one clip's semantics onto another rig."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from poseydon.core.batch import Cond, Masks
from poseydon.models.base import Denoiser
from poseydon.ops.base import OPERATIONS, Operation
from poseydon.process.base import Process
from poseydon.sampling.base import Control, Sampler
from poseydon.sampling.controls.latent_pin import LatentPin


@OPERATIONS.register("retarget")
class Retarget(Operation):
    """Carry one clip's motion onto a different skeleton.

    The semantic latent is pooled over joints, so it describes WHAT is being
    done without describing what is doing it. Encoding a 40-joint Flamingo and
    decoding on a 63-joint Scorpion is therefore shape-valid by construction;
    only the frame count has to match.
    """

    def __init__(self, content: torch.Tensor, content_cond: Cond) -> None:
        self.content = content
        self.content_cond = content_cond
        self._latent: torch.Tensor | None = None

    def build_controls(self, cond: Cond) -> Sequence[Control]:
        if self._latent is None:
            raise RuntimeError("`run` encodes the content before building controls")
        return [LatentPin(self._latent)]

    def run(
        self,
        model: Denoiser,
        process: Process,
        sampler: Sampler,
        shape: tuple[int, ...],
        cond: Cond,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
        masks: Masks | None = None,
    ) -> torch.Tensor:
        encode = getattr(model, "encode", None)
        if encode is None:
            raise TypeError(
                f"{type(model).__name__} has no `encode`, so there is no semantic "
                "latent to retarget. Sampling anyway would produce unconditional "
                "motion that looks like a retarget while ignoring the content clip."
            )

        # ONCE, before the trajectory: the latent describes the content, which
        # does not change as the target is denoised. Encoding per step would be
        # the same answer computed a thousand times.
        self._latent, _ = encode(self.content.to(device), self.content_cond)
        return super().run(
            model, process, sampler, shape, cond, device, generator, masks
        )
```

- [ ] **Step 4: Run the tests, then the suite**

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/ops/retarget.py tests/ops/test_retarget.py
git commit -m "feat(ops): Retarget, decoding one clip's semantics onto another rig"
```

---

## Task 3: `run_retarget` and the CLI verb

**Files:**
- Create: `src/poseydon/ops/retarget_run.py`
- Modify: `src/poseydon/cli.py`
- Test: `tests/ops/test_retarget_run.py`

**Interfaces:**
- Consumes: `Retarget`, `MotionDataset`, `CorpusIndex.query`, `reconstruct`, `BVH.write`, `io/render.py::render_skeleton`, `training/recipe.py`.
- Produces:
  ```python
  def run_retarget(
      model, process, sampler, dataset, content: ClipRecord, target_rig: str,
      out_dir: Path, device="cpu", generator=None,
  ) -> RetargetResult
  ```
  where `RetargetResult` carries `npz`, `bvh`, `mp4` paths plus the arrays the metrics need (`predicted_positions`, `solved_positions`, `latent`, `target_anim`, `content_anim`).

**Context the brief cannot know:** this function is the whole point of the task — the CLI and the training callback must both call it, so what is watched during training is the same code path that ships. Do not let the callback grow its own copy.

Naming: `runs/<run>/validation/step_<N>/<content_rig>_<action>__to__<target>.{npz,bvh,mp4}`. Sampling is DDPM; output length is the content clip's own length; reconstruction is `positions_ik` and comes from the recipe, so `sample`, `retarget` and the callback cannot disagree about it.

- [ ] **Step 1: Write the failing test**

Assert: it writes all three files; the `.bvh` parses back with the TARGET rig's joint names and parents; the output frame count equals the content clip's; and the reconstruction method actually used is the one the recipe records (mutate the recipe and confirm the run changes or refuses).

- [ ] **Step 2: Run it to confirm it fails**

- [ ] **Step 3: Implement `run_retarget`**

Structure: resolve the content clip through `dataset`/`CorpusIndex.query`; take one item of the target rig for its conditioning (topology, tpose, norm stats describe the RIG, not the clip — `cli.py::_sample` already does this and is the pattern to follow); build `Retarget(content_features, content_cond)`; sample under `torch.no_grad()`; denormalize; `reconstruct` with the recipe's method; write `.npz`, `.bvh`, `.mp4`.

- [ ] **Step 4: Add the `retarget` CLI verb**

`poseydon retarget --checkpoint <ckpt> --content <rig>/<action> --target <rig> --out <dir>`, reading the recipe beside the checkpoint. Mirror `_sample`'s argument handling.

- [ ] **Step 5: Run the tests, then the suite**

- [ ] **Step 6: Commit**

```bash
git add src/poseydon/ops/retarget_run.py src/poseydon/cli.py tests/ops/test_retarget_run.py
git commit -m "feat(ops): run_retarget, shared by the CLI and the training callback"
```

---

## Task 4: The five metrics

**Files:**
- Create: `src/poseydon/training/metrics.py`
- Test: `tests/training/test_retarget_metrics.py`

**Interfaces:**
- Produces: `foot_skate`, `bone_length_drift`, `ik_residual`, `root_trajectory_error`, `latent_round_trip`, and `retarget_metrics(result, model, ...) -> dict[str, float]`.

**Context the brief cannot know — one subtlety that is the whole reason two of these exist.** **Bone-length drift is measured BEFORE IK, deliberately.** On the solved output it is identically zero by construction, which measures the solver's parameterization rather than the model. Measuring it pre-IK, alongside the IK residual, separates *"the model predicts bad bone lengths"* from *"the solver had to move joints a long way"*. A previous phase's retrospective named that confusion as its recurring bug.

| metric | definition |
|---|---|
| foot skate | horizontal displacement of a foot joint while its contact flag is set, on the SOLVED output |
| bone-length drift | \|‖p_j − p_parent‖ − ‖offset_j‖\| on the RAW predicted positions, in bone-length units |
| IK residual | ‖p_predicted − p_solved‖, mean and max, in bone-length units |
| root trajectory error | generated root XZ velocity vs the content clip's, each rescaled by its own rig's `scale_factor` from `prepare.npz` |
| latent round-trip | distance between `encode(generated)` and the injected `z_sem` |

The target rig's foot joints come from its manifest, remapped through the rig's reduction `JointEdit` by the same `augment/topology.py::_reindex_resolved` path augmentation already uses.

- [ ] **Step 1: Write the failing test (test 14, "metric sanity")**

Each metric is checked against an external definition, not against its own implementation:
- a **static** clip scores ~zero foot skate;
- an **FK-generated** clip (positions produced by forward kinematics from a rigid skeleton) scores ~zero bone-length drift and ~zero IK residual, since it is already exactly consistent;
- a clip translated by a known constant scores a known root trajectory error;
- `latent_round_trip` of a latent against itself is zero.

- [ ] **Step 2: Run it to confirm it fails**

- [ ] **Step 3: Implement**

- [ ] **Step 4: Run the tests, then the suite**

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/training/metrics.py tests/training/test_retarget_metrics.py
git commit -m "feat(training): the five retarget validation metrics"
```

---

## Task 5: The `RetargetValidation` callback

**Files:**
- Create: `src/poseydon/training/validation.py`
- Modify: `configs/train.yaml`, `src/poseydon/training/build.py`, `src/poseydon/cli.py`
- Test: `tests/training/test_retarget_validation.py`

**Interfaces:**
- Consumes: `run_retarget`, `retarget_metrics`, `CorpusIndex.query`.
- Produces: `RetargetValidation(pairs, every_n_steps=25000, run_at_start=True, ...)`, a Lightning `Callback`.

**Config block, exact:**

```yaml
validation:
  every_n_steps: 25000
  run_at_start: true
  pairs:
    - {content: {rig: Flamingo, action: onelegbent}, target: Scorpion}
    - {content: {rig: Coyote,   action: attack3},    target: Crab}
    - {content: {rig: Goat,     action: headbutt},   target: Raptor}
```

**Context the brief cannot know — four properties, each of which has bitten this kind of callback before:**

1. **It must never kill the run.** The whole log-and-upload block is wrapped; a failure degrades to a warning on the logger and a note in the console. Losing one observation is cheap; losing 72 hours to a transient network error is not.
2. **A dedicated `torch.Generator` seeded from the step.** Validation must be reproducible across a resume and must never consume the training RNG stream — otherwise the training trajectory changes depending on whether you resumed.
3. **`eval()`/`train()` around itself, under `no_grad`.** Restoring `train()` matters: leaving the model in `eval()` silently disables dropout for the rest of the run.
4. **wandb artifact, not just video.** All three files per pair go into ONE artifact per firing, named `retarget-<run_id>`, typed `retarget-validation`, aliased with the step (`step-25000`); the MP4 *additionally* as `wandb.Video`. The artifact is the point — a video renders but cannot be diffed or loaded into Blender, and a three-day run is one whose step-25000 output you want to open next week.

- [ ] **Step 1: Write the failing test (test 13, "callback contract")**

Assert, with a stub trainer and a stub logger: it fires at step 0 and at each `every_n_steps` and **not** in between; it writes all three artifact types; it logs five scalars per pair plus the across-pair mean; it restores `train()` mode; and — the one that matters most — **the training RNG stream is untouched**: record `torch.rand(1)` before and after a firing with a fixed global seed and assert the sequence is unchanged.

Also assert that a `run_retarget` which raises does **not** propagate: the callback logs a warning and training continues.

- [ ] **Step 2: Run it to confirm it fails**

- [ ] **Step 3: Implement the callback**

- [ ] **Step 4: Wire it into `build_callbacks` and `configs/train.yaml`**

Gated on the config block being present, so a run without `validation:` behaves exactly as today.

- [ ] **Step 5: Run the tests, then the suite**

- [ ] **Step 6: Prove it end to end on the GPU**

```bash
docker compose run --rm train python -m poseydon.cli train \
  --out /tmp/valsmoke --quiet trainer.max_steps=2 data.num_workers=2 \
  validation.every_n_steps=1
```
Expected: two firings, nine files written, a wandb artifact visible on the run, five scalars per pair. Report the run URL. **Do NOT start the 600k run** — that is mine.

- [ ] **Step 7: Commit**

```bash
git add src/poseydon/training/validation.py src/poseydon/training/build.py configs/train.yaml src/poseydon/cli.py tests/training/test_retarget_validation.py
git commit -m "feat(training): retarget validation every 25k steps, logged as wandb artifacts"
```

---

## Definition of Done

- `docker compose run --rm test bash -c "ruff check . && pytest -q"` green, no fewer passing than the 371 this plan starts from.
- `poseydon retarget` produces a `.bvh` that opens on the target rig.
- A short GPU run fires the callback, writes nine files, and logs an artifact plus five scalars per pair.
- The 600 000-step run is NOT started by this plan.
