# Conditioning masking, CFG and multi-head modulation — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `MoDiffAE` a masked-conditioning bottleneck and real classifier-free guidance, make its semantic modulation multi-head, and start a training run using all three.

**Architecture:** Two sibling `nn.Module`s own the learned null embeddings — `MotionMasker` hides frames, joints and kinematic chains from the semantic encoder's input, `ConditionMasker` hides the whole latent for CFG. Guidance's two forward passes live in a `GuidedDenoiser` wrapper rather than in any model's `forward` or in a `Control`, which is never handed the model. `ConditionAttention` gains independent full-width heads reduced by mean or sum, and stays byte-identical at one head so existing checkpoints load.

**Tech Stack:** PyTorch 2.x, PyTorch Lightning, Hydra/OmegaConf, pytest, ruff, Docker Compose (`test` service for the suite, `train` service for the GPU run), wandb.

**Spec:** `docs/superpowers/specs/2026-09-15-conditioning-masking-and-cfg-design.md`

## Global Constraints

- **Never edit `external/neural_motion_blending/`.** It is a read-only reference checkout.
- Every command runs in Docker. The suite is `docker compose run --rm test pytest`; lint is `docker compose run --rm test ruff check src tests`. There is no local numpy or torch.
- **Defaults stay off.** `masking.frames/joints/chains: 0.0`, `cond_drop: 0.0`, `n_modulation_heads: 1`, `modulation_aggregate: mean`. A default-config run must remain bit-identical to the run the gradient measurements were taken on, and the existing `runs/modiffae-attn5-b16-d128/checkpoints/*.ckpt` must still load. The new settings are applied as command-line overrides on the launch, which the recipe records.
- **`n_modulation_heads=1` must be numerically identical to today's `ConditionAttention`, with the same state-dict keys** (`query_projector.*`, `value_map.*`, `out_proj.*`, no `heads.*`).
- **`COND_DROP` takes precedence over a supplied `Z_SEM`.** Applied after the latent is read, never before.
- The rest frame (token index 0) is never masked; padded joints are never masked; attention masks are never touched by masking.
- Tests must pass on a checkout with no Truebones corpus: anything needing `data/truebones` uses the `corpus_root` fixture from `tests/conftest.py`, which skips.
- `poseydon train` must be launched with `--out runs/masked-cfg`. `run_name` does not change with these keys, so the default `runs/` would target `runs/modiffae-attn5-b16-d128/` and `ModelCheckpoint(save_last=True)` would overwrite the 150k run's `last.ckpt` and `recipe.yaml`.

## Already on disk

A premature start left four pieces in the working tree. Task 1 tests them as they
stand; do not rewrite them unless a test fails.

- `src/poseydon/models/modules/masking.py` — `chain_ids`, `one_chain_per_sample`, `MotionMasker`, `ConditionMasker`.
- `src/poseydon/sampling/guidance.py` — `GuidedDenoiser`.
- `src/poseydon/models/base.py` — `COND_DROP = "cond_drop"` with its docstring.
- `src/poseydon/sampling/controls/guidance.py` deleted; `ClassifierFreeGuidance` dropped from `sampling/controls/__init__.py`.

**`src/poseydon/ops/generate.py` still imports the deleted module, so the package
does not import right now.** Task 5 fixes that. Until then, expect collection
errors from any test that imports `poseydon.ops`.

A deliberate deviation from the spec: `chain_ids` lives in `masking.py`, not
`core/topology.py`. That module is numpy-only, and this repo's convention for a
torch twin is a separate module (`core/kinematics.py` → `core/torch_kinematics.py`).
One caller does not justify a new `core/torch_topology.py`.

---

### Task 1: Test the maskers and the chain decomposition

**Files:**
- Create: `tests/models/__init__.py`, `tests/models/conftest.py`, `tests/models/test_masking.py`
- Modify: `src/poseydon/models/modules/masking.py` (only if a test fails)

**Interfaces:**
- Consumes: nothing.
- Produces: the `tiny_rig` and `tiny_valid` fixtures, reused by Tasks 2 and 3.
  - `tiny_rig() -> torch.Tensor` — `(1, 9)` int64 parents.
  - `tiny_valid() -> torch.Tensor` — `(1, 9)` bool, all true.
  - Module surface under test: `chain_ids(parents, joint_valid) -> (B, J) int64`,
    `one_chain_per_sample(parents, joint_valid) -> (B, J) bool`,
    `MotionMasker(d_model, frames=0.0, joints=0.0, chains=0.0)` with
    `.enabled`, `.mask(joint_valid, parents, n_frames) -> (T, B, J) bool | None`,
    `.apply(motion, mask) -> (T, B, J, C)`, `.null_token`;
    `ConditionMasker(d_model, cond_drop=0.0)` with `.drop(z_sem, cond) -> (T, B, 1, C)`, `.null_latent`.

- [ ] **Step 1: Create the package and its fixtures**

`tests/models/__init__.py` is empty. `tests/models/conftest.py`:

```python
"""Fixtures for model-level tests.

A nine-joint rig, small enough to assert on by hand and shaped to exercise
every branch of the chain decomposition: a two-joint chain (neck, head), a
three-joint chain (the tail), three single-joint chains at a branch point, and
a root with two children so no chain runs through it.

    0 root
    |-- 1 spine ----- 2 neck -- 3 head
    |              |- 4 leg_l
    |              `- 5 leg_r
    `-- 6 tail1 ---- 7 tail2 -- 8 tail3

Chains: {1}, {2,3}, {4}, {5}, {6,7,8}, plus the root's own {0}.
"""

from __future__ import annotations

import pytest
import torch

PARENTS = (-1, 0, 1, 2, 1, 1, 0, 6, 7)


@pytest.fixture
def tiny_rig() -> torch.Tensor:
    return torch.tensor([PARENTS], dtype=torch.long)


@pytest.fixture
def tiny_valid() -> torch.Tensor:
    return torch.ones(1, len(PARENTS), dtype=torch.bool)
```

- [ ] **Step 2: Write the failing tests**

`tests/models/test_masking.py`:

```python
"""What the model is not told, and what stands in for it."""

from __future__ import annotations

import pytest
import torch

from poseydon.core.batch import Cond
from poseydon.models.base import COND_DROP
from poseydon.models.modules.masking import (
    ConditionMasker,
    MotionMasker,
    chain_ids,
    one_chain_per_sample,
)

CHAINS = ({1}, {2, 3}, {4}, {5}, {6, 7, 8})


def test_chain_ids_group_a_run_and_cut_at_a_branch(tiny_rig, tiny_valid):
    ids = chain_ids(tiny_rig, tiny_valid)[0].tolist()
    assert ids[2] == ids[3], "neck and head are one chain"
    assert ids[6] == ids[7] == ids[8], "the tail is one chain"
    assert ids[1] != ids[2], "the spine is cut where the legs branch"
    assert ids[4] != ids[5], "two leaves of one parent are two chains"
    # Five chains plus the root's own.
    assert len(set(ids)) == 6


def test_a_chain_never_runs_through_the_root(tiny_rig, tiny_valid):
    ids = chain_ids(tiny_rig, tiny_valid)[0].tolist()
    assert ids[0] != ids[1] and ids[0] != ids[6]


def test_one_chain_per_sample_picks_exactly_one_whole_chain(tiny_rig, tiny_valid):
    for seed in range(25):
        torch.manual_seed(seed)
        mask = one_chain_per_sample(tiny_rig, tiny_valid)
        picked = {joint for joint, hit in enumerate(mask[0].tolist()) if hit}
        assert picked in CHAINS, f"seed {seed} picked {picked}"
        assert not mask[0, 0], "the root slot carries the trajectory, never mask it"


def test_padded_joints_are_not_part_of_any_chain(tiny_rig):
    """The tail is padding here, so it must never be drawn."""
    valid = torch.ones(1, 9, dtype=torch.bool)
    valid[0, 6:] = False
    for seed in range(15):
        torch.manual_seed(seed)
        mask = one_chain_per_sample(tiny_rig, valid)
        assert not mask[0, 6:].any()


def test_an_unconfigured_masker_is_a_true_no_op(tiny_rig, tiny_valid):
    masker = MotionMasker(d_model=4)
    assert not masker.enabled
    assert masker.mask(tiny_valid, tiny_rig, n_frames=5) is None
    motion = torch.randn(5, 1, 9, 4)
    assert masker.apply(motion, None) is motion


def test_eval_mode_never_masks(tiny_rig, tiny_valid):
    masker = MotionMasker(d_model=4, frames=1.0, joints=1.0, chains=1.0).eval()
    assert masker.mask(tiny_valid, tiny_rig, n_frames=5) is None


def test_the_null_token_lands_only_at_masked_slots():
    masker = MotionMasker(d_model=4)
    motion = torch.randn(3, 1, 9, 4)
    mask = torch.zeros(3, 1, 9, dtype=torch.bool)
    mask[1, 0, 2] = True

    out = masker.apply(motion, mask)
    assert torch.equal(out[1, 0, 2], masker.null_token)
    untouched = ~mask[..., None].expand_as(motion)
    assert torch.equal(out[untouched], motion[untouched])


def test_frame_masking_hides_a_frame_across_every_joint(tiny_rig, tiny_valid):
    torch.manual_seed(0)
    masker = MotionMasker(d_model=4, frames=1.0)
    mask = masker.mask(tiny_valid, tiny_rig, n_frames=40)
    per_frame = mask[:, 0].sum(dim=1)
    assert set(per_frame.tolist()) <= {0, 9}, "a masked frame hides all of its joints"
    assert mask.any(), "rate 1.0 over 40 frames must hide something"


def test_padding_is_never_masked(tiny_rig):
    torch.manual_seed(0)
    valid = torch.ones(1, 9, dtype=torch.bool)
    valid[0, 5:] = False
    masker = MotionMasker(d_model=4, frames=1.0, joints=1.0, chains=1.0)
    mask = masker.mask(valid, tiny_rig, n_frames=20)
    assert not mask[:, :, 5:].any()


@pytest.mark.parametrize("rate", [-0.1, 1.5])
def test_a_rate_outside_zero_to_one_is_rejected(rate):
    with pytest.raises(ValueError, match="must be in"):
        MotionMasker(d_model=4, frames=rate)


def test_cond_drop_replaces_the_whole_latent():
    torch.manual_seed(0)
    masker = ConditionMasker(d_model=4, cond_drop=1.0)
    z_sem = torch.randn(3, 2, 1, 4)
    out = masker.drop(z_sem, Cond({}))
    assert torch.equal(out[0, 0, 0], masker.null_latent)
    assert torch.equal(out[2, 1, 0], masker.null_latent)


def test_cond_drop_is_off_in_eval_unless_asked(tiny_valid):
    masker = ConditionMasker(d_model=4, cond_drop=1.0).eval()
    z_sem = torch.randn(3, 2, 1, 4)
    assert torch.equal(masker.drop(z_sem, Cond({})), z_sem)
    forced = masker.drop(z_sem, Cond({COND_DROP: True}))
    assert torch.equal(forced[0, 0, 0], masker.null_latent)


def test_cond_drop_honours_a_per_sample_selection():
    masker = ConditionMasker(d_model=4).eval()
    z_sem = torch.randn(3, 2, 1, 4)
    selection = torch.tensor([True, False])
    out = masker.drop(z_sem, Cond({COND_DROP: selection}))
    assert torch.equal(out[0, 0, 0], masker.null_latent)
    assert torch.equal(out[:, 1], z_sem[:, 1])
```

- [ ] **Step 3: Run them**

Run: `docker compose run --rm test pytest tests/models/test_masking.py -v`

Expected: all pass, since the module is already on disk. Any failure is a bug in
`masking.py` — fix the module, not the test. `tests/models/test_masking.py`
imports nothing from `poseydon.ops`, so the broken `generate.py` import does not
affect it.

- [ ] **Step 4: Commit**

```bash
git add tests/models src/poseydon/models/modules/masking.py src/poseydon/models/base.py
git commit -m "test(models): cover the maskers and the chain decomposition"
```

---

### Task 2: Wire `MotionMasker` into the encoder's input

**Files:**
- Modify: `src/poseydon/models/modiffae.py` (`InputProcess.forward`, `SemanticEncoder.__init__`/`forward`, `MoDiffAE.__init__`/`encode`)
- Test: `tests/models/test_modiffae_conditioning.py` (create), `tests/models/conftest.py` (extend)

**Interfaces:**
- Consumes: `MotionMasker` and its `.mask`/`.apply` from Task 1.
- Produces:
  - `InputProcess.forward(x, tpose, name_embeddings, crop_start, masking=None, mask=None)` — two new trailing optional args; the decoder passes neither.
  - `SemanticEncoder.forward(x, tpose, names, crop_start, hops, edges, joint_valid, temporal_valid, masking=None, parents=None)`.
  - `MoDiffAE.__init__(..., masking: dict | None = None)`, exposing `self.motion_masking: MotionMasker`.

- [ ] **Step 1: Extend the fixtures with a runnable model and batch**

REPLACE `tests/models/conftest.py` with the version below rather than appending
to it: appended imports land mid-file and `ruff` rejects them (E402). The
docstring, `PARENTS`, `tiny_rig` and `tiny_valid` are carried over from Task 1
unchanged.

```python
"""Fixtures for model-level tests.

A nine-joint rig, small enough to assert on by hand and shaped to exercise
every branch of the chain decomposition: a two-joint chain (neck, head), a
three-joint chain (the tail), three single-joint chains at a branch point, and
a root with two children so no chain runs through it.

    0 root
    |-- 1 spine ----- 2 neck -- 3 head
    |              |- 4 leg_l
    |              `- 5 leg_r
    `-- 6 tail1 ---- 7 tail2 -- 8 tail3

Chains: {1}, {2,3}, {4}, {5}, {6,7,8}, plus the root's own {0}.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from poseydon.core.batch import Cond, Masks
from poseydon.core.topology import edge_relations, hop_distances
from poseydon.models.base import CLEAN_MOTION
from poseydon.models.modiffae import MoDiffAE

PARENTS = (-1, 0, 1, 2, 1, 1, 0, 6, 7)
FEATURE_DIM = 6
N_FRAMES = 7


@pytest.fixture
def tiny_rig() -> torch.Tensor:
    return torch.tensor([PARENTS], dtype=torch.long)


@pytest.fixture
def tiny_valid() -> torch.Tensor:
    return torch.ones(1, len(PARENTS), dtype=torch.bool)


@pytest.fixture
def tiny_model():
    """A MoDiffAE small enough to run on CPU in a test.

    `dropout=0.0` deliberately: these tests compare two forward passes for
    equality, and dropout would make them differ for reasons that have nothing
    to do with what is being asserted.
    """

    def _build(**overrides):
        settings = dict(
            feature_dim=FEATURE_DIM,
            d_model=16,
            n_layers_semantic=1,
            n_layers_stochastic=1,
            n_heads=2,
            ff_size=32,
            dropout=0.0,
            n_virtual_joints=2,
        )
        settings.update(overrides)
        return MoDiffAE(**settings)

    return _build


@pytest.fixture
def tiny_batch(tiny_rig, tiny_valid):
    """`(z_t, t, cond, masks)` for one sample of the nine-joint rig."""
    parents = np.asarray(PARENTS, dtype=np.int64)
    joints = len(PARENTS)
    clean = torch.randn(1, joints, FEATURE_DIM, N_FRAMES)
    cond = Cond(
        {
            "topology": {
                "parents": tiny_rig,
                "offsets": torch.zeros(1, joints, 3),
                "relations": torch.from_numpy(edge_relations(parents))[None],
                "hops": torch.from_numpy(hop_distances(parents))[None],
            },
            "tpose": torch.randn(1, joints, FEATURE_DIM),
            CLEAN_MOTION: clean,
        }
    )
    masks = Masks(
        frames=torch.ones(1, N_FRAMES, dtype=torch.bool),
        joints=tiny_valid.clone(),
    )
    return torch.randn(1, joints, FEATURE_DIM, N_FRAMES), torch.zeros(1, dtype=torch.long), cond, masks
```

- [ ] **Step 2: Write the failing tests**

`tests/models/test_modiffae_conditioning.py`:

```python
"""Masked conditioning and guidance, through a real MoDiffAE."""

from __future__ import annotations

import torch

from poseydon.core.batch import Cond
from poseydon.models.base import CLEAN_MOTION, COND_DROP
from poseydon.models.modiffae import Z_SEM


def _latent(model, cond, masks):
    z_sem, _ = model.encode(cond[CLEAN_MOTION], cond, masks)
    return z_sem


def test_masking_off_leaves_the_latent_untouched(tiny_model, tiny_batch):
    _, _, cond, masks = tiny_batch
    model = tiny_model().train()
    torch.manual_seed(0)
    first = _latent(model, cond, masks)
    torch.manual_seed(0)
    second = _latent(model, cond, masks)
    assert torch.equal(first, second)


def test_masking_changes_the_latent_while_training(tiny_model, tiny_batch):
    _, _, cond, masks = tiny_batch
    model = tiny_model(masking={"frames": 1.0, "joints": 1.0, "chains": 1.0}).train()
    baseline = tiny_model()
    baseline.load_state_dict(model.state_dict(), strict=False)

    torch.manual_seed(0)
    masked = _latent(model, cond, masks)
    unmasked = _latent(baseline.eval(), cond, masks)
    assert not torch.allclose(masked, unmasked)


def test_masking_is_inert_in_eval(tiny_model, tiny_batch):
    """Sampling and retargeting must see whole conditioning."""
    _, _, cond, masks = tiny_batch
    model = tiny_model(masking={"frames": 1.0, "joints": 1.0, "chains": 1.0}).eval()
    baseline = tiny_model().eval()
    baseline.load_state_dict(model.state_dict(), strict=False)
    assert torch.equal(_latent(model, cond, masks), _latent(baseline, cond, masks))


def test_the_rest_frame_is_never_masked(tiny_model, tiny_batch):
    """Token 0 describes the rig. Masking it would corrupt what both CFG
    branches have to agree on, so the masked and unmasked latents must still
    agree there."""
    _, _, cond, masks = tiny_batch
    model = tiny_model(masking={"frames": 1.0}).train()
    baseline = tiny_model().eval()
    baseline.load_state_dict(model.state_dict(), strict=False)

    torch.manual_seed(0)
    # The rest token attends only itself (`temporal_pair_mask`), so its latent
    # row cannot be reached by any masked motion frame.
    assert torch.allclose(
        _latent(model, cond, masks)[0], _latent(baseline, cond, masks)[0], atol=1e-6
    )
```

```python
def test_a_masked_slot_still_carries_its_position(tiny_model, tiny_batch):
    """The null token replaces the MOTION embedding; names and frame positions
    are added on top of it. If the substitution ever moves after that addition
    every masked slot becomes the same vector, and the encoder can no longer
    tell WHERE the hole is -- which is the whole reason to learn a token
    instead of dropping the slot."""
    _, _, cond, masks = tiny_batch
    model = tiny_model()
    masker = model.motion_masking
    clean = cond[CLEAN_MOTION]

    every_slot = torch.ones(clean.shape[3], 1, clean.shape[1], dtype=torch.bool)
    tokens = model.semantic_encoder.input_process(
        clean, cond["tpose"], None, torch.zeros(1, dtype=torch.long), masker, every_slot
    )
    # Token 0 is the rest frame; 1 onwards are the fully-masked motion frames.
    assert not torch.allclose(tokens[1, 0, 0], tokens[2, 0, 0]), "positions differ"
    assert not torch.allclose(tokens[1, 0, 0], masker.null_token), "position was added"
```

- [ ] **Step 3: Run them to verify they fail**

Run: `docker compose run --rm test pytest tests/models/test_modiffae_conditioning.py -v -k "masking or rest_frame or masked_slot"`

Expected: FAIL — `MoDiffAE.__init__() got an unexpected keyword argument 'masking'`.

- [ ] **Step 4: Thread the masker through**

In `src/poseydon/models/modiffae.py`, add the import:

```python
from poseydon.models.modules.masking import ConditionMasker, MotionMasker
```

`InputProcess.forward` — change the signature and substitute right after the
motion embedding is built, before the rest frame is prepended and before names
and positions are added:

```python
    def forward(self, x, tpose, name_embeddings, crop_start, masking=None, mask=None):
        """``x`` is ``(B, J, D, T)``; returns ``(T + 1, B, J, C)``.

        ``masking`` and ``mask`` travel together: the encoder draws the mask,
        because it is the one holding ``joint_valid`` and the topology, and this
        substitutes the null token for the masked MOTION embeddings. Names and
        frame positions are added afterwards on purpose -- the encoder should
        read "joint 7 at frame 12 is hidden", not find a hole where a token
        should be. The decoder passes neither argument.
        """
        x = x.permute(3, 0, 1, 2)  # (T, B, J, D)
        tpose = tpose.unsqueeze(0)  # (1, B, J, D)

        motion = torch.cat(
            [self.root_embedding(x[:, :, 0:1]), self.joint_embedding(x[:, :, 1:])], dim=2
        )
        if masking is not None:
            motion = masking.apply(motion, mask)
```

The rest of the method is unchanged: `rest` is built from `tpose` and never
masked, and `out = torch.cat([rest, motion], dim=0)` keeps the rest frame at
index 0.

`SemanticEncoder.forward` — draw the mask and hand both down:

```python
    def forward(
        self, x, tpose, names, crop_start, hops, edges, joint_valid, temporal_valid,
        masking=None, parents=None,
    ):
        mask = (
            masking.mask(joint_valid, parents, x.shape[3])
            if masking is not None and parents is not None
            else None
        )
        tokens = self.input_process(x, tpose, names, crop_start, masking, mask)
```

The remaining body is unchanged.

`MoDiffAE.__init__` — accept the block and build the masker. Add to the
signature after `kl_bottleneck: bool = False,`:

```python
        masking: dict | None = None,
```

and after `self.temporal_window = temporal_window`:

```python
        # A plain dict of rates rather than a `_target_` block: Hydra's
        # recursive instantiate would build this before `MoDiffAE` and without
        # a `d_model` to size the null token, forcing the width to be
        # duplicated in the config.
        self.motion_masking = MotionMasker(d_model, **(masking or {}))
```

`MoDiffAE.encode` — pass them, replacing the `return self.semantic_encoder(...)` call's tail:

```python
        return self.semantic_encoder(
            clean,
            cond["tpose"].to(clean.device),
            cond.get(JOINT_NAMES),
            cond.get("crop_start", torch.zeros(batch, dtype=torch.long)),
            topology["hops"].to(clean.device),
            topology["relations"].to(clean.device),
            joint_valid,
            pair,
            masking=self.motion_masking,
            parents=topology["parents"].to(clean.device),
        )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `docker compose run --rm test pytest tests/models -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/poseydon/models/modiffae.py tests/models
git commit -m "feat(models): mask frames, joints and chains out of the semantic encoder"
```

---

### Task 3: Wire `ConditionMasker` into `MoDiffAE.forward`

**Files:**
- Modify: `src/poseydon/models/modiffae.py` (`MoDiffAE.__init__`, `MoDiffAE.forward`)
- Test: `tests/models/test_modiffae_conditioning.py` (append)

**Interfaces:**
- Consumes: `ConditionMasker.drop(z_sem, cond)` from Task 1; `COND_DROP` from `models/base.py`.
- Produces: `MoDiffAE.__init__(..., cond_drop: float = 0.0)`, exposing `self.cond_masking: ConditionMasker`. `MoDiffAE.forward` applies the drop after reading `cond[Z_SEM]`, and `Prediction.aux[Z_SEM]` carries the post-drop latent.

- [ ] **Step 1: Write the failing tests**

Append to `tests/models/test_modiffae_conditioning.py`:

```python
def test_cond_drop_makes_the_output_independent_of_the_clean_motion(
    tiny_model, tiny_batch
):
    z_t, t, cond, masks = tiny_batch
    model = tiny_model(cond_drop=1.0).train()

    other = Cond({**cond.payloads, CLEAN_MOTION: torch.randn_like(cond[CLEAN_MOTION])})
    first = model(z_t, t, cond, masks).out
    second = model(z_t, t, other, masks).out
    assert torch.allclose(first, second, atol=1e-6)


def test_cond_drop_off_keeps_the_output_dependent_on_it(tiny_model, tiny_batch):
    z_t, t, cond, masks = tiny_batch
    model = tiny_model().eval()

    other = Cond({**cond.payloads, CLEAN_MOTION: torch.randn_like(cond[CLEAN_MOTION])})
    assert not torch.allclose(
        model(z_t, t, cond, masks).out, model(z_t, t, other, masks).out
    )


def test_the_drop_key_beats_a_supplied_latent(tiny_model, tiny_batch):
    """`Retarget` pins a latent through `cond[Z_SEM]`. A guided retarget's
    unconditional branch has to override that pin, or both branches are the
    same and guidance is silently zero."""
    z_t, t, cond, masks = tiny_batch
    model = tiny_model().eval()

    pinned = torch.randn(z_t.shape[3] + 1, 1, 1, model.d_model)
    conditional = Cond({**cond.payloads, Z_SEM: pinned})
    dropped = Cond({**cond.payloads, Z_SEM: pinned, COND_DROP: True})

    assert torch.allclose(model(z_t, t, conditional, masks).aux[Z_SEM], pinned)
    assert torch.equal(
        model(z_t, t, dropped, masks).aux[Z_SEM][0, 0, 0], model.cond_masking.null_latent
    )
    assert not torch.allclose(
        model(z_t, t, conditional, masks).out, model(z_t, t, dropped, masks).out
    )
```

- [ ] **Step 2: Run to verify failure**

Run: `docker compose run --rm test pytest tests/models/test_modiffae_conditioning.py -v -k "cond_drop or drop_key"`

Expected: FAIL — `unexpected keyword argument 'cond_drop'`.

- [ ] **Step 3: Apply the drop**

In `MoDiffAE.__init__`, add to the signature after `masking: dict | None = None,`:

```python
        cond_drop: float = 0.0,
```

and beside the motion masker:

```python
        self.cond_masking = ConditionMasker(d_model, cond_drop)
```

In `MoDiffAE.forward`, replace the latent block:

```python
        supplied = cond.get(Z_SEM)
        if supplied is not None:
            z_sem, aux = supplied.to(device), {}
        else:
            z_sem, aux = self.encode(cond[CLEAN_MOTION].to(device), cond, masks)
        # AFTER the supplied latent is read, never before: `COND_DROP` is how
        # an unconditional branch is asked for, and it has to beat a pin.
        z_sem = self.cond_masking.drop(z_sem, cond)
```

The `Prediction` at the end already returns `{**aux, Z_SEM: z_sem}`, so it now
carries the post-drop latent, which is the one that actually conditioned the
decoder.

- [ ] **Step 4: Run to verify they pass**

Run: `docker compose run --rm test pytest tests/models -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/models/modiffae.py tests/models
git commit -m "feat(models): drop the whole semantic latent for CFG training"
```

---

### Task 4: Multi-head semantic modulation

**Files:**
- Modify: `src/poseydon/models/modules/graph_backbone.py` (`ConditionAttention`, `GraphMotionDecoderLayer`), `src/poseydon/models/modiffae.py` (`StochasticDecoder`, `MoDiffAE`)
- Test: `tests/models/test_condition_attention.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `ConditionAttention(d_model, dropout=0.1, n_heads=1, aggregate="mean")`.
  - `GraphMotionDecoderLayer(..., n_modulation_heads: int = 1, modulation_aggregate: str = "mean")`.
  - `StochasticDecoder(..., n_modulation_heads: int = 1, modulation_aggregate: str = "mean")`.
  - `MoDiffAE(..., n_modulation_heads: int = 1, modulation_aggregate: str = "mean")`.

- [ ] **Step 1: Write the failing tests**

`tests/models/test_condition_attention.py`:

```python
"""Semantic modulation: one gated transform, or several averaged."""

from __future__ import annotations

import pytest
import torch

from poseydon.models.modules.graph_backbone import ConditionAttention

D_MODEL = 8


def _inputs():
    torch.manual_seed(0)
    return torch.randn(3, 2, 4, D_MODEL), torch.randn(3, 2, 1, D_MODEL)


def test_one_head_keeps_the_published_parameter_names():
    """The reference's checkpoints load into this module by name. One head must
    stay exactly the module it has always been."""
    keys = set(ConditionAttention(D_MODEL).state_dict())
    assert keys == {
        "query_projector.0.weight", "query_projector.0.bias",
        "query_projector.1.weight", "query_projector.1.bias",
        "query_projector.3.weight", "query_projector.3.bias",
        "value_map.weight", "value_map.bias",
        "out_proj.weight", "out_proj.bias",
    }


def test_one_head_is_the_gated_transform_it_has_always_been():
    """Not just the same parameter names -- the same arithmetic. Guards the
    `n_heads == 1` path against a refactor that changes what it computes."""
    x, memory = _inputs()
    module = ConditionAttention(D_MODEL, dropout=0.0).eval()

    expanded = memory.expand(-1, -1, x.shape[2], -1)
    gate = torch.sigmoid(module.query_projector(torch.cat([x, expanded], dim=-1)))
    expected = module.out_proj(module.value_map(expanded) * gate)
    assert torch.allclose(module(x, memory), expected, atol=1e-6)


def test_mean_over_identical_heads_equals_one_head():
    x, memory = _inputs()
    many = ConditionAttention(D_MODEL, dropout=0.0, n_heads=3, aggregate="mean").eval()
    for head in list(many.heads)[1:]:
        head.load_state_dict(many.heads[0].state_dict())

    one = ConditionAttention(D_MODEL, dropout=0.0).eval()
    one.query_projector.load_state_dict(many.heads[0]["query_projector"].state_dict())
    one.value_map.load_state_dict(many.heads[0]["value_map"].state_dict())
    one.out_proj.load_state_dict(many.out_proj.state_dict())

    assert torch.allclose(many(x, memory), one(x, memory), atol=1e-6)


def test_sum_over_identical_heads_is_n_times_one_head():
    x, memory = _inputs()
    many = ConditionAttention(D_MODEL, dropout=0.0, n_heads=3, aggregate="sum").eval()
    for head in list(many.heads)[1:]:
        head.load_state_dict(many.heads[0].state_dict())

    one = ConditionAttention(D_MODEL, dropout=0.0).eval()
    one.query_projector.load_state_dict(many.heads[0]["query_projector"].state_dict())
    one.value_map.load_state_dict(many.heads[0]["value_map"].state_dict())
    # `out_proj` is linear but has a bias, so compare before it: 3x the summed
    # pre-projection signal is not 3x the projected one.
    one.out_proj.load_state_dict(many.out_proj.state_dict())
    with torch.no_grad():
        one.out_proj.bias.zero_()
        many.out_proj.bias.zero_()

    assert torch.allclose(many(x, memory), 3.0 * one(x, memory), atol=1e-6)


def test_heads_are_independent_by_default():
    many = ConditionAttention(D_MODEL, n_heads=2)
    first = many.heads[0]["value_map"].weight
    second = many.heads[1]["value_map"].weight
    assert not torch.allclose(first, second)


@pytest.mark.parametrize("bad", [0, -1])
def test_a_head_count_below_one_is_rejected(bad):
    with pytest.raises(ValueError, match="at least one"):
        ConditionAttention(D_MODEL, n_heads=bad)


def test_an_unknown_aggregation_is_rejected():
    with pytest.raises(ValueError, match="aggregate"):
        ConditionAttention(D_MODEL, n_heads=2, aggregate="concat")
```

- [ ] **Step 2: Run to verify failure**

Run: `docker compose run --rm test pytest tests/models/test_condition_attention.py -v`

Expected: FAIL — `__init__() got an unexpected keyword argument 'n_heads'`.

- [ ] **Step 3: Implement the heads**

Replace `ConditionAttention` in `src/poseydon/models/modules/graph_backbone.py`:

```python
class ConditionAttention(nn.Module):
    """Gated semantic conditioning, optionally multi-head.

    Rather than cross-attending over a single latent -- where attention weights
    necessarily collapse to 1.0 and carry no information -- the latent is mapped
    to a value and gated by a projection of ``[joint, latent]``. The gate is what
    varies per joint.

    ``n_heads`` above 1 runs that transform several times over, each head with
    its own projector and value map, and reduces the results by ``mean`` or
    ``sum``. The heads are independent full-width transforms of one latent, not
    channel subspaces: slice-width heads could only be concatenated, and the
    gate here is already elementwise per channel, so subspaces would restrict
    what each gate sees rather than add capacity.

    ``mean`` is the default because ``sum`` scales the residual branch by
    ``n_heads`` at initialization.

    At ``n_heads=1`` this builds exactly the modules it always has, under the
    same names, so the reference's published checkpoints still load.
    """

    AGGREGATIONS = ("mean", "sum")

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        n_heads: int = 1,
        aggregate: str = "mean",
    ) -> None:
        super().__init__()
        if n_heads < 1:
            raise ValueError(f"semantic modulation needs at least one head, got {n_heads}")
        if aggregate not in self.AGGREGATIONS:
            raise ValueError(
                f"aggregate must be one of {self.AGGREGATIONS}, got {aggregate!r}"
            )
        self.n_heads = n_heads
        self.aggregate = aggregate
        self.gate_gen = nn.Sigmoid()
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        if n_heads == 1:
            self.query_projector = self._projector(d_model)
            self.value_map = nn.Linear(d_model, d_model)
        else:
            self.heads = nn.ModuleList(
                nn.ModuleDict(
                    {
                        "query_projector": self._projector(d_model),
                        "value_map": nn.Linear(d_model, d_model),
                    }
                )
                for _ in range(n_heads)
            )

    @staticmethod
    def _projector(d_model: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def _head(self, index: int) -> tuple[nn.Module, nn.Module]:
        if self.n_heads == 1:
            return self.query_projector, self.value_map
        return self.heads[index]["query_projector"], self.heads[index]["value_map"]

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        joints = x.shape[2]
        expanded = memory.expand(-1, -1, joints, -1)
        combined = torch.cat([x, expanded], dim=-1)

        total = None
        for index in range(self.n_heads):
            projector, value_map = self._head(index)
            gated = value_map(expanded) * self.gate_gen(projector(combined))
            total = gated if total is None else total + gated
        if self.aggregate == "mean" and self.n_heads > 1:
            total = total / self.n_heads
        return self.dropout(self.out_proj(total))
```

`GraphMotionDecoderLayer.__init__` — add two parameters after `virtual: bool = False,`:

```python
        n_modulation_heads: int = 1,
        modulation_aggregate: str = "mean",
```

and build the module with them:

```python
        self.cross_attn = (
            ConditionAttention(d_model, dropout, n_modulation_heads, modulation_aggregate)
            if cross
            else None
        )
```

- [ ] **Step 4: Pass them down from the model**

In `src/poseydon/models/modiffae.py`, `StochasticDecoder.__init__` — add after `text_dim: int = 768,`:

```python
        n_modulation_heads: int = 1,
        modulation_aggregate: str = "mean",
```

and extend its layer factory:

```python
            lambda: GraphMotionDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                max_path_len=max_path_len,
                timestep=True,
                cross=True,
                virtual=False,
                n_modulation_heads=n_modulation_heads,
                modulation_aggregate=modulation_aggregate,
            ),
```

In `MoDiffAE.__init__` — add after `cond_drop: float = 0.0,`:

```python
        n_modulation_heads: int = 1,
        modulation_aggregate: str = "mean",
```

and pass them to the decoder, which is the only half that has a
`ConditionAttention` at all (the encoder builds its layers with `cross=False`):

```python
        self.stochastic_decoder = StochasticDecoder(
            feature_dim, d_model, n_layers_stochastic, n_heads, ff_size, dropout,
            max_path, text_dim, n_modulation_heads, modulation_aggregate,
        )
```

- [ ] **Step 5: Run the tests**

Run: `docker compose run --rm test pytest tests/models -v`

Expected: PASS.

- [ ] **Step 6: Verify the existing checkpoint still loads at one head**

Run:

```bash
docker compose run --rm --entrypoint python test -c "
import torch
from poseydon.models.modiffae import MoDiffAE
state = torch.load('runs/modiffae-attn5-b16-d128/checkpoints/last.ckpt', map_location='cpu')
weights = {k[len('model.'):]: v for k, v in state['state_dict'].items() if k.startswith('model.')}
model = MoDiffAE(feature_dim=13, d_model=128, n_virtual_joints=5, ff_size=1024)
report = model.load_state_dict(weights, strict=False)
assert set(report.missing_keys) == {'motion_masking.null_token', 'cond_masking.null_latent'}, report.missing_keys
assert not report.unexpected_keys, report.unexpected_keys
print('the 150k checkpoint still loads; only the two new null embeddings are new')
"
```

Expected: the printed line. `strict=False` deliberately: the two maskers add
`motion_masking.null_token` and `cond_masking.null_latent`, which a checkpoint
trained before they existed cannot carry. The assertions are what make this a
test rather than a shrug — they fail if ANY other key moved, which is what
would actually break the reference checkpoints.

- [ ] **Step 7: Commit**

```bash
git add src/poseydon/models tests/models
git commit -m "feat(models): multi-head semantic modulation, averaged or summed"
```

---

### Task 5: Guidance at sampling time

**Files:**
- Modify: `src/poseydon/ops/base.py` (`Operation.wrap_model`, `Operation.run`), `src/poseydon/ops/generate.py`
- Test: `tests/sampling/test_guidance.py` (create)

**Interfaces:**
- Consumes: `GuidedDenoiser(inner, scale)` from `poseydon.sampling.guidance` (already on disk); `COND_DROP`.
- Produces: `Operation.wrap_model(model) -> Denoiser`, identity by default, applied inside `Operation.run`. `Generate.wrap_model` returns a `GuidedDenoiser` when `guidance_scale > 0`, and `Generate.build_controls` returns `[]` always.

This task also repairs the package: `ops/generate.py` currently imports the
deleted `sampling.controls.guidance`, so `import poseydon.ops` fails until
Step 3 lands.

- [ ] **Step 1: Write the failing tests**

`tests/sampling/test_guidance.py`:

```python
"""Classifier-free guidance, as a wrapper around a denoiser."""

from __future__ import annotations

import pytest
import torch

from poseydon.core.batch import Cond
from poseydon.models.base import COND_DROP, Denoiser, Prediction
from poseydon.ops.generate import Generate
from poseydon.sampling.guidance import GuidedDenoiser


class Branchy(Denoiser):
    """Returns 1.0 conditioned and 3.0 unconditioned, and counts its calls."""

    requires = ("topology",)
    optional = ("joint_names",)

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, z_t, t, cond, masks=None):
        self.calls += 1
        value = 3.0 if cond.get(COND_DROP) else 1.0
        return Prediction(out=torch.full_like(z_t, value), aux={"branch": value})


class Encoding(Branchy):
    def encode(self, clean, cond, masks=None, temporal_valid=None):
        return torch.zeros(1), {}


def test_the_wrapper_extrapolates_away_from_the_unconditional_branch():
    guided = GuidedDenoiser(Branchy(), scale=2.0)
    out = guided(torch.zeros(1, 2, 3, 4), torch.zeros(1), Cond({})).out
    # u + s * (c - u) = 3 + 2 * (1 - 3) = -1
    assert torch.allclose(out, torch.full_like(out, -1.0))


def test_scale_zero_costs_one_forward_and_returns_the_conditional():
    inner = Branchy()
    guided = GuidedDenoiser(inner, scale=0.0)
    out = guided(torch.zeros(1, 2, 3, 4), torch.zeros(1), Cond({})).out
    assert torch.allclose(out, torch.ones_like(out))
    assert inner.calls == 1, "the shipped default must not pay for a second pass"


def test_the_aux_comes_from_the_conditional_branch():
    guided = GuidedDenoiser(Branchy(), scale=2.0)
    aux = guided(torch.zeros(1, 2, 3, 4), torch.zeros(1), Cond({})).aux
    assert aux["branch"] == 1.0


def test_a_negative_scale_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        GuidedDenoiser(Branchy(), scale=-1.0)


def test_the_wrapper_asks_for_the_inner_model_s_conditioners():
    guided = GuidedDenoiser(Branchy(), scale=1.0)
    assert guided.requires == ("topology",)
    with pytest.raises(ValueError, match="topology"):
        guided.check_conditioners(set())


def test_encode_delegates_when_the_inner_model_has_one():
    guided = GuidedDenoiser(Encoding(), scale=1.0)
    latent, aux = guided.encode(torch.zeros(1), Cond({}))
    assert latent.shape == (1,)


def test_a_model_without_encode_still_looks_like_one_through_the_wrapper():
    """`Retarget` branches on `getattr(model, "encode", None)`; the wrapper
    must not invent an `encode` that a bare denoiser does not have."""
    guided = GuidedDenoiser(Branchy(), scale=1.0)
    assert getattr(guided, "encode", None) is None


def test_generate_wraps_only_when_guidance_is_asked_for():
    inner = Branchy()
    assert Generate(guidance_scale=0.0).wrap_model(inner) is inner
    wrapped = Generate(guidance_scale=2.5).wrap_model(inner)
    assert isinstance(wrapped, GuidedDenoiser)
    assert wrapped.scale == 2.5


def test_generate_declares_no_controls():
    assert list(Generate(guidance_scale=2.5).build_controls(Cond({}))) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `docker compose run --rm test pytest tests/sampling/test_guidance.py -v`

Expected: FAIL at collection — `ModuleNotFoundError: poseydon.sampling.controls.guidance`,
raised from `ops/generate.py`.

- [ ] **Step 3: Add the hook and rewrite `Generate`**

In `src/poseydon/ops/base.py`, add a method to `Operation` above `run`:

```python
    def wrap_model(self, model: Denoiser) -> Denoiser:
        """The denoiser this operation actually samples with.

        Identity for everything that samples the model as-is. Guidance needs
        two forward passes per step, which a `Control` cannot do -- it is given
        the state and the conditioning, never the model -- so it wraps here
        instead.
        """
        return model
```

and apply it in `run`:

```python
        return sampler.sample(
            model=self.wrap_model(model),
            process=process,
```

Replace `src/poseydon/ops/generate.py` entirely:

```python
"""Unconditional and conditional synthesis."""

from __future__ import annotations

from collections.abc import Sequence

from poseydon.core.batch import Cond
from poseydon.models.base import Denoiser
from poseydon.ops.base import OPERATIONS, Operation
from poseydon.sampling.base import Control
from poseydon.sampling.guidance import GuidedDenoiser


@OPERATIONS.register("generate")
class Generate(Operation):
    """Sample motion from noise. The base case: no controls at all."""

    def __init__(self, guidance_scale: float = 0.0) -> None:
        self.guidance_scale = guidance_scale

    def build_controls(self, cond: Cond) -> Sequence[Control]:
        return []

    def wrap_model(self, model: Denoiser) -> Denoiser:
        if self.guidance_scale > 0.0:
            return GuidedDenoiser(model, scale=self.guidance_scale)
        return model
```

- [ ] **Step 4: Run to verify they pass**

Run: `docker compose run --rm test pytest tests/sampling tests/ops -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/ops src/poseydon/sampling tests/sampling
git commit -m "feat(sampling): real classifier-free guidance, as a wrapper denoiser"
```

---

### Task 6: Let old recipes stay valid

**Files:**
- Modify: `src/poseydon/training/recipe.py` (`check_recipe`)
- Test: `tests/training/test_recipe_absent.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: no signature change. `check_recipe(active, recorded)` now ignores a
  differing row whose RECORDED side is `ABSENT`.

- [ ] **Step 1: Write the failing tests**

`tests/training/test_recipe_absent.py`:

```python
"""A recipe written before a config key existed cannot disagree about it."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from poseydon.training.recipe import check_recipe


def _recipe(model: dict) -> OmegaConf:
    return OmegaConf.create(
        {
            "features": ["ric_pos"],
            "conditioners": ["topology"],
            "model": model,
            "process": {"_target_": "p"},
            "dataset": {"reduce_tolerance": 1e-8, "normalize": "per_channel"},
        }
    )


def test_a_key_the_recipe_predates_is_not_checked():
    """Adding a model config key must not invalidate every checkpoint on disk."""
    recorded = _recipe({"_target_": "m", "d_model": 128})
    active = _recipe({"_target_": "m", "d_model": 128, "cond_drop": 0.0})
    check_recipe(active, recorded)


def test_a_key_the_active_config_lacks_still_raises():
    recorded = _recipe({"_target_": "m", "d_model": 128, "cond_drop": 0.1})
    active = _recipe({"_target_": "m", "d_model": 128})
    with pytest.raises(ValueError, match="cond_drop"):
        check_recipe(active, recorded)


def test_a_real_disagreement_still_raises():
    recorded = _recipe({"_target_": "m", "d_model": 128, "cond_drop": 0.1})
    active = _recipe({"_target_": "m", "d_model": 128, "cond_drop": 0.0})
    with pytest.raises(ValueError, match="cond_drop"):
        check_recipe(active, recorded)


def test_a_whole_missing_path_is_not_checked_either():
    recorded = _recipe({"_target_": "m"})
    recorded.pop("process")
    active = _recipe({"_target_": "m"})
    check_recipe(active, recorded)
```

- [ ] **Step 2: Run to verify failure**

Run: `docker compose run --rm test pytest tests/training/test_recipe_absent.py -v`

Expected: `test_a_key_the_recipe_predates_is_not_checked` and
`test_a_whole_missing_path_is_not_checked_either` FAIL with
"this checkpoint was trained under a different configuration".

- [ ] **Step 3: Add the escape**

In `src/poseydon/training/recipe.py`, `check_recipe`, replace the loop body:

```python
    rows: list[tuple[str, Any, Any]] = []
    for path in CHECKED:
        was = _plain(OmegaConf.select(recorded, path, default=ABSENT))
        now = _plain(OmegaConf.select(active, path, default=ABSENT))
        if was != now:
            # A key the RECORDED side lacks is a key that postdates the recipe,
            # and a recipe has nothing to say about a key that did not exist
            # when it was written -- otherwise adding one config key rejects
            # every checkpoint already on disk. Scoped to that side only: a key
            # the ACTIVE config lacks is a config that has drifted from the one
            # the checkpoint needs, and still raises.
            rows.extend(row for row in _rows(path, was, now) if row[1] != ABSENT)
```

Then update the comment block above `CHECKED` (`recipe.py:48-54`), replacing the
paragraph that begins "Adding a path here invalidates every recipe":

```python
# Adding a path here, or a key inside one of these blocks, does NOT invalidate
# the recipes already written: `check_recipe` ignores a row the recorded side
# lacks, because a recipe cannot disagree about a key that postdates it. The
# cost is that an old checkpoint loaded under a new value of such a key fails
# later, in `load_state_dict`, rather than here.
```

- [ ] **Step 4: Run to verify they pass**

Run: `docker compose run --rm test pytest tests/training -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/training/recipe.py tests/training/test_recipe_absent.py
git commit -m "fix(training): a recipe cannot disagree about a key that postdates it"
```

---

### Task 7: Configure, and prove the whole thing runs

**Files:**
- Modify: `configs/model/modiffae.yaml`
- Test: the full suite, then a 5-step `trainer=debug` run

**Interfaces:**
- Consumes: every earlier task.
- Produces: the six config keys, all off, which `recipe_of` records under `model`.

- [ ] **Step 1: Add the keys, off**

Append to `configs/model/modiffae.yaml`:

```yaml
# What the semantic encoder is NOT shown, so its latent has to describe the
# motion rather than carry a copy of it. Each rate is the MAXIMUM of a
# per-sample draw from U(0, rate) -- the idiom `DropEndEffector` uses -- so
# some samples in every batch are still seen whole and an always-on mask opens
# no gap against a sampling path that never masks. All three compose by OR.
# Masked slots take a learned null token, not zeros, and keep their joint-name
# and frame-position embeddings: the encoder is told WHERE the hole is.
# Off by default; this is a training-time bottleneck with no measured value yet.
masking:
  frames: 0.0      # fraction of frames hidden across every joint
  joints: 0.0      # fraction of individual (frame, joint) slots hidden
  chains: 0.0      # probability ONE kinematic chain is hidden for the whole window
# Probability the whole latent is replaced by its learned null, which is what
# classifier-free guidance needs to have something to guide away from. Sampling
# then asks for that branch with `operation.guidance_scale` -- which did
# nothing at all before this existed, because the control that implemented it
# set a conditioning key no model read.
cond_drop: 0.0
# Semantic modulation runs this many independent gated transforms of the
# latent and reduces them by `mean` or `sum`. 1 is the module the reference
# published and the only value its checkpoints load into; each extra head adds
# 4 * d_model^2 (about 65k at d_model 128) to every decoder layer.
n_modulation_heads: 1
modulation_aggregate: mean
```

- [ ] **Step 2: Confirm the default config is unchanged in effect**

Run:

```bash
docker compose run --rm test python -m poseydon.cli train --print-config | grep -A6 "^  masking:"
```

Expected: the six keys, all zero / 1 / mean.

- [ ] **Step 3: Run the whole suite and the linter**

Run:

```bash
docker compose run --rm test pytest
docker compose run --rm test ruff check src tests
```

Expected: no regressions against the pre-task baseline of **527 passed, 9
skipped, 12 xfailed**, plus the new tests; "All checks passed!".

- [ ] **Step 4: Prove the new settings actually train**

Run:

```bash
docker compose run --rm test python -m poseydon.cli train \
  trainer=debug \
  model.masking.frames=0.2 model.masking.joints=0.2 model.masking.chains=0.25 \
  model.cond_drop=0.1 \
  model.n_modulation_heads=4 model.modulation_aggregate=mean \
  validation.pairs=[] \
  --out /tmp/masked-cfg-debug
```

Expected: 5 steps complete and a recipe is written. `validation.pairs=[]`
because the retarget-validation callback runs a full 100-step DDPM per pair and
has nothing to do with whether these settings compose. A
`RuntimeError: unable to allocate shared memory` is benign noise from the
`test` service, which has no `shm_size` (only `train` does); the run still
completes.

- [ ] **Step 5: Commit**

```bash
git add configs/model/modiffae.yaml
git commit -m "feat(config): masking, cond drop and modulation heads, all off by default"
```

---

### Task 8: Launch the run

**Files:** none. This task runs commands.

**Interfaces:**
- Consumes: Task 7's verified configuration.

- [ ] **Step 1: Confirm wandb is configured**

Run: `grep -c WANDB .env`

Expected: `2` (`WANDB_API_KEY` and `WANDB_ENTITY`). `build_logger` returns
`None` without the key and the run would train unlogged, which is not what was
asked for.

- [ ] **Step 2: Launch**

```bash
docker compose run -d --name poseydon-masked-cfg train \
  python -m poseydon.cli train \
    model.masking.frames=0.2 \
    model.masking.joints=0.2 \
    model.masking.chains=0.25 \
    model.cond_drop=0.1 \
    model.n_modulation_heads=4 \
    model.modulation_aggregate=mean \
    --out runs/masked-cfg
```

Three things about this command are deliberate:

- **`--out runs/masked-cfg`.** `run_name` is built from target, pooling, batch
  and `d_model`, none of which these overrides change, so the default `runs/`
  resolves to `runs/modiffae-attn5-b16-d128/` — where the 75k–150k checkpoints
  and their recipe live, and where `ModelCheckpoint(save_last=True)` would
  overwrite `last.ckpt`.
- **Overrides, not new config defaults.** The run is self-describing because
  `recipe_of` records the whole `model` block, and the repo default stays the
  configuration the existing checkpoints load into.
- **From scratch, not a resume.** Four modulation heads means new parameter
  names and shapes; the 150k checkpoint cannot continue into this architecture.

The rates: `0.2` is the maximum of a per-sample `U(0, 0.2)` draw, so about 10%
of frames and 10% of slots on average, roughly 19% of slots once ORed --
close to BERT's 15%. `chains: 0.25` hides one chain on a quarter of samples,
which the measurement in the spec puts at a median 3.9% and at most 48.3% of a
rig. `cond_drop: 0.1` is the standard CFG value. None of these are measured
for this corpus; they are a first run.

- [ ] **Step 2b: Watch the first steps**

Run: `docker logs -f poseydon-masked-cfg`

Expected, in order: the recipe path, a wandb run URL, `run_at_start` validation
retargets for the three pairs, then the step counter advancing. Confirm the
parameter count is about **4.9M**, not 4.1M: three extra heads x 65k x 4
decoder layers is +786k, and its absence would mean the heads did not reach the
decoder.

- [ ] **Step 3: Report**

Give the wandb URL, the run directory, the parameter count, and the first
logged `train/total`. Flag that `grad_norm` is now unclipped
(`gradient_clip_val: null`) so its scale is directly comparable to the
125k–150k measurements: median 5.29, p99 38.4.
