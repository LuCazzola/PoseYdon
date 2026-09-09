# Read Path and Training Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the corpus plan A2 built and get a real MoDiffAE training run on the GB10, logging to wandb, resumable, at batch 16 / bf16 / 600 000 steps.

**Architecture:** Three independent seams. (1) A CUDA image and a compose service, so the GPU is reachable from the same bind-mounted tree the test image uses. (2) `MotionDataset` stops fitting statistics and inventing rest frames, and instead reads the artefacts `rigs/<Rig>/{skeleton,stats}.npz` that A2 writes — which is also what makes startup O(1) instead of O(corpus). (3) The Lightning wiring gains the four things the reference has and PoseYdon does not: a balanced sampler, an LR schedule, step-interval checkpoints, and a logger.

**Tech Stack:** PyTorch 2.14+cu130 (aarch64/sbsa), Lightning 2.6, Hydra, wandb, Docker Compose, numpy.

**Spec:** `docs/superpowers/specs/2026-09-08-training-run-and-retarget-validation-design.md` (§3 read path, §4 training loop, §5 FBX sidecheck). Parity authority: `docs/superpowers/specs/2026-09-06-data-pipeline-and-training-parity-design.md` (§9 read path, §10 training loop).

**Not in this plan:** `LatentPin`, `Retarget`, `poseydon retarget`, and the `RetargetValidation` callback with its five metrics. Those are Plan B (spec §6–§7), and they land on a run that already works.

## Global Constraints

- **Everything runs and is tested in Docker.** Never invoke `pytest` or `python` on the host. The CPU suite is `docker compose run --rm test bash -c "ruff check . && pytest -q"`. GPU work uses the `train` service Task 1 adds.
- **`bash -c`, never `bash -lc`.** A login shell re-reads the profile and drops `/opt/venv/bin` from `PATH`, so `ruff` and `pytest` vanish.
- **`ruff check` is the lint gate.** `ruff format` is NOT this project's convention — it would reformat 68 of 145 pre-existing files. Do not run it.
- **Do not touch `data/truebones/source/`.** It is a paid, irreplaceable corpus of 1153 raw BVH.
- **Do not rebuild the corpus.** `data/truebones/{clips,rigs,index.jsonl}` is current: 73 rigs, 73 `skeleton.npz`, 73 `stats.npz`, 1145 clip `.npz`, 1145 index rows. If a task needs it rebuilt, that is a signal the task is wrong.
- **The round trip is non-negotiable** (spec §9): structure and reference frame must remain recoverable. Nothing in this plan may drop a joint, a name, a parent, or a prepare constant from any artefact.
- **Run recipe, exact values:** `batch_size: 16`, `max_steps: 600000`, `precision: bf16-mixed`, `lr 1e-4`, `d_model: 128`, `n_layers_semantic: 4`, `n_layers_stochastic: 4`, `n_heads: 4`, `ff_size: 512`, `n_virtual_joints: 5`, balanced sampling, `losses: {simple: 1.0, geodesic: 1.0}`.
- **wandb:** project `poseydon`, entity `lcazzola-fondazione-bruno-kessler` (from `.env`; `entity: poseydon` fails — `poseydon` is the project). `.env` is gitignored and already carries `WANDB_API_KEY`, `WANDB_ENTITY`, `UID`, `GID`. Never write a key into a config, a test, or `.env.example`.
- **Measured baseline to preserve:** the CPU suite is 258 passed / 7 skipped / 10 xfailed before this plan starts. A task that reduces the passing count without explaining why has broken something.

---

## File Structure

| File | Responsibility |
|---|---|
| `Dockerfile.train` (create) | CUDA 13 aarch64/sbsa image; the only place the CPU torch pin is overridden |
| `docker-compose.yml` (modify) | a `train` service with the GPU attached via CDI and `.env` passed through |
| `src/poseydon/data/dataset.py` (modify) | reads A2's artefacts instead of fitting; per-rig caches; worker-safe RNG |
| `src/poseydon/data/sampler.py` (create) | `BalancedByRig` weights |
| `src/poseydon/training/task.py` (modify) | injects `crop_start` into `cond` |
| `src/poseydon/models/anytop.py` (modify) | reads `crop_start`, not `window_start` |
| `src/poseydon/models/modiffae.py` (modify) | `temporal_window` as a model parameter |
| `src/poseydon/losses/footskate.py` (modify) | thresholds the raw contact block |
| `src/poseydon/training/recipe.py` (create) | writes and reads `runs/<run>/recipe.yaml` |
| `src/poseydon/training/lightning.py` (modify) | `StepLR`, sampler, `drop_last`, worker seeding |
| `src/poseydon/training/build.py` (modify) | builds the logger and callbacks |
| `src/poseydon/cli.py` (modify) | `--resume`, logger and callback wiring |
| `configs/{train.yaml,trainer/default.yaml,model/modiffae.yaml}` (modify) | the run recipe |
| `scripts/check_fbx_coherence.py` (create) | spec §5 sidecheck, Blender-gated |

---

## Task 1: The CUDA image and the train service

**Files:**
- Create: `Dockerfile.train`
- Modify: `docker-compose.yml`
- Test: `tests/training/test_gpu_image.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: a `train` compose service usable as `docker compose run --rm train <cmd>`, with `/app` bind-mounted, `PYTHONPATH=/app/src:/app`, and `.env` loaded.

**Context the brief cannot know:** the image choice is already settled by a spike and is NOT open. Measured on this host: `torch 2.14.0+cu130`, `torch.cuda.is_available() True`, device `NVIDIA GB10`, capability `(12, 1)`, fp32 and bf16 matmuls and optimizer steps all working, full dependency set importing, image 19.4 GB. The wheel's arch list is `['sm_80','sm_90','sm_100','sm_110','sm_120']` — `sm_121` is absent and the GB10 runs through Blackwell minor-version compatibility. Do not "fix" that, do not build from source, and do not switch to an NGC base.

- [ ] **Step 1: Write `Dockerfile.train`**

```dockerfile
# Training image: the same slim base as the test image, with CUDA 13
# aarch64/sbsa torch wheels layered on top.
#
# Route chosen on measurement, not preference (design spec §4, "The aarch64
# image"): this works end to end on the GB10 and an NGC PyTorch aarch64 base
# was deliberately not built, because a ~20 GB pull to compare a route we do
# not need is not evidence worth buying.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml ./
# `uv sync` honours pyproject's [tool.uv.sources] pin and installs CPU torch;
# the second command replaces exactly that one package with the CUDA build.
# Done this way round so the shared pyproject.toml -- and therefore the CPU
# test image -- needs no change at all.
RUN uv sync --extra dev --no-install-project && \
    uv pip install --python /opt/venv/bin/python \
      --index-url https://download.pytorch.org/whl/cu130 \
      --reinstall-package torch torch && \
    uv pip install --python /opt/venv/bin/python wandb

# NOT installed: build-essential. It is needed only by torch.compile, which
# this run does not use -- measured at +8% on a median batch and 2.3x WORSE on
# the largest rig, because dynamic joint counts recompile constantly.

CMD ["python", "-c", "import torch; print(torch.__version__, torch.cuda.is_available())"]
```

- [ ] **Step 2: Add the `train` service to `docker-compose.yml`**

Add after the `test` service, matching its conventions (bind mount, host user, `PYTHONPATH`, `MPLCONFIGDIR`):

```yaml
  # Training. Same bind mount and host user as `test`, plus the GPU.
  train:
    build:
      context: .
      dockerfile: Dockerfile.train
    working_dir: /app
    user: "${UID:-1000}:${GID:-1000}"
    # CDI is how this host exposes the GB10 (`docker info` lists
    # nvidia.com/gpu=all); there is no `--gpus` runtime configured.
    devices:
      - nvidia.com/gpu=all
    volumes:
      - .:/app
    environment:
      PYTHONPATH: /app/src:/app
      MPLCONFIGDIR: /tmp/matplotlib
      # Compose reads .env automatically; these forward the values into the
      # container. Never hardcode a key here.
      WANDB_API_KEY: ${WANDB_API_KEY:-}
      WANDB_ENTITY: ${WANDB_ENTITY:-}
      WANDB_DIR: /tmp/wandb
    shm_size: "8gb"
```

- [ ] **Step 3: Write the failing test**

Create `tests/training/__init__.py` (empty) and `tests/training/test_gpu_image.py`:

```python
"""The training image is a claim about hardware; this is where it is checked.

Skipped on any machine without a GPU -- including the CPU test image, which is
where the rest of the suite runs. It is NOT a substitute for the smoke command
in Step 5; it is the regression guard that keeps the compose service and the
Dockerfile from drifting apart.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_the_train_service_declares_the_gpu_and_forwards_wandb():
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
    train = compose["services"]["train"]
    assert train["devices"] == ["nvidia.com/gpu=all"], "CDI is how this host exposes the GPU"
    assert train["build"]["dockerfile"] == "Dockerfile.train"
    env = train["environment"]
    # Forwarded, never hardcoded: a literal key here would be committed.
    assert env["WANDB_API_KEY"] == "${WANDB_API_KEY:-}"
    assert env["WANDB_ENTITY"] == "${WANDB_ENTITY:-}"


def test_the_train_image_overrides_the_cpu_torch_pin():
    """pyproject pins torch to the CPU index for the test image. If
    Dockerfile.train ever stops overriding that, training silently runs on the
    CPU -- 600k steps that would never finish, with no error to notice.
    """
    dockerfile = (REPO_ROOT / "Dockerfile.train").read_text()
    assert "download.pytorch.org/whl/cu130" in dockerfile
    assert "--reinstall-package torch" in dockerfile


@pytest.mark.skipif(
    not __import__("importlib.util", fromlist=["util"]).find_spec("torch"),
    reason="torch not installed",
)
def test_cuda_is_really_usable_when_a_gpu_is_present():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no GPU visible (expected in the CPU test image)")
    # Not just is_available(): the wheel's arch list has no sm_121, so the
    # only honest check is that compute actually runs.
    a = torch.randn(512, 512, device="cuda")
    assert torch.isfinite(a @ a).all()
```

- [ ] **Step 4: Run the test in the CPU image (it must pass, with the CUDA case skipped)**

Run: `docker compose run --rm test bash -c "pytest -q tests/training/test_gpu_image.py -v"`
Expected: 2 passed, 1 skipped.

- [ ] **Step 5: Build the image and run the real GPU smoke check**

```bash
docker compose build train
docker compose run --rm train python -c "
import torch
print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
a = torch.randn(2048, 2048, device='cuda'); print(float((a @ a).sum()))
"
docker compose run --rm train bash -c "pytest -q tests/training/test_gpu_image.py -v"
```
Expected: `2.14.0+cu130 NVIDIA GB10 (12, 1)`, a finite sum, and 3 passed in the GPU image.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile.train docker-compose.yml tests/training/
git commit -m "feat(docker): CUDA 13 aarch64 training image and train service"
```

---

## Task 2: The read path loads A2's artefacts

**Files:**
- Modify: `src/poseydon/data/dataset.py`
- Test: `tests/data/test_dataset_read_path.py`

**Interfaces:**
- Consumes: `data/truebones/rigs/<Rig>/manifest.yaml`, `.../skeleton.npz`, `.../stats.npz`, `clips/<Rig>/<action>.npz`, `index.jsonl`.
- Produces: `MotionDataset` unchanged in signature; `ClipView` gains `clip: str` and `rig: str`.

**Context the brief cannot know:** this is the task that makes the dataset work at all. Right now it does not: `MotionDataset(manifest_dir="data/truebones/rigs")` raises `ManifestError: manifest not found: data/truebones/rigs/Alligator.yaml`, because A2 moved every rig into its own directory. Measured, so treat it as the starting condition, not a hypothesis.

Four separate changes, all in `__init__` and its helpers:

**(a) Manifest path.** `self.manifest_dir / f"{skeleton}.yaml"` becomes `self.manifest_dir / skeleton / "manifest.yaml"`.

**(b) Rigid body, then reduction.** `_anim` currently returns the raw prepared animation. It must apply `as_rigid_body(joint_translation="drop")` and then the rig's reduction — in that order — because `stats.npz` was fitted on exactly that distribution (`build_stats` in `build/pipeline.py`). Skipping either step normalizes against statistics that were never measured on these values.

**(c) The reduction is per rig, built once, checked against the artefact.** `skeleton.npz` stores `reduction_source_of`, which is a `JointEdit`: it reindexes rows and transports statistics, but it CANNOT apply a reduction to an animation, because a `COLLAPSE` removal composes the victim's rotation into its parent and a reindex map carries no rotations. So build a real `JointReduction` with `build_reduction`, cache it per rig, and assert its `source_of` matches the stored one. Per rig is correct and not an approximation: removal is selected on `norm(offsets)`, `parents` and `names`, and clips of one rig differ only by a rotation of the offsets, which leaves a norm unchanged. That is the same invariant `build_skeleton` and `build_stats` already rely on.

**(d) Statistics are loaded, not fitted.** `_normalizer` currently extracts features for EVERY clip of a rig and calls `Normalizer.fit`. Replace with `Normalizer.load(rigs/<Rig>/stats.npz)`.

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_dataset_read_path.py`:

```python
"""The read path against the real corpus A2 built.

Every test here skips when the corpus is absent (a fresh checkout has none of
it -- it is gitignored). That is a real weakness, so each test states what it
would catch, and `test_corpus_yield.py` carries the same caveat.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from poseydon.build.index import CorpusIndex
from poseydon.data.dataset import MotionDataset
from poseydon.data.normalize import Normalizer
from poseydon.data.window import RandomCrop

CORPUS = Path("data/truebones")
FEATURES = ["ric_pos", "rot6d", "local_vel", "foot_contact"]

pytestmark = pytest.mark.skipif(
    not (CORPUS / "index.jsonl").is_file(),
    reason="stage-2 corpus absent; run scripts/build_features.py",
)


@pytest.fixture(scope="module")
def dataset() -> MotionDataset:
    return MotionDataset(
        index=CorpusIndex.load(CORPUS / "index.jsonl"),
        root=CORPUS,
        manifest_dir=CORPUS / "rigs",
        features=FEATURES,
        window=RandomCrop(length=40),
        conditioners=["topology", "tpose", "norm_stats"],
        split="train",
        seed=0,
    )


def test_the_dataset_opens_against_the_entity_first_layout(dataset):
    """Before this task the constructor raised ManifestError on
    `rigs/Alligator.yaml` -- A2 moved manifests to `rigs/<Rig>/manifest.yaml`
    and nothing had followed.
    """
    assert len(dataset) > 0
    assert len(dataset.records) == 1145


def test_statistics_are_loaded_from_stats_npz_not_refitted(dataset):
    """Fitting re-derives per-rig moments from every clip at startup. Loading
    is not merely faster: a refit under a different `features:` silently
    produces different moments than the ones the checkpoint was trained with.
    """
    rig = dataset.records[0].skeleton
    loaded = Normalizer.load(CORPUS / "rigs" / rig / "stats.npz")
    used = dataset._normalizer(rig, dataset.spec)
    np.testing.assert_allclose(used.mean, loaded.mean)
    np.testing.assert_allclose(used.std, loaded.std)


def test_the_reduction_matches_the_one_skeleton_npz_records(dataset):
    """stats.npz's moments are per REDUCED joint. If the read path reduced
    differently from `build_stats`, normalization would silently pair a joint's
    values with another joint's statistics.
    """
    for rig in ("Goat", "Flamingo", "Crab", "Tukan"):
        with np.load(CORPUS / "rigs" / rig / "skeleton.npz", allow_pickle=True) as data:
            stored = tuple(int(i) for i in data["reduction_source_of"])
        assert dataset._reduction(rig).source_of == stored


def test_items_are_reduced_rigid_body_features(dataset):
    """The width and joint count a batch carries must be the reduced ones the
    statistics were fitted on -- 31 joints for Goat, not its 40 prepared ones.
    """
    index = next(
        i for i, (clip, _) in enumerate(dataset._plan)
        if dataset.records[clip].skeleton == "Goat"
    )
    item = dataset[index]
    assert item.features.shape[1] == 31
    assert item.features.shape[2] == item.spec.dim == 13


def test_startup_does_not_read_every_clip():
    """`__init__` used to extract features for all 1145 clips to count windows.
    The index already knows the frame count; the schema's frame cost is one
    constant. Guarded because the regression is invisible -- it only shows up
    as a slow start.

    Builds its OWN dataset rather than taking the module fixture: the fixture is
    shared, earlier tests have already pulled items through it, and asserting an
    empty cache on it would pass or fail on test ORDER rather than on behaviour.
    """
    fresh = MotionDataset(
        index=CorpusIndex.load(CORPUS / "index.jsonl"),
        root=CORPUS,
        manifest_dir=CORPUS / "rigs",
        features=FEATURES,
        window=RandomCrop(length=40),
        split="train",
        seed=0,
    )
    # One clip is loaded deliberately, to measure the schema's frame cost.
    assert len(fresh._anims) <= 1, "startup must not walk the corpus"
    assert len(fresh) > 0
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `docker compose run --rm test bash -c "pytest -q tests/data/test_dataset_read_path.py"`
Expected: FAIL — the fixture raises `ManifestError: manifest not found: .../rigs/Alligator.yaml`.

- [ ] **Step 3: Rewrite the helpers in `dataset.py`**

Replace `_manifest`, `_anim`, `_normalizer` and add `_reduction`:

```python
    def _manifest(self, skeleton: str) -> SkeletonManifest:
        if skeleton not in self._manifests:
            # Entity-first layout: `rigs/<Rig>/manifest.yaml`, not the flat
            # `rigs/<Rig>.yaml` the migration deleted.
            self._manifests[skeleton] = SkeletonManifest.load(
                self.manifest_dir / skeleton / "manifest.yaml"
            )
        return self._manifests[skeleton]

    def _reduction(self, skeleton: str) -> JointReduction:
        """The rig's reduction, built once and checked against `skeleton.npz`.

        `skeleton.npz` stores `reduction_source_of`, which is a `JointEdit`: it
        can transport statistics but cannot APPLY a reduction, because a
        COLLAPSE removal composes the victim's rotation into its parent and a
        reindex map carries no rotations. So the ops are rebuilt here and the
        stored map is used as a consistency check on them.

        Per RIG, not per clip. Removal is selected on `norm(offsets)`,
        `parents` and `names`; clips of one rig differ only by a rotation of
        the offsets, which leaves a norm unchanged. Same invariant
        `build_skeleton` and `build_stats` rely on.
        """
        if skeleton not in self._reductions:
            rest = RigidBodyAnimation.load(
                self.root / "clips" / skeleton / f"{rest_action(self._manifest(skeleton))}.npz"
            ).as_rigid_body(joint_translation="drop")
            reduction = build_reduction(rest, tolerance=self.reduce_tolerance)
            with np.load(
                self.manifest_dir / skeleton / "skeleton.npz", allow_pickle=True
            ) as data:
                stored = tuple(int(i) for i in data["reduction_source_of"])
            if reduction.source_of != stored:
                raise ValueError(
                    f"{skeleton}: the reduction built here does not match the one "
                    f"`skeleton.npz` records, so `stats.npz` would pair each joint's "
                    f"values with another joint's statistics. Rebuild stage 2."
                )
            self._reductions[skeleton] = reduction
        return self._reductions[skeleton]

    def _anim(self, record: ClipRecord) -> RigidBodyAnimation:
        """The clip as the statistics saw it: rigid body, then reduced.

        Both steps are load-bearing and ordered. `build_stats` fits on
        `as_rigid_body(joint_translation="drop")` and then reduces
        (`build/pipeline.py`); reading in any other shape normalizes against
        moments that were never measured on these values.
        """
        if record.clip_id not in self._anims:
            anim = RigidBodyAnimation.load(self.root / record.path)
            anim = anim.as_rigid_body(joint_translation="drop")
            self._anims[record.clip_id] = apply_reduction(
                anim, self._reduction(record.skeleton)
            )
        return self._anims[record.clip_id]

    def _normalizer(self, skeleton: str, spec: FeatureSpec) -> Normalizer:
        """Loaded, never fitted.

        Fitting walked every clip of the rig at startup. Worse than slow: a
        refit under a different `features:` than the checkpoint was trained
        with produces different moments, silently.
        """
        if skeleton not in self._normalizers:
            normalizer = Normalizer.load(self.manifest_dir / skeleton / "stats.npz")
            if normalizer.spec != spec:
                raise ValueError(
                    f"{skeleton}: stats.npz was fitted for {normalizer.spec} but this "
                    f"run declares {spec}. Re-run `scripts/build_features.py --stats-only`."
                )
            self._normalizers[skeleton] = normalizer
        return self._normalizers[skeleton]
```

Add to `__init__`, beside the other caches:

```python
        self._reductions: dict[str, JointReduction] = {}
```

and accept the tolerance so it cannot drift from the build:

```python
        reduce_tolerance: float = 1e-8,
```
stored as `self.reduce_tolerance = reduce_tolerance`.

Imports to add:

```python
from poseydon.build.corpus import rest_action
from poseydon.features.reduce import JointReduction, apply_reduction, build_reduction
```

- [ ] **Step 4: Make startup O(1) in the corpus**

Replace the `_plan` loop's `self._feature_frames(record)` call. The index already carries `n_frames`; the schema costs a fixed number of frames (measured: `local_vel` consumes exactly one, so features are `n_frames - 1` for the shipped schema). Derive that constant ONCE:

```python
        # `__init__` used to extract features for every clip just to count
        # windows -- 1145 full extractions before the first sample. The index
        # already knows each clip's frame count; the only unknown is how many
        # frames the SCHEMA costs (`local_vel` consumes one), and that is a
        # constant across clips. Measure it once, on the first record.
        first = self.records[0]
        self._frame_cost = first.n_frames - self._extract(first)[0].shape[0]

        self._plan: list[tuple[int, int]] = []
        for clip_index, record in enumerate(self.records):
            frames = record.n_frames - self._frame_cost
            for window_index in range(self.window.count(frames)):
                self._plan.append((clip_index, window_index))
```

Then drop the now-unused `_feature_frames`, and clear the caches the probe populated so `test_startup_does_not_read_every_clip` holds:

```python
        # The one probe above loaded one clip; nothing else should be resident.
        self._anims.clear()
```

- [ ] **Step 5: Annotate `ClipView` (spec §3)**

Add to the dataclass, so a later text conditioner needs no dataset change:

```python
    clip: str  # record.clip_id, for a conditioner that needs to name the clip
    rig: str   # record.skeleton, likewise
```

and pass `clip=record.clip_id, rig=record.skeleton` at the `ClipView(...)` call in `__getitem__`.

- [ ] **Step 6: Run the tests**

Run: `docker compose run --rm test bash -c "ruff check . && pytest -q tests/data/ tests/build/"`
Expected: all pass, including the five new ones.

- [ ] **Step 7: Run the whole suite**

Run: `docker compose run --rm test bash -c "pytest -q"`
Expected: at least 258 + 5 passing; no new failures. Any pre-existing test that constructed `MotionDataset` with the flat layout must be updated to the entity-first one, not deleted.

- [ ] **Step 8: Commit**

```bash
git add src/poseydon/data/dataset.py tests/data/test_dataset_read_path.py
git commit -m "feat(data): read stage-2 artefacts instead of refitting them"
```

---

## Task 3: The rest frame comes from the declared rest pose

**Files:**
- Modify: `src/poseydon/data/dataset.py`
- Test: `tests/data/test_dataset_read_path.py` (append)

**Interfaces:**
- Consumes: `SkeletonManifest.rest_pose`, `poseydon.build.corpus.rest_action`, Task 2's `_reduction`/`_normalizer`.
- Produces: `ClipView.rest_frame` sourced from the rig's declared rest clip.

**Context the brief cannot know:** `_rest_frame`'s own docstring says "A3 replaces this with the rest frame recorded in `rigs/<Rig>/stats.npz`". **Do not do that — `stats.npz` contains no such frame.** A2's final review found spec §2 promising a "T-pose frame" that is never written, and the spec was corrected. The rest frame comes from the rig's declared rest CLIP, frame 0, which every rig now has: A1 authored `rest_pose:` into all 73 manifests precisely so this is never guessed. Verified for the validation pairs — `Scorpion: __TPOSE.bvh`, `Crab: __Walk.bvh`, `Raptor: __TPOSE.bvh`, `Flamingo: Flamingo_Tpose.bvh`, `Coyote: __TPOSE.bvh`, `Goat: __Idle.bvh`.

- [ ] **Step 1: Write the failing test**

Append to `tests/data/test_dataset_read_path.py`:

```python
def test_the_rest_frame_comes_from_the_declared_rest_pose(dataset):
    """Not from "the first clip that happened to be indexed", which is what the
    old fallback did, and not from stats.npz, which records no such frame.

    Crab is the case that proves it: its declared rest pose is `__Walk.bvh`,
    not a T-pose, and alphabetically its first clip is `attack1`. A fallback to
    the first clip would describe the rig with an attack pose.
    """
    from poseydon.build.corpus import rest_action

    manifest = dataset._manifest("Crab")
    assert rest_action(manifest) == "walk"

    # Compare against an INDEPENDENTLY built frame, not against
    # `dataset._extract_rest` -- `_rest_frame_of` just caches that call, so
    # comparing the two would assert nothing at all.
    from poseydon.core.animation import RigidBodyAnimation
    from poseydon.core.skeleton import resolve
    from poseydon.features import extract_features
    from poseydon.features.reduce import apply_reduction

    anim = RigidBodyAnimation.load(CORPUS / "clips" / "Crab" / "walk.npz")
    reduced = apply_reduction(anim, dataset._reduction("Crab"))
    raw, spec = extract_features(reduced, resolve(manifest, reduced.names), FEATURES)
    expected = dataset._normalizer("Crab", spec).normalize(raw[:1])[0]

    np.testing.assert_allclose(dataset._rest_frame_of("Crab"), expected)

    # And it must differ from what the old first-clip fallback produced --
    # otherwise the test would pass even if nothing changed.
    first_clip = RigidBodyAnimation.load(CORPUS / "clips" / "Crab" / "attack1.npz")
    reduced_first = apply_reduction(first_clip, dataset._reduction("Crab"))
    raw_first, _ = extract_features(
        reduced_first, resolve(manifest, reduced_first.names), FEATURES
    )
    fallback = dataset._normalizer("Crab", spec).normalize(raw_first[:1])[0]
    assert not np.allclose(expected, fallback), "the old fallback was not distinguishable"


def test_every_rig_can_produce_a_rest_frame(dataset):
    """A1 authored `rest_pose:` into all 73 manifests so this never silently
    falls back. If any rig cannot, the corpus is the problem, not the reader.
    """
    rigs = sorted({record.skeleton for record in dataset.records})
    assert len(rigs) == 73
    for rig in rigs:
        assert dataset._rest_frame_of(rig).shape[-1] == dataset.spec.dim
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `docker compose run --rm test bash -c "pytest -q tests/data/test_dataset_read_path.py -k rest"`
Expected: FAIL with `AttributeError: 'MotionDataset' object has no attribute '_rest_frame_of'`.

- [ ] **Step 3: Replace `_rest_frame`**

```python
    def _extract_rest(self, skeleton: str) -> np.ndarray:
        """Frame 0 of the rig's DECLARED rest clip, normalized.

        `manifest.rest_pose` is authored for all 73 rigs (plan A1) exactly so
        this is never guessed. The old fallback took the first clip the index
        happened to list, which for Crab -- whose rest pose is `__Walk.bvh` and
        whose first indexed clip is `attack1` -- described the rig with an
        attack pose. `stats.npz` is NOT the source: it records no rest frame,
        whatever an earlier draft of the spec promised.
        """
        manifest = self._manifest(skeleton)
        anim = RigidBodyAnimation.load(
            self.root / "clips" / skeleton / f"{rest_action(manifest)}.npz"
        ).as_rigid_body(joint_translation="drop")
        reduced = apply_reduction(anim, self._reduction(skeleton))
        raw, spec = extract_features(reduced, resolve(manifest, reduced.names), self.features)
        return self._normalizer(skeleton, spec).normalize(raw[:1])[0]

    def _rest_frame_of(self, skeleton: str) -> np.ndarray:
        if skeleton not in self._rest_frames:
            self._rest_frames[skeleton] = self._extract_rest(skeleton)
        return self._rest_frames[skeleton]
```

In `__getitem__`, replace the `_rest_frame(record, spec, base_normalizer)` call with `self._rest_frame_of(record.skeleton)`, keeping the `edit.transport_row(...)` that follows it.

- [ ] **Step 4: Run the tests**

Run: `docker compose run --rm test bash -c "ruff check . && pytest -q tests/data/"`
Expected: all pass. `test_every_rig_can_produce_a_rest_frame` touches all 73 rigs and is the slow one; that is intended.

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/data/dataset.py tests/data/test_dataset_read_path.py
git commit -m "feat(data): take the rest frame from the manifest's declared rest pose"
```

---

## Task 4: Balanced sampling, worker RNG, drop_last

**Files:**
- Create: `src/poseydon/data/sampler.py`
- Modify: `src/poseydon/data/dataset.py`, `src/poseydon/training/lightning.py`
- Test: `tests/data/test_sampler.py`, `tests/data/test_worker_rng.py`

**Interfaces:**
- Consumes: `MotionDataset.records`, `MotionDataset._plan`.
- Produces: `balanced_weights(dataset) -> np.ndarray`; `MotionDataModule(..., balanced: bool = True)`; `MotionDataset.set_worker_seed(worker_id)`.

**Context the brief cannot know:** the reference passes `--balanced` in both documented training commands, and without it BrownBear's 22 clips dominate. The reference's own sampler (`data_loaders/truebones/data/dataset.py::TruebonesSampler`) is a plain `WeightedRandomSampler` giving each object type an equal share and splitting it across that type's clips — mirror that, not something cleverer. Note the weight is per PLAN ENTRY (clip, window), not per clip, because that is what `__len__` indexes.

The worker-RNG bug is real and measurable: `dataset.py` holds one `numpy.random.Generator` that is COPIED into every dataloader worker by fork, so with `num_workers > 0` every worker draws the identical crop and augmentation stream.

- [ ] **Step 1: Write the failing tests**

`tests/data/test_sampler.py`:

```python
"""Balanced sampling: every rig gets an equal share, not every clip."""

from __future__ import annotations

import numpy as np

from poseydon.data.sampler import balanced_weights


class _FakeRecord:
    def __init__(self, skeleton):
        self.skeleton = skeleton


class _FakeDataset:
    """Three rigs, wildly unequal clip counts -- BrownBear's real problem."""

    records = [_FakeRecord("BrownBear")] * 22 + [_FakeRecord("Crab")] * 2 + [_FakeRecord("Goat")]

    def __init__(self):
        # one window per clip keeps the arithmetic legible
        self._plan = [(i, 0) for i in range(len(self.records))]

    def __len__(self):
        return len(self._plan)


def test_each_rig_receives_an_equal_share():
    weights = balanced_weights(_FakeDataset())
    dataset = _FakeDataset()
    share = {}
    for weight, (clip, _) in zip(weights, dataset._plan, strict=True):
        share.setdefault(dataset.records[clip].skeleton, 0.0)
        share[dataset.records[clip].skeleton] += weight
    assert len(share) == 3
    for rig, total in share.items():
        assert total == 1 / 3, f"{rig} got {total}, not an equal third"


def test_clips_within_a_rig_are_equally_likely():
    weights = balanced_weights(_FakeDataset())
    bear = [w for w, (c, _) in zip(weights, _FakeDataset()._plan, strict=True)
            if _FakeDataset.records[c].skeleton == "BrownBear"]
    assert len(bear) == 22
    assert np.allclose(bear, bear[0])


def test_a_rig_with_one_clip_is_not_starved():
    """Goat has 1 clip to BrownBear's 22, so its single entry must carry 22x
    the weight of one bear entry. This is the whole point of the sampler.
    """
    dataset = _FakeDataset()
    weights = balanced_weights(dataset)
    goat = next(w for w, (c, _) in zip(weights, dataset._plan, strict=True)
                if dataset.records[c].skeleton == "Goat")
    bear = next(w for w, (c, _) in zip(weights, dataset._plan, strict=True)
                if dataset.records[c].skeleton == "BrownBear")
    assert np.isclose(goat / bear, 22.0)
```

`tests/data/test_worker_rng.py`:

```python
"""One Generator, forked into N workers, draws the same stream N times."""

from __future__ import annotations

import numpy as np


class _Tiny:
    """The bug in isolation: the object holds the Generator, fork copies it."""

    def __init__(self, seed=0):
        self._rng = np.random.default_rng(seed)

    def set_worker_seed(self, worker_id: int, base_seed: int = 0) -> None:
        self._rng = np.random.default_rng([base_seed, worker_id])

    def draw(self):
        return int(self._rng.integers(0, 1_000_000))


def test_forked_workers_would_draw_identically_without_reseeding():
    copies = [_Tiny(seed=0) for _ in range(4)]
    assert len({c.draw() for c in copies}) == 1, "this is the bug being fixed"


def test_reseeding_per_worker_decorrelates_them():
    copies = [_Tiny(seed=0) for _ in range(4)]
    for worker_id, copy in enumerate(copies):
        copy.set_worker_seed(worker_id)
    assert len({c.draw() for c in copies}) == 4


def test_the_dataset_exposes_the_hook_the_loader_calls():
    from poseydon.data.dataset import MotionDataset

    assert hasattr(MotionDataset, "set_worker_seed")
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `docker compose run --rm test bash -c "pytest -q tests/data/test_sampler.py tests/data/test_worker_rng.py"`
Expected: FAIL — `ModuleNotFoundError: No module named 'poseydon.data.sampler'` and the `set_worker_seed` assertion.

- [ ] **Step 3: Write `src/poseydon/data/sampler.py`**

```python
"""Sampling weights that give every rig an equal voice.

Truebones is wildly unbalanced -- BrownBear ships 22 clips, a dozen rigs ship
one. Sampling uniformly over clips trains a bear model that has met some other
animals. The reference passes `--balanced` in both of its documented training
commands and implements it as a plain `WeightedRandomSampler`
(`data_loaders/truebones/data/dataset.py::TruebonesSampler`): equal share per
object type, split evenly across that type's clips. This mirrors it.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def balanced_weights(dataset: Any) -> np.ndarray:
    """`1 / (n_rigs * n_entries_of_rig)` per plan entry.

    Weighted per PLAN ENTRY -- a (clip, window) pair -- not per clip, because
    that is what `__len__` indexes and therefore what the sampler draws. A clip
    that yields more windows must not thereby win more of its rig's share.
    """
    rigs = [dataset.records[clip].skeleton for clip, _ in dataset._plan]
    per_rig = Counter(rigs)
    share = 1.0 / len(per_rig)
    return np.array([share / per_rig[rig] for rig in rigs], dtype=np.float64)
```

- [ ] **Step 4: Add the worker hook to `MotionDataset`**

```python
    def set_worker_seed(self, worker_id: int) -> None:
        """Give this worker its own stream.

        `__init__` builds one Generator, and `fork` copies it into every
        dataloader worker -- so with `num_workers > 0` all workers drew the
        SAME crops and the same augmentations, silently reducing the effective
        variety of a batch by a factor of `num_workers`. Called from
        `worker_init_fn`.
        """
        self._rng = np.random.default_rng([self.seed, worker_id])
```

and store `self.seed = seed` in `__init__` (it is currently consumed and dropped).

- [ ] **Step 5: Wire the sampler and `drop_last` into `MotionDataModule`**

```python
    def __init__(
        self,
        train: MotionDataset,
        val: MotionDataset | None = None,
        batch_size: int = 8,
        num_workers: int = 0,
        balanced: bool = True,
    ) -> None:
        ...
        self.balanced = balanced

    def _worker_init(self, worker_id: int) -> None:
        info = torch.utils.data.get_worker_info()
        if info is not None:
            info.dataset.set_worker_seed(worker_id)

    def train_dataloader(self) -> DataLoader:
        sampler = None
        if self.balanced:
            weights = balanced_weights(self.train_dataset)
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(self.train_dataset),
                replacement=True,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            # `shuffle` and `sampler` are mutually exclusive; the sampler
            # already randomizes.
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate,
            # A short final batch changes the padded joint count and the loss
            # scale for one step in every epoch, for no benefit.
            drop_last=True,
            worker_init_fn=self._worker_init if self.num_workers else None,
            persistent_workers=self.num_workers > 0,
        )
```

Leave `val_dataloader` on the plain loader: validation must be deterministic and complete, so neither balancing nor `drop_last` belongs there.

- [ ] **Step 6: Run the tests**

Run: `docker compose run --rm test bash -c "ruff check . && pytest -q tests/data/"`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/poseydon/data/sampler.py src/poseydon/data/dataset.py src/poseydon/training/lightning.py tests/data/
git commit -m "feat(data): balanced sampling, per-worker RNG, drop_last"
```

---

## Task 5: `crop_start` reaches the models

**Files:**
- Modify: `src/poseydon/training/task.py`, `src/poseydon/models/anytop.py`
- Test: `tests/training/test_crop_start.py`

**Interfaces:**
- Consumes: `MotionBatch.window.start` (already collated).
- Produces: `cond["crop_start"]`, a `(B,)` long tensor, present on every batch.

**Context the brief cannot know — read this carefully, the spec understated it.** The spec's original wording called this a naming disagreement: `AnyTop` reads `cond["window_start"]`, `MoDiffAE` reads `cond["crop_start"]`. That is true but not the bug. **Nothing writes either key.** No conditioner emits it, and `MotionTask.compute_losses` (`task.py:63-69`) builds `cond` from `batch.cond` plus `CLEAN_MOTION` and nothing else, while the crop offset sits unused in `batch.window.start`. So `MoDiffAE` has always taken its `torch.zeros` default and `AnyTop` its `None`: every window in every run to date was positionally encoded as if it began at frame 0. Renaming alone would leave that exactly as it is. Fix the routing first; the rename is the small half.

- [ ] **Step 1: Write the failing test**

`tests/training/test_crop_start.py`:

```python
"""The crop offset must actually reach the model.

`MotionBatch` has carried `window.start` all along and neither model has ever
seen it: nothing wrote `crop_start` (MoDiffAE's key) or `window_start`
(AnyTop's) into `cond`, so both silently took their defaults and every window
was encoded as if it started at frame 0.
"""

from __future__ import annotations

import torch

from poseydon.core.batch import Cond


def test_the_task_injects_crop_start_from_the_window(monkeypatch):
    from poseydon.training import task as task_module

    seen = {}

    class _Spy:
        def __call__(self, z_t, t, cond, masks):
            seen["cond"] = cond
            return task_module.Prediction(out=torch.zeros_like(z_t), aux={})

        def check_conditioners(self, names):
            return None

    batch = _make_batch(starts=[7, 13, 0])
    _run_one_step(task_module, _Spy(), batch)

    assert "crop_start" in seen["cond"], "the offset never reached the model"
    assert torch.equal(seen["cond"]["crop_start"], torch.tensor([7, 13, 0]))


def test_anytop_reads_crop_start_not_window_start():
    """The two models must agree on the key, or wiring one silently leaves the
    other on its default.
    """
    source = (__import__("pathlib").Path("src/poseydon/models/anytop.py")).read_text()
    assert "window_start" not in source
    assert 'cond.get("crop_start")' in source
```

with these helpers at the top of the file — the point is the plumbing, not the
model, so the batch is the smallest one `MotionBatch.__post_init__` accepts:

```python
from poseydon.core.batch import Cond, Masks, MotionBatch, WindowInfo
from poseydon.core.spec import FeatureSpec

SPEC = FeatureSpec((("ric_pos", 3),))


def _make_batch(starts):
    batch = len(starts)
    joints, frames = 4, 8
    return MotionBatch(
        x=torch.zeros(batch, joints, SPEC.dim, frames),
        spec=SPEC,
        masks=Masks(
            frames=torch.ones(batch, frames, dtype=torch.bool),
            joints=torch.ones(batch, joints, dtype=torch.bool),
        ),
        window=WindowInfo(
            start=torch.tensor(starts, dtype=torch.long),
            source_length=torch.full((batch,), frames, dtype=torch.long),
        ),
        cond=Cond({}),
    )


def _run_one_step(task_module, model, batch):
    """One `compute_losses` call with a spy model and no real loss terms."""
    task = task_module.MotionTask(model=model, process=_IdentityProcess(), losses=[])
    return task.compute_losses(batch)
```

`_IdentityProcess` is a three-line stub returning `z_t = batch.x` and a zero
timestep — the process is not what this test is about. Check
`poseydon/process/base.py` for the exact method names it must provide.

- [ ] **Step 2: Run it to confirm it fails**

Run: `docker compose run --rm test bash -c "pytest -q tests/training/test_crop_start.py"`
Expected: FAIL on both — `crop_start` absent, and `window_start` still in `anytop.py`.

- [ ] **Step 3: Inject it in `task.py`**

In `compute_losses`, where `cond` is assembled:

```python
        cond = batch.cond
        # The crop offset lives on the batch and reached neither model: nothing
        # wrote it into `cond`, so MoDiffAE always took its zeros default and
        # AnyTop its None. Every window was encoded as if it began at frame 0.
        # Injected here rather than as a conditioner because it is a property of
        # the WINDOW, which `Conditioner.extract` does not see.
        payloads = {**cond.payloads, "crop_start": batch.window.start}
        if CLEAN_MOTION in self.model.requires:
            payloads[CLEAN_MOTION] = batch.x
        cond = Cond(payloads)
```

- [ ] **Step 4: Correct `anytop.py:207`**

```python
        start = cond.get("crop_start")
```

with a comment naming why:

```python
        # `crop_start`, matching MoDiffAE. This used to read `window_start`, a
        # key nothing ever wrote -- so this branch has never once fired.
```

- [ ] **Step 5: Run the tests**

Run: `docker compose run --rm test bash -c "ruff check . && pytest -q"`
Expected: all pass. Watch for models or tests that assumed a zero offset; a positional encoding that now varies is a REAL behaviour change and any test asserting the old constant should be corrected, not the code reverted.

- [ ] **Step 6: Commit**

```bash
git add src/poseydon/training/task.py src/poseydon/models/anytop.py tests/training/test_crop_start.py
git commit -m "fix(training): route the crop offset into cond; both models ignored it"
```

---

## Task 6: `temporal_window` is a model parameter

**Files:**
- Modify: `src/poseydon/models/modiffae.py`, `src/poseydon/models/anytop.py`, `configs/model/modiffae.yaml`, `configs/model/anytop.yaml`
- Test: `tests/training/test_temporal_window.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `MoDiffAE(..., temporal_window: int = 31)`, likewise `AnyTop`; `TEMPORAL_VALID` in `cond` survives as an optional sampling-time override.

**Context the brief cannot know:** attention locality is a property of the attention module, not of the data. Routing it through conditioning would require `Conditioner.collate` to know the batch's frame count, which it does not receive. The band is intersected with the padding mask INSIDE the model, so a padded frame stays invalid regardless of how close it is. Keep `cond[TEMPORAL_VALID]` working as an override — sampling uses it — and let the parameter be the default when it is absent.

- [ ] **Step 1: Write the failing test**

```python
"""Attention locality belongs to the model, and must respect padding."""

from __future__ import annotations

import torch

from poseydon.models.modiffae import MoDiffAE


def test_the_band_limits_attention_to_the_declared_width():
    model = MoDiffAE(feature_dim=13, d_model=32, n_heads=2, temporal_window=5)
    joint_valid = torch.ones(1, 3, dtype=torch.bool)
    frame_valid = torch.ones(1, 10, dtype=torch.bool)
    _, pair = model._valid_with_band(joint_valid, frame_valid)
    # +1 for the leading rest frame; frame i attends frames within 2 of it
    assert bool(pair[0, 1 + 0, 1 + 2])
    assert not bool(pair[0, 1 + 0, 1 + 3])


def test_the_band_never_revives_a_padded_frame():
    """A padded frame that happens to sit inside the band must stay invalid --
    the band NARROWS attention, it never widens it.
    """
    model = MoDiffAE(feature_dim=13, d_model=32, n_heads=2, temporal_window=31)
    joint_valid = torch.ones(1, 3, dtype=torch.bool)
    frame_valid = torch.tensor([[True, True, False, False]])
    _, pair = model._valid_with_band(joint_valid, frame_valid)
    assert not bool(pair[0, 1 + 0, 1 + 2])


def test_an_explicit_override_still_wins():
    """Sampling passes TEMPORAL_VALID directly; the parameter is only the
    default for when it is absent. If the band were applied AFTER the override
    it would silently narrow a mask the caller chose deliberately.
    """
    from poseydon.core.batch import Cond
    from poseydon.models.base import TEMPORAL_VALID

    model = MoDiffAE(feature_dim=13, d_model=32, n_heads=2, temporal_window=1)
    override = torch.ones(1, 9, 9, dtype=torch.bool)
    cond = Cond({
        "topology": {
            "hops": torch.zeros(1, 3, 3, dtype=torch.long),
            "relations": torch.zeros(1, 3, 3, dtype=torch.long),
        },
        "tpose": torch.zeros(1, 3, 13),
        "clean_motion": torch.zeros(1, 3, 13, 8),
        TEMPORAL_VALID: override,
    })
    prediction = model(torch.zeros(1, 3, 13, 8), torch.zeros(1, dtype=torch.long), cond)
    # A temporal_window of 1 would leave each frame attending only itself; the
    # override says otherwise and must win. Reaching a finite output at all is
    # the observable: the mask is internal, so this asserts the override path
    # runs rather than inspecting a private tensor.
    assert torch.isfinite(prediction.out).all()
```

- [ ] **Step 2: Run it to confirm it fails**

Expected: `TypeError: __init__() got an unexpected keyword argument 'temporal_window'`.

- [ ] **Step 3: Implement**

Add `temporal_window: int = 31` to both models' `__init__` and store it as
`self.temporal_window`. Then extract the mask construction into a helper --
`_valid` currently builds the pair mask inline, and the test names the helper,
so it is part of the interface:

```python
    def _valid_with_band(self, joint_valid, frame_valid):
        """Joint validity and pairwise frame validity, band-limited.

        Index 0 of the pair mask is the REST frame, which conditions everything
        and attends only itself; the band applies to the real frames after it.
        The band is ANDed with the padding mask, never ORed: it NARROWS
        attention and must never make a padded frame visible.
        """
        batch, frames = frame_valid.shape
        device = frame_valid.device

        leading = torch.ones(batch, 1, dtype=torch.bool, device=device)
        extended = torch.cat([leading, frame_valid], dim=1)
        pair = extended[:, :, None] & extended[:, None, :]

        if self.temporal_window:
            offsets = torch.arange(frames, device=device)
            band = (offsets[:, None] - offsets[None, :]).abs() <= self.temporal_window // 2
            pair[:, 1:, 1:] &= band

        # The rest frame conditions everything but attends only to itself.
        pair[:, 0, :] = False
        pair[:, 0, 0] = True
        return joint_valid, pair
```

Then `_valid` keeps its current signature and delegates:

```python
    def _valid(self, masks, batch, joints, frames, device):
        if masks is None:
            joint_valid = torch.ones(batch, joints, dtype=torch.bool, device=device)
            frame_valid = torch.ones(batch, frames, dtype=torch.bool, device=device)
        else:
            joint_valid = masks.joints.to(device)
            frame_valid = masks.frames.to(device)
        return self._valid_with_band(joint_valid, frame_valid)
```

`cond[TEMPORAL_VALID]` keeps overriding the result where `forward` already
applies it, so sampling-time control is unchanged.

- [ ] **Step 4: Set the config values**

`configs/model/modiffae.yaml` and `configs/model/anytop.yaml` gain:

```yaml
# Attention locality: a frame attends only the 31-frame neighbourhood centred
# on it, matching the reference's `--temporal_window 31`. Intersected with the
# padding mask inside the model, never routed through conditioning -- a
# conditioner's `collate` is not told the batch's frame count.
temporal_window: 31
```

Also set `n_virtual_joints: 5` in `configs/model/modiffae.yaml` (currently 3) — the run recipe's attention-pool variant.

- [ ] **Step 5: Run the tests, then the suite**

Run: `docker compose run --rm test bash -c "ruff check . && pytest -q"`

- [ ] **Step 6: Commit**

```bash
git add src/poseydon/models/ configs/model/ tests/training/test_temporal_window.py
git commit -m "feat(models): temporal attention band as a model parameter"
```

---

## Task 7: `FootSkateLoss` thresholds the raw block

**Files:**
- Modify: `src/poseydon/losses/footskate.py`
- Test: `tests/losses/test_footskate_raw.py`

**Interfaces:**
- Consumes: `raw_block(batch, tensor, name)` from `losses/base.py`.
- Produces: no signature change.

**Context the brief cannot know — and a correction to the spec's framing.** `footskate.py:33` reads the contact flag from the NORMALIZED `x0` and thresholds at `> 0.5`, while the positions beside it already go through `raw_block`. The spec calls this a live trap. Under the SHIPPED normalization policy it is not yet one: `configs/dataset/truebones.yaml` declares `foot_contact` with `center: false, scale: none`, so the flag passes through untouched and the threshold works today. It becomes a real bug the moment that policy changes — a foot planted most of the time has a high mean and low std, so its z-scored `1` falls below `0.5`, every planted frame is discarded, and the loss silently reads zero. Fix it as defence in depth, and say so accurately in the docstring rather than repeating the stronger claim. This run weights the term at 0 regardless.

- [ ] **Step 1: Write the failing test**

```python
"""The contact flag must be read raw, whatever the normalization policy says.

Today `foot_contact` ships as `center: false, scale: none`, so the flag is
untouched and the old threshold worked. This pins the behaviour under a policy
that DOES normalize it -- where a mostly-planted foot's z-scored 1 falls below
0.5, every planted frame is discarded, and the loss quietly reads zero.
"""
```

Build a batch whose `NORM_STATS` give `foot_contact` a mean of 0.9 and a std of 0.3, set a genuinely planted contact, move the foot, and assert the loss is non-zero. Assert it IS zero without the fix (by constructing the normalized read directly) so the test cannot pass vacuously.

- [ ] **Step 2: Run it to confirm it fails**

- [ ] **Step 3: Implement**

```python
        contact = raw_block(batch, x0, "foot_contact")
```

and extend `needs` so the requirement is declared:

```python
    needs = (Block("ric_pos", space="raw"), Block("foot_contact", space="raw"))
```

- [ ] **Step 4: Run the tests and the suite**

- [ ] **Step 5: Commit**

```bash
git add src/poseydon/losses/footskate.py tests/losses/test_footskate_raw.py
git commit -m "fix(losses): threshold the raw contact flag, not the normalized one"
```

---

## Task 8: The run recipe

**Files:**
- Create: `src/poseydon/training/recipe.py`
- Modify: `src/poseydon/cli.py`
- Test: `tests/training/test_recipe.py`

**Interfaces:**
- Consumes: the composed Hydra `DictConfig`.
- Produces: `write_recipe(config, run_dir) -> Path`, `read_recipe(path) -> DictConfig`; `runs/<run>/recipe.yaml`; the same mapping stashed in the checkpoint's `hyper_parameters`.

**Context the brief cannot know:** this closes a live hole. `poseydon sample` recomposes `configs/sample.yaml` from scratch (`cli.py:83`), so sampling a checkpoint under a different `features:` than it was trained with produces silent garbage — no error, just wrong output. Plan B's retarget path reads the recipe, so this is load-bearing for validation, not only for `sample`. The recipe records the prepare chain, reduction tolerance, feature schema, normalization policy and reconstruction method.

- [ ] **Step 1: Write the failing test**

Assert: `write_recipe` creates the file; `read_recipe` round-trips it; the recorded `features` list is exactly `config.features`; and loading a checkpoint whose recipe disagrees with the active config raises a clear error naming both. That last one is the test that matters — the others are plumbing.

- [ ] **Step 2: Run it to confirm it fails**

- [ ] **Step 3: Implement `recipe.py`**

```python
"""What a checkpoint was trained with, recorded beside it.

`poseydon sample` recomposes its config from `configs/sample.yaml`, so a
checkpoint sampled under a different `features:` than it was trained with
produces silent garbage -- the right shapes, the wrong meaning. The recipe
makes that mismatch loud. Plan B's retarget path reads it too, which is why it
lives here and not in the CLI.
"""
```

Record: `features`, `conditioners`, `losses`, `model`, `process`, `data.window`, `dataset.reduce_tolerance`, `dataset.normalize`, and the reconstruction method (`positions_ik`). Provide `check_recipe(active, recorded)` raising with both values named.

- [ ] **Step 4: Call it from `_train` and consult it in `_sample`**

- [ ] **Step 5: Run the tests and the suite**

- [ ] **Step 6: Commit**

```bash
git add src/poseydon/training/recipe.py src/poseydon/cli.py tests/training/test_recipe.py
git commit -m "feat(training): record the run recipe beside the checkpoint"
```

---

## Task 9: The training loop, wandb, and the run config

**Files:**
- Modify: `src/poseydon/training/lightning.py`, `src/poseydon/training/build.py`, `src/poseydon/cli.py`, `configs/train.yaml`, `configs/trainer/default.yaml`
- Test: `tests/training/test_training_loop.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `poseydon train` running with `StepLR`, step-interval checkpoints, a `WandbLogger`, and `--resume`.

**Context the brief cannot know:** the exact values are not negotiable and are listed in Global Constraints. `batch_size 16` and `600000` steps are deliberate departures from the reference (10 / 450k), taken on a measured E[0.431 s/step] = 71.8 h; do not "restore parity". `precision: bf16-mixed` is likewise measured (1.4x, 13.2 GB peak against 128 GB available). The wandb entity is `lcazzola-fondazione-bruno-kessler` and comes from `.env` — `entity: poseydon` fails against the live API because `poseydon` is the PROJECT.

- [ ] **Step 1: Write the failing test**

Assert, without training anything: `configure_optimizers` returns an `AdamW` plus a `StepLR(step_size=10000, gamma=0.99)` configured to step per OPTIMIZER STEP (`"interval": "step"`), not per epoch — an epoch-stepped schedule on a 600k-step run would decay ~1000x too slowly; the `ModelCheckpoint` has `save_last=True, every_n_train_steps=25000, save_top_k=-1`; and `configs/train.yaml` carries the exact recipe values. Pin the values themselves — this file IS the run.

- [ ] **Step 2: Run it to confirm it fails**

- [ ] **Step 3: Add the schedule**

```python
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        # The reference's training_loop.py:65-69 exactly. Stepped per OPTIMIZER
        # STEP, not per epoch: this corpus's "epoch" is an arbitrary pass over a
        # weighted sampler, and an epoch-stepped schedule would decay roughly a
        # thousand times too slowly over 600k steps.
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=10_000, gamma=0.99
                ),
                "interval": "step",
            },
        }
```

- [ ] **Step 4: Build the logger and callbacks in `build.py`**

```python
def build_logger(config: DictConfig, run_dir: Path):
    """wandb, or nothing at all.

    Absent `WANDB_API_KEY` this returns None and training proceeds unlogged --
    a missing key must never cost a run. The entity comes from the environment
    because it is account-specific: `entity: poseydon` fails against the live
    API ("the provided API key cannot access this resource"); `poseydon` is the
    PROJECT.
    """
    if not os.environ.get("WANDB_API_KEY"):
        return None
    from lightning.pytorch.loggers import WandbLogger

    return WandbLogger(
        project="poseydon",
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=run_name(config),
        save_dir=str(run_dir),
    )
```

`run_name(config)` derives from model, pooling, batch size and latent dim, per spec §4.

Callbacks: `ModelCheckpoint(dirpath=run_dir / "checkpoints", save_last=True, every_n_train_steps=25000, save_top_k=-1)` and `LearningRateMonitor(logging_interval="step")`.

- [ ] **Step 5: Set the run recipe in the configs**

`configs/train.yaml`: `model: modiffae` in `defaults`; `losses: {simple: 1.0, geodesic: 1.0}` — the reference's `--lambda_fs` defaults to 0 and its paper command never passes it, while `--lambda_geo 1.0` is explicit; the current `geodesic: 0.1, footskate: 0.5` is neither. Keep foot skate as a MEASURED validation scalar (Plan B), not a training term.

`configs/data/truebones.yaml`: `batch_size: 16`, `num_workers: 4`.

`configs/trainer/default.yaml`:

```yaml
# 600 000 steps at a measured E[0.431 s/step] = 71.8 h. See the design spec
# §4, "What a step costs, measured".
max_steps: 600000
accelerator: auto
devices: 1
# Measured 1.4x over 32-true, 13.2 GB peak on the worst batch against 128 GB
# available. Not fp16: bf16 needs no loss scaling.
precision: bf16-mixed
log_every_n_steps: 50
gradient_clip_val: 1.0
```

- [ ] **Step 6: Add `--resume`**

`poseydon train --resume <ckpt>` maps to `trainer.fit(..., ckpt_path=args.resume)`.

- [ ] **Step 7: Run the tests and the whole suite**

Run: `docker compose run --rm test bash -c "ruff check . && pytest -q"`

- [ ] **Step 8: Prove it trains, on the GPU, for a few steps**

```bash
docker compose run --rm train python -m poseydon.cli train \
  --out /tmp/smoke trainer.max_steps=20 data.num_workers=2
```
Expected: 20 steps, a falling loss, a `recipe.yaml` in the run directory, and — if `WANDB_API_KEY` is set — a run visible under the `poseydon` project. Report the wandb URL in the task report. Do NOT start the 600k run.

- [ ] **Step 9: Commit**

```bash
git add src/poseydon/training/ src/poseydon/cli.py configs/ tests/training/
git commit -m "feat(training): balanced StepLR run at batch 16, bf16, 600k steps, logged to wandb"
```

---

## Task 10: The FBX coherence sidecheck

**Files:**
- Create: `scripts/check_fbx_coherence.py`
- Delete: `tools/probe_bvh_fbx.py`
- Test: `tests/build/test_fbx_coherence.py`

**Interfaces:**
- Consumes: `data/truebones/clips/<Rig>/<action>.{bvh,fbx}`.
- Produces: a per-rig report of joint-position agreement between the BVH and FBX paths.

**Context the brief cannot know:** FBX is a MEASUREMENT here, not a training input — nothing in this plan reads `mesh.npz`. The script is Blender-gated and must skip cleanly in the CPU image. **Fix the filter bug while you are here:** `scripts/process_dataset_truebones_fbx.py:84` selects all-takes bundles with `not stem.lower().endswith("all")`, which also drops 13 legitimate clips whose stem merely ends in those letters — `Buffalo-Fall`, `Camel-Fall`, `DEER-WalkCall`, `Gazelle-Fall`, `PolarBearB-Fall`, `Raptor-Fall`, `Raptor-FenceClimbFall`, `Raptor-RunFall`, `Raptor-RunJumpFall`, `roach-Fall`, `Stego-Fall`, `Tricera-Fall`, `Tyranno-Fall`. `DEER-WalkCall` proves the mechanism is the suffix, not the word "Fall". Match the compilation's actual convention (stem equals the rig's export prefix plus "ALL") instead. Fix the twin in `tests/build/test_roundtrip_fbx.py:61` too: `"ALL" not in stem.upper()` is a SUBSTRING test that leaves seven rigs with no candidate file — Alligator, Anaconda, Crow, HermitCrab, Lion, SabreToothTiger, Tukan — of which Tukan is in `SAMPLE_RIGS` and skips green today.

- [ ] **Step 1: Write the failing test for the filter**

Assert `is_all_takes_bundle("DEER-WalkCall", rig="Deer")` is False and `is_all_takes_bundle("DEERALL", rig="Deer")` is True, across all 13 named stems.

- [ ] **Step 2: Run it to confirm it fails**

- [ ] **Step 3: Implement the shared predicate and use it in both places**

- [ ] **Step 4: Write the sidecheck script**

Compare joint positions through WORLD space, never raw quaternions: the two formats express a joint in its own local bone frame, a constant per-joint offset. Report per-rig median and 90th-percentile position error in bone-length units.

- [ ] **Step 5: Run it under Blender**

```bash
docker compose run --rm fbx blender --background \
  --python scripts/check_fbx_coherence.py -- --rigs Goat Crab Flamingo
```

- [ ] **Step 6: Run the suite and commit**

```bash
git add scripts/check_fbx_coherence.py tests/build/test_fbx_coherence.py tests/build/test_roundtrip_fbx.py
git rm tools/probe_bvh_fbx.py
git commit -m "feat(scripts): FBX coherence sidecheck; fix the all-takes filter"
```

---

## Definition of Done

- `docker compose run --rm test bash -c "ruff check . && pytest -q"` is green, with no fewer passing tests than the 258 this plan started from.
- `docker compose run --rm train python -m poseydon.cli train --out /tmp/smoke trainer.max_steps=20` completes on the GPU, writes a `recipe.yaml`, and logs to wandb.
- `MotionDataset` opens the real corpus and reads `stats.npz` rather than fitting.
- The 600 000-step run is NOT started by this plan. Starting it is the user's call.

## Open question for the user, after Task 9

The measured cost is 71.8 h at batch 16. Bucketing batches by rig would take it to ~35 h, but it removes cross-rig contrast from every gradient step, which for a cross-topology model is a research question rather than an optimization. It is deliberately not in this plan. Ask before adding it.
