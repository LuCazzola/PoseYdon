# Stage 1 Corrections and Corpus Rebuild — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct stage-1 preparation — authored rest poses, root promotion, and the round-trip test gaps — then rebuild the full 73-rig corpus.

**Architecture:** Two hardcoded Truebones tables enter `scripts/process_dataset_truebones.py`: which file supplies a rig's rest pose when the named T-pose is missing or unusable, and which joint to promote to root for the 14 rigs that root at a ground locator. Promotion is a new `PrepareStage` in `src/poseydon/build/prepare.py`, placed second in the chain. The round-trip test is extended through feature extraction and reconstruction, and the rest-pose selection rule — currently duplicated in three places, fixed in one — is consolidated onto a value recorded in `prepare.npz`.

**Tech Stack:** Python 3.12, NumPy, pytest, Hydra, Docker Compose. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-08-training-run-and-retarget-validation-design.md` — §1, §1.1, §1.2, §8 tests 1–7, §9.

This is plan **A1** of three. A2 covers the stage-2 build package; A3 covers the read path, training loop and GPU image. A1 ends with ~1145 prepared clips over 73 rigs and a round-trip test that covers a promoted rig.

## Global Constraints

- **Everything runs in Docker.** `docker compose run --rm test <cmd>`. Never invoke `pytest` or the preprocessing scripts on the host.
- **`ruff` must be clean** before every commit: `docker compose run --rm test ruff check .`
- The canonical mean bone length is `HML_MEAN_BONE_LENGTH = 0.20921428571428569` (`poseydon.core.skeleton`). Never inline the literal in production code.
- Quaternions are **scalar-last** `(x, y, z, w)`; `QUAT_IDENTITY = [0, 0, 0, 1]` from `poseydon.core.rotations`.
- Joint arrays are hierarchy-ordered: **a parent always precedes its children**. Several tasks rely on this.
- **Assertions must measure, not restate.** Phase 1's retrospective records that four of eleven tasks shipped an assertion that reproduced the implementation's own definition; a rig-dependent scale bug survived all eleven because the guard restated the stage. Every assertion below compares against an independently-derived quantity.
- A test that skips is a test that passes. Any new `pytest.skip` must be justified by absent data or absent Blender, never by a rig the code cannot handle.

---

### Task 1: Fix the container user, and add `.env`

`docker-compose.yml` sets `user: "${UID:-1000}:${GID:-1000}"`. Bash does not export `UID`, so the `1000` fallback always wins. On a machine whose developer is uid 1001, every file a container writes into the bind mount is owned by the wrong user and cannot be removed from the host — the exact failure the file's own comment warns about. This must land first: Task 9 writes ~1145 files into the repo.

**Files:**
- Create: `.env.example`
- Modify: `.gitignore`
- Modify: `docker-compose.yml:5-8` (the `test` service comment) — no change to the substitution itself, which becomes correct once `.env` exists

**Interfaces:**
- Consumes: nothing.
- Produces: a `.env` file that Compose loads automatically for variable substitution in every service. A3 adds `WANDB_API_KEY` and `WANDB_ENTITY` to the same file.

- [ ] **Step 1: Reproduce the bug**

```bash
docker compose run --rm test sh -c 'touch /app/.uid_probe' && ls -ln .uid_probe
```

Expected: the file is owned by uid `1000`, while `id -u` on the host reports something else (1001 on this machine). If they already match, this task is a no-op on your machine but still needed for portability — continue anyway.

- [ ] **Step 2: Create `.env.example`**

```bash
cat > .env.example <<'EOF'
# Copy to .env and fill in. Compose loads .env automatically.
#
# The container runs as this user so files it writes into the bind mount
# belong to you. docker-compose.yml reads ${UID}/${GID}; bash does not export
# UID, so without this file the 1000 fallback silently wins.
#   printf 'UID=%s\nGID=%s\n' "$(id -u)" "$(id -g)" >> .env
UID=1000
GID=1000
EOF
```

- [ ] **Step 3: Generate your own `.env`**

```bash
printf 'UID=%s\nGID=%s\n' "$(id -u)" "$(id -g)" > .env
cat .env
```

- [ ] **Step 4: Ignore `.env`, keep `.env.example`**

Add to `.gitignore`, near the other local-only entries:

```
# Local container user and API keys. .env.example is committed; .env is not.
.env
```

- [ ] **Step 5: Verify the fix**

```bash
rm -f .uid_probe 2>/dev/null || docker compose run --rm --user root test rm -f /app/.uid_probe
docker compose run --rm test sh -c 'touch /app/.uid_probe' && ls -ln .uid_probe
```

Expected: owned by your uid. Then remove it: `rm .uid_probe` — which now succeeds from the host without root.

- [ ] **Step 6: Confirm the suite still runs**

```bash
docker compose run --rm test pytest -q
```

Expected: same result as before this task (7 passed / 1 skipped in `test_roundtrip.py`, whole suite green apart from the recorded xfails).

- [ ] **Step 7: Commit**

```bash
git add .env.example .gitignore
git commit -m "fix(docker): load UID/GID from .env so containers run as the developer

docker-compose.yml has always read \${UID:-1000}, but bash does not export
UID, so the fallback won on every machine. Files a container wrote into the
bind mount were owned by uid 1000 and needed root to delete -- the exact
failure the service comment warns about."
```

---

### Task 2: Record the rest clip, and consolidate rest selection

The rule that picks a rig's rest-pose file exists in **three** copies, and only one has the `b2b0151` modal-set fix:

| location | rule | consequence |
|---|---|---|
| `scripts/process_dataset_truebones.py::find_tpose` | modal set, then name | correct |
| `tests/build/test_roundtrip.py::_rest_path` | name only | **Crab skips today** |
| `tests/build/test_bvh_fbx_agreement.py::_rest_path` | name only | will skip once Crab's rest clip is `walk.bvh` |

Duplicating a decision is what made this possible. The fix records the choice where every consumer can read it: `prepare.npz` gains a `rig/rest/clip` entry naming the action slug stage 1 actually fitted. Spec §2 requires this independently — stage 2 must read rig-level offsets from the rest clip, and after Task 3 that file is frequently not named `tpose`.

**Files:**
- Modify: `scripts/process_dataset_truebones.py` — `process_species`, to record the chosen rest action
- Modify: `tests/build/test_roundtrip.py:60-68` — `_rest_path` reads the recording
- Modify: `tests/build/test_bvh_fbx_agreement.py:33-48` — same
- Test: `tests/build/test_rest_selection.py`

**Interfaces:**
- Consumes: `find_tpose(clip_paths: list[Path]) -> tuple[Path | None, list[str]]`, `RigTransform.rig_params`.
- Produces: `prepare.npz` carries `rig/rest/clip` — a 0-d object array holding the action slug (e.g. `"walk"` for Crab). Read with `RigTransform.load(path).rig_params["rest"]["clip"]`. A2 reads the same key.

- [ ] **Step 1: Write the failing test**

Append to `tests/build/test_rest_selection.py`:

```python
def test_the_chosen_rest_clip_is_recorded_in_prepare_npz(tmp_path):
    """Stage 1's rest choice must be readable downstream, not re-derived.

    Three consumers picked the rest file independently and two of them had a
    stale copy of the rule. The choice is data, so it is recorded.
    """
    from poseydon.build.prepare import RigTransform

    transform = RigTransform(
        rig_params={"rest": {"clip": np.array("walk", dtype=object)}},
        clip_params={"walk": {}},
    )
    path = tmp_path / "prepare.npz"
    transform.save(path)

    assert str(RigTransform.load(path).rig_params["rest"]["clip"]) == "walk"
```

Add `import numpy as np` to that file's imports.

- [ ] **Step 2: Run it to see it fail**

Run: `docker compose run --rm test pytest tests/build/test_rest_selection.py -v`
Expected: FAIL — `RigTransform.save` writes object arrays but `load` may not round-trip a 0-d one. If it passes unchanged, the storage already works; proceed to Step 3, which is the real change.

- [ ] **Step 3: Record the choice in `process_species`**

In `scripts/process_dataset_truebones.py::process_species`, after `rig_params = chain.fit_rig(...)` (around line 152), add:

```python
    # The rest choice is data, not a rule to be re-derived. Three consumers
    # picked this file independently and two carried a stale copy of the rule,
    # so the round trip silently skipped Crab. Stage 2 needs it too: rig-level
    # offsets must come from the rest clip, and after the authored table that
    # file is frequently not named `tpose`.
    rest_action = strip_skeleton_prefix(action_slug(rest_path.stem), manifest.name)
    rig_params["rest"] = {"clip": np.array(rest_action, dtype=object)}
```

- [ ] **Step 4: Point the two test copies at the recording**

In `tests/build/test_roundtrip.py`, replace `_rest_path` entirely:

```python
def _rest_path(clips):
    """The rig's rest-pose file, by the PRODUCTION rule.

    This used to carry its own copy that matched on filename only -- the rule
    b2b0151 fixed in the script and not here. Crab therefore resolved to its
    54-joint __TPOSE.bvh, every 64-joint clip disagreed, and the case skipped.
    Import the real thing rather than restate it.
    """
    chosen, _warnings = find_tpose(list(clips))
    if chosen is None:
        pytest.skip("no rest-pose file for this rig")
    return chosen
```

Add to that file's imports:

```python
from scripts.process_dataset_truebones import find_tpose
```

In `tests/build/test_bvh_fbx_agreement.py`, replace its `_rest_path` body — this one selects among *prepared* clips, so it reads the recorded action rather than re-running `find_tpose` on raw files:

```python
def _rest_path(bvhs):
    """The rig's prepared rest clip, read from what stage 1 recorded.

    `FaceAxis` is clip-scoped and `rotate_rig` turns OFFSETS with the motion,
    so every prepared clip carries a differently-rotated offset block --
    only the rest clip's offsets correspond to mesh.npz's bind pose. Which
    clip that is, is recorded; matching on the filename was a stale copy of
    a rule that no longer holds (Crab's rest clip is `walk`).
    """
    rig = bvhs[0].parent.name
    prepare = CORPUS / "rigs" / rig / "prepare.npz"
    if not prepare.is_file():
        pytest.skip(f"{rig}: no prepare.npz -- run stage 1")
    action = str(RigTransform.load(prepare).rig_params["rest"]["clip"])
    path = bvhs[0].parent / f"{action}.bvh"
    if not path.is_file():
        pytest.skip(f"{rig}: recorded rest clip {action}.bvh is not on disk")
    return path
```

Add to its imports:

```python
from poseydon.build.prepare import RigTransform
```

- [ ] **Step 5: Run the tests**

Run: `docker compose run --rm test pytest tests/build/ -v -rs`

Expected: `test_rest_selection.py` passes. `test_roundtrip.py::test_round_trip_returns_the_source_rig[Crab]` still **skips** — but now with a different reason: `find_tpose` picks Crab's `__Attack1.bvh` (its modal-set rest file), the clip loop then picks the first non-rest clip, and that clip agrees, so it should now **pass**. If it still skips, read the skip reason before proceeding; the `_rest_path(clips)` call inside the test body is invoked twice and must return the same path both times.

`test_bvh_fbx_agreement.py` cases will skip with "no prepare.npz" or "recorded rest clip not on disk" until Task 9 rebuilds — that is correct, and Task 9 verifies they stop skipping.

- [ ] **Step 6: Verify Crab actually round-trips now**

Run: `docker compose run --rm test pytest "tests/build/test_roundtrip.py::test_round_trip_returns_the_source_rig[Crab]" -v -rs`
Expected: PASSED, not SKIPPED. This is the whole point of the task.

- [ ] **Step 7: `ruff` and commit**

```bash
docker compose run --rm test ruff check .
git add scripts/process_dataset_truebones.py tests/build/test_roundtrip.py tests/build/test_bvh_fbx_agreement.py tests/build/test_rest_selection.py
git commit -m "fix(test): read the recorded rest clip instead of re-deriving it

The rest-selection rule existed in three copies and only the script had the
modal-set fix from b2b0151. test_roundtrip._rest_path matched on filename, so
Crab resolved to its 54-joint __TPOSE.bvh, every 64-joint clip disagreed, and
the round trip -- the test the application story rests on -- silently skipped
for the milliped SAMPLE_RIGS exists to cover.

prepare.npz now records the action slug stage 1 fitted, under rig/rest/clip,
and every consumer reads it. Stage 2 needs the same value: rig-level offsets
must come from the rest clip, which after the authored table is frequently
not named tpose."
```

---

### Task 3: Author the rest-pose file for the 17 rigs that need one

Spec §1.1. `find_tpose` resolves T-pose → idle → walk (fly for flying creatures). The first is discovered; the last two are authored, because matching `idle` as a substring picks Lion's `__DeathIdle.bvh`, Jaguar's `__LieIdle.bvh` and Trex's `__idle_attack.bvh` — a dying pose, a lying pose and a crouched attack.

**Files:**
- Modify: `scripts/process_dataset_truebones.py` — add `REST_POSE_FILE`, consult it inside `find_tpose`
- Test: `tests/build/test_rest_selection.py`

**Interfaces:**
- Consumes: `find_tpose(clip_paths) -> tuple[Path | None, list[str]]`, `BVH.read_names(path) -> tuple[str, ...]`.
- Produces: `REST_POSE_FILE: dict[str, str]` mapping rig name to raw filename. Task 5's promotion table sits beside it.

- [ ] **Step 1: Write the failing test**

Append to `tests/build/test_rest_selection.py`:

```python
def test_every_authored_rest_file_exists_and_carries_the_modal_skeleton():
    """The authored table is a judgement about content; this pins what it can.

    Trex is why the modal-set guard stays outside the table: its natural
    neutral pick, __STILL.bvh, IS that rig's outlier file (66 joints against
    the modal 78), so a table trusted on its own would have silently destroyed
    the largest rig in the corpus.
    """
    from collections import Counter

    from scripts.process_dataset_truebones import REST_POSE_FILE

    source = CORPUS / "source"
    if not source.is_dir():
        pytest.skip("Truebones corpus not present")

    for rig, filename in sorted(REST_POSE_FILE.items()):
        rig_dir = source / rig
        assert rig_dir.is_dir(), f"{rig}: no source directory"
        path = rig_dir / filename
        assert path.is_file(), f"{rig}: authored rest file {filename} does not exist"

        names_by_path = {p: BVH.read_names(p) for p in sorted(rig_dir.glob("*.bvh"))}
        modal, _count = Counter(names_by_path.values()).most_common(1)[0]
        assert names_by_path[path] == modal, (
            f"{rig}: authored rest file {filename} has {len(names_by_path[path])} "
            f"joints but the rig's modal skeleton has {len(modal)}"
        )
```

Add to that file's imports:

```python
import pytest

from poseydon.io.bvh import BVH
from tests.conftest import CORPUS
```

- [ ] **Step 2: Run it to verify it fails**

Run: `docker compose run --rm test pytest tests/build/test_rest_selection.py::test_every_authored_rest_file_exists_and_carries_the_modal_skeleton -v`
Expected: FAIL with `ImportError: cannot import name 'REST_POSE_FILE'`.

- [ ] **Step 3: Add the table**

In `scripts/process_dataset_truebones.py`, after the imports and before `_pick_by_name`:

```python
#: Rest-pose file for rigs whose named T-pose is missing or unusable.
#:
#: Preference is T-pose, then idle, then a walk (or a fly, for a flying
#: creature). The last two are authored rather than matched, because matching
#: "idle" as a substring picks Lion's __DeathIdle.bvh, Jaguar's __LieIdle.bvh
#: and Trex's __idle_attack.bvh -- a dying pose, a lying pose and a crouched
#: attack, which are the poses this rule exists to avoid. The previous
#: fallback was worse still: "first file in the modal set" is what gave Crab a
#: rest geometry fitted from __Attack1.bvh.
#:
#: An entry here is a JUDGEMENT about animation content and cannot be verified
#: by test. What IS verified is that the file exists and carries the rig's
#: modal skeleton -- see test_rest_selection.py. Trex is why that guard is not
#: ceremony: its natural pick, __STILL.bvh, is the rig's outlier file.
REST_POSE_FILE: dict[str, str] = {
    # No T-pose file at all.
    "Anaconda": "__Idle.bvh",
    "Bird": "__IdleLoop.bvh",
    "Camel": "__IdleLoop.bvh",
    "Cricket": "__Idle.bvh",
    "Dog": "__Idle.bvh",
    "Goat": "__Idle.bvh",
    "Lion": "__SlowIdle.bvh",          # __DeathIdle.bvh is the trap
    "Monkey": "__Idle1.bvh",
    "Pteranodon": "__FlyLoop.bvh",     # no idle clip; flying creature
    "Rat": "__Trottle.bvh",            # no idle clip; trot is its nearest gait
    "SabreToothTiger": "__Startwalk.bvh",  # no idle among 44; frame 0 stands
    "Scorpion-2": "__Idle.bvh",
    "Trex": "__walk_loop.bvh",         # every idle_* clip is idle-plus-action
    # T-pose file exists but declares a minority skeleton.
    "Ant": "__Idle.bvh",
    "Crab": "__Walk.bvh",              # no idle clip
    "Deer": "__Idle.bvh",
    "Jaguar": "__Idle.bvh",            # __LieIdle.bvh is the trap
}
```

- [ ] **Step 4: Consult the table inside `find_tpose`**

`find_tpose` currently takes only `clip_paths`. Give it the rig name so it can look the rig up, keeping the modal-set guard *outside* the table. Replace the body between `modal_paths = ...` and `old_pick = ...`:

```python
    names_by_path = {path: BVH.read_names(path) for path in clip_paths}
    counts = Counter(names_by_path.values())
    modal_names, _ = counts.most_common(1)[0]
    modal_paths = [path for path in clip_paths if names_by_path[path] == modal_names]

    # An authored choice wins over the naming rule, but never over the modal
    # set: the table is a judgement about pose content, and the modal check is
    # what stops a plausible-looking pick from destroying a rig. Trex's
    # __STILL.bvh is exactly that case.
    authored = REST_POSE_FILE.get(rig)
    if authored is not None:
        matches = [path for path in modal_paths if path.name == authored]
        if matches:
            return matches[0], warnings
        warnings.append(
            f"authored rest file {authored} is absent or not in the modal joint "
            f"set ({len(modal_paths)}/{len(clip_paths)} clips); falling back to "
            "the naming rule"
        )

    old_pick = _pick_by_name(clip_paths)
    chosen = _pick_by_name(modal_paths)
```

Change the signature and the docstring's `Returns` line:

```python
def find_tpose(clip_paths: list[Path], rig: str = "") -> tuple[Path | None, list[str]]:
```

Update the one production call site in `process_species`:

```python
    rest_path, tpose_warnings = find_tpose(clip_paths, manifest.name)
```

The `rig=""` default keeps `test_rest_selection.py`'s existing synthetic test and `test_roundtrip.py::_rest_path` working unchanged — neither names a real rig.

- [ ] **Step 5: Run the tests**

Run: `docker compose run --rm test pytest tests/build/test_rest_selection.py -v`
Expected: all pass, including the new table check across all 17 rigs.

- [ ] **Step 6: Confirm the four recovered rigs still recover**

```bash
docker compose run --rm test python scripts/process_dataset_truebones.py \
    --out-root /tmp/probe_out --rigs Crab Ant Deer Jaguar 2>&1 | tail -20
```

This needs manifests under `/tmp/probe_out/rigs/`; create them first:

```bash
docker compose run --rm test sh -c '
  for r in Crab Ant Deer Jaguar; do
    mkdir -p /tmp/probe_out/rigs/$r
    cp data/truebones/rigs/$r/manifest.yaml /tmp/probe_out/rigs/$r/
  done
  cp data/truebones/rigs/_base.yaml /tmp/probe_out/rigs/ 2>/dev/null || true
  python scripts/process_dataset_truebones.py --out-root /tmp/probe_out \
      --rigs Crab Ant Deer Jaguar' 2>&1 | tail -20
```

Expected: `Crab 10 · Ant 17 · Deer 20 · Jaguar 13` clips written, and the rest file now reported as `__Walk.bvh` for Crab and `__Idle.bvh` for the other three — not `__Attack1.bvh`.

- [ ] **Step 7: `ruff` and commit**

```bash
docker compose run --rm test ruff check .
git add scripts/process_dataset_truebones.py tests/build/test_rest_selection.py
git commit -m "feat(prepare): author the rest-pose file for the 17 rigs needing one

find_tpose resolves T-pose, then idle, then walk -- or fly, for a flying
creature. The last two are an authored table rather than a name match, because
matching 'idle' as a substring picks Lion's __DeathIdle, Jaguar's __LieIdle and
Trex's __idle_attack: a dying pose, a lying pose and a crouched attack.

The modal-set check stays outside the table and earns its keep -- Trex's
natural neutral pick, __STILL.bvh, is that rig's outlier file (66 joints
against the modal 78)."
```

---

### Task 4: The `PromoteRoot` stage

Spec §1.2. 14 rigs root the skeleton at a ground locator; `ric_pos` is root-relative, so they hand the model every joint with a constant vertical bias while the other 59 do not. This task builds the stage against synthetic rigs; Task 5 wires the table.

**Key structural fact:** the joints above the first branching joint form a single-child chain, and hierarchy order is depth-first, so they occupy indices `0 … b-1` **contiguously** and every other joint is a descendant of `b`. Promotion is therefore a slice, not a reindex.

**Files:**
- Modify: `src/poseydon/build/prepare.py` — add `PromoteRoot` after `RestRelative`
- Test: `tests/build/test_prepare.py`

**Interfaces:**
- Consumes: `PrepareStage`, `RIG`, `Animation`, `quat_mul`, `QUAT_IDENTITY`.
- Produces:
  ```python
  @dataclass(frozen=True)
  class PromoteRoot(PrepareStage):
      target: str | None = None
      name: ClassVar[str] = "promote_root"
      scope: ClassVar[str] = RIG
  ```
  Fitted params: `{"n_removed": np.int64, "names": (b+1,) object, "offsets": (b+1, 3) float64}` — the removed chain plus the promoted joint itself, root-first. `target=None` makes every method the identity and fits `n_removed = 0`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/build/test_prepare.py`. It already has a synthetic-rig helper; if the existing one does not produce a chain, add this one beside it:

```python
def _locator_rig(n_frames: int = 4) -> Animation:
    """Hips -> Cog -> Pelvis -> {LeftLeg, RightLeg}: a two-step locator chain.

    Hips sits at the origin and Pelvis a real distance above it, which is the
    shape of Camel, Horse and Trex. Cog and Hips both carry a non-identity
    rotation, so a promotion that merely dropped them would move the body.
    """
    names = ("Hips", "Cog", "Pelvis", "LeftLeg", "RightLeg")
    parents = np.array([-1, 0, 1, 2, 2], dtype=np.int32)
    offsets = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 5.0, 0.0],
         [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0]]
    )
    rng = np.random.default_rng(0)
    rotations = rng.normal(size=(n_frames, len(names), 4))
    rotations /= np.linalg.norm(rotations, axis=-1, keepdims=True)

    translations = np.broadcast_to(offsets, (n_frames, len(names), 3)).copy()
    translations[:, 0] = rng.normal(size=(n_frames, 3))
    return Animation(
        rotations=rotations, translations=translations, offsets=offsets,
        parents=parents, names=names, fps=30.0,
    )


def test_promote_root_moves_no_surviving_joint():
    """The promoted joint and its descendants keep their world positions.

    This is the assertion that matters. Checking the joint COUNT would restate
    the table; checking world positions measures the transform.
    """
    anim = _locator_rig()
    stage = PromoteRoot(target="Pelvis")
    params = stage.fit(anim, None)
    promoted = stage.apply(anim, params)

    before = anim.global_positions()
    after = promoted.global_positions()
    for name in ("Pelvis", "LeftLeg", "RightLeg"):
        np.testing.assert_allclose(
            after[:, promoted.names.index(name)],
            before[:, anim.names.index(name)],
            atol=1e-9,
            err_msg=f"{name} moved",
        )


def test_promote_root_makes_the_target_the_root():
    anim = _locator_rig()
    stage = PromoteRoot(target="Pelvis")
    promoted = stage.apply(anim, stage.fit(anim, None))

    assert promoted.names == ("Pelvis", "LeftLeg", "RightLeg")
    assert list(promoted.parents) == [-1, 0, 0]


def test_promote_root_restores_the_source_structure():
    """invert returns the hierarchy the user supplied -- spec §9's contract."""
    anim = _locator_rig()
    stage = PromoteRoot(target="Pelvis")
    params = stage.fit(anim, None)
    restored = stage.invert(stage.apply(anim, params), params)

    assert restored.names == anim.names
    assert list(restored.parents) == list(anim.parents)
    np.testing.assert_allclose(restored.offsets, anim.offsets, atol=1e-12)


def test_promote_root_round_trip_keeps_the_body_in_place():
    """Structure returns exactly; the body returns exactly; the locators do not.

    The chain's rotations all turn the same subtree, so only their product is
    observable and invert puts that product on the promoted joint with identity
    above it. Pelvis and its descendants therefore land exactly where they
    were. Hips and Cog do not, and asserting they would is asserting something
    the design explicitly does not promise (spec §9).
    """
    anim = _locator_rig()
    stage = PromoteRoot(target="Pelvis")
    params = stage.fit(anim, None)
    restored = stage.invert(stage.apply(anim, params), params)

    before = anim.global_positions()
    after = restored.global_positions()
    for name in ("Pelvis", "LeftLeg", "RightLeg"):
        index = anim.names.index(name)
        np.testing.assert_allclose(after[:, index], before[:, index], atol=1e-9)


def test_promote_root_with_no_target_is_the_identity():
    """59 of 73 rigs are not in the table and must be untouched."""
    anim = _locator_rig()
    stage = PromoteRoot(target=None)
    params = stage.fit(anim, None)

    for produced in (stage.apply(anim, params), stage.invert(anim, params)):
        assert produced.names == anim.names
        np.testing.assert_allclose(
            produced.global_positions(), anim.global_positions(), atol=1e-12
        )


def test_promote_root_refuses_a_target_that_is_not_on_the_root_chain():
    """LeftLeg has a sibling, so the joints above it are not a single-child
    chain and slicing them away would delete RightLeg's subtree."""
    anim = _locator_rig()
    stage = PromoteRoot(target="LeftLeg")
    with pytest.raises(ValueError, match="single-child chain"):
        stage.fit(anim, None)


def test_promote_root_refuses_an_unknown_target():
    anim = _locator_rig()
    stage = PromoteRoot(target="NoSuchJoint")
    with pytest.raises(ValueError, match="NoSuchJoint"):
        stage.fit(anim, None)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `docker compose run --rm test pytest tests/build/test_prepare.py -k promote -v`
Expected: FAIL — `NameError: name 'PromoteRoot' is not defined`.

- [ ] **Step 3: Implement the stage**

In `src/poseydon/build/prepare.py`, after the `RestRelative` class:

```python
@dataclass(frozen=True)
class PromoteRoot(PrepareStage):
    """Move the root from a ground locator onto the first real body joint.

    14 of 73 Truebones rigs root the skeleton at a locator sitting at y=0 while
    the body hangs 0.7-6.5 bone lengths above it. `ric_pos` is root-relative, so
    those rigs hand the model every joint with a constant vertical bias, and
    their root-trajectory block is a ground projection where the other 59 give a
    body trajectory -- two conventions for one feature across 19% of the corpus.

    ONLY the root can absorb an offset, which is what makes this principled
    rather than an exception: `EnforceRigid` drops non-root translation, so no
    other joint has a channel to put one in. That is exactly why the reduction
    rule's collapse case requires a zero offset.

    The joints above `target` form a single-child chain and hierarchy order is
    depth-first, so they occupy indices 0..b-1 contiguously and every other
    joint descends from `target`. Promotion is a slice.

    `invert` re-inserts the chain with IDENTITY rotations and the composed
    product left on the promoted joint. Every joint returns with its name,
    parent and offset, and `target` and its descendants return to their exact
    world positions; the locator joints above it do not, because the chain's
    rotations all turn the same subtree and only their product was ever
    observable. Spec §9 states this as the contract: structure and reference
    frame, not motion values.
    """

    #: Joint to promote. ``None`` makes every method the identity, which is what
    #: the 59 rigs not in the table get.
    target: str | None = None

    name: ClassVar[str] = "promote_root"
    scope: ClassVar[str] = RIG

    def fit(self, anim: Animation, resolved: Any) -> dict[str, np.ndarray]:
        if self.target is None:
            return {
                "n_removed": np.int64(0),
                "names": np.empty(0, dtype=object),
                "offsets": np.zeros((0, 3)),
            }

        names = list(anim.names)
        if self.target not in names:
            raise ValueError(
                f"cannot promote `{self.target}`: this rig has no such joint "
                f"(it has {len(names)}, starting {names[:3]})"
            )
        index = names.index(self.target)

        # Every joint above the target must have exactly one child, or the
        # joints being sliced away are not a chain and the slice would delete
        # a sibling subtree.
        for joint in range(index):
            children = [k for k, p in enumerate(anim.parents) if p == joint]
            if len(children) != 1:
                raise ValueError(
                    f"cannot promote `{self.target}`: joint `{names[joint]}` above "
                    f"it has {len(children)} children, so the joints above the "
                    "target are not a single-child chain"
                )

        chain = np.empty(index + 1, dtype=object)
        chain[:] = names[: index + 1]
        return {
            "n_removed": np.int64(index),
            "names": chain,
            "offsets": anim.offsets[: index + 1].copy(),
        }

    def apply(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        index = int(params["n_removed"])
        if index == 0:
            return anim

        positions, rotations_world = anim.global_transforms()

        translations = anim.translations[:, index:].copy()
        # The new root's translation slot holds a GLOBAL position (Animation's
        # contract for joint 0), which is exactly where the target already is.
        translations[:, 0] = positions[:, index]

        rotations = anim.rotations[:, index:].copy()
        # Its rotation absorbs the whole chain: with no ancestors left, its
        # local rotation must equal what its world rotation was.
        rotations[:, 0] = rotations_world[:, index]

        parents = anim.parents[index:] - index
        parents = parents.astype(np.int32, copy=True)
        parents[0] = -1

        return Animation(
            rotations=rotations,
            translations=translations,
            offsets=anim.offsets[index:].copy(),
            parents=parents,
            names=tuple(anim.names[index:]),
            fps=anim.fps,
        )

    def invert(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        index = int(params["n_removed"])
        if index == 0:
            return anim

        chain_names = [str(n) for n in params["names"]]
        chain_offsets = params["offsets"]
        n_frames = anim.n_frames
        n_joints = anim.n_joints + index

        rotations = np.empty((n_frames, n_joints, 4))
        rotations[:, :index] = QUAT_IDENTITY
        rotations[:, index:] = anim.rotations

        translations = np.empty((n_frames, n_joints, 3))
        # Chain joints below the root sit at their own rest offsets; with
        # identity rotations throughout, the old root's global position is the
        # target's position minus the offsets accumulated down the chain.
        translations[:, 1:index] = chain_offsets[1:index]
        translations[:, 0] = anim.translations[:, 0] - chain_offsets[1 : index + 1].sum(
            axis=0
        )
        translations[:, index:] = anim.translations
        translations[:, index] = chain_offsets[index]

        offsets = np.concatenate([chain_offsets, anim.offsets[1:]], axis=0)

        parents = np.empty(n_joints, dtype=np.int32)
        parents[0] = -1
        parents[1 : index + 1] = np.arange(index, dtype=np.int32)
        parents[index + 1 :] = anim.parents[1:] + index

        return Animation(
            rotations=rotations,
            translations=translations,
            offsets=offsets,
            parents=parents,
            names=tuple(chain_names[:index]) + tuple(anim.names),
            fps=anim.fps,
        )
```

- [ ] **Step 4: Run the tests**

Run: `docker compose run --rm test pytest tests/build/test_prepare.py -k promote -v`
Expected: all seven PASS.

- [ ] **Step 5: Run the whole prepare suite for regressions**

Run: `docker compose run --rm test pytest tests/build/ -v`
Expected: no new failures; `test_roundtrip.py` unchanged from Task 2.

- [ ] **Step 6: `ruff` and commit**

```bash
docker compose run --rm test ruff check .
git add src/poseydon/build/prepare.py tests/build/test_prepare.py
git commit -m "feat(prepare): add PromoteRoot, moving the root off a ground locator

14 of 73 rigs root the skeleton at a locator at y=0 with the body hanging
0.7-6.5 bone lengths above. ric_pos is root-relative, so those rigs present
every joint with a constant vertical bias and their root-trajectory block is a
ground projection where the other 59 give a body trajectory.

Only the root can absorb an offset -- EnforceRigid drops non-root translation,
which is why the reduction rule's collapse case requires a zero one. The joints
above the target form a single-child chain and hierarchy order is depth-first,
so promotion is a slice; a target whose ancestors branch is refused."
```

---

### Task 5: Wire the promotion table into stage 1

Spec §1.2's table, and the contractual ordering: `PromoteRoot` is stage 2, after `RestRelative` (whose `_check` a promoted clip would fail) and before `EnforceRigid` (so the translation channel `EnforceRigid` preserves is the promoted root's, not the locator's) and before `ScaleToMeanBoneLength` (the removed connectors are long enough to move the canonical scale).

**Files:**
- Modify: `scripts/process_dataset_truebones.py` — add `PROMOTE_ROOT`, insert the stage into `_chain_for`
- Test: `tests/build/test_promote_table.py` (create)

**Interfaces:**
- Consumes: `PromoteRoot(target=...)` from Task 4, `REST_POSE_FILE` from Task 3.
- Produces: `PROMOTE_ROOT: dict[str, str]` mapping rig name to the joint promoted to root. `_chain_for(source_channels, rig)` gains a `rig` parameter.

- [ ] **Step 1: Write the failing test**

Create `tests/build/test_promote_table.py`:

```python
"""§1.2's promotion table: every entry must be reachable and must fix the rig.

The gate is the ground-locator classification, never the presence of an offset.
Lynx and BrownBear have a correct root on the pelvis and a chain continuing to
Bip01_Spine at a real offset (0.79 and 0.89 bone lengths); a rule keyed on "the
chain carries an offset" would promote their root onto the spine and break two
healthy rigs. They are asserted absent from the table explicitly.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from poseydon.io.bvh import BVH
from scripts.process_dataset_truebones import PROMOTE_ROOT, REST_POSE_FILE
from tests.conftest import CORPUS

LOCATOR_BAND = 0.05


def _rest_anim(rig: str):
    rig_dir = CORPUS / "source" / rig
    if not rig_dir.is_dir():
        pytest.skip(f"{rig}: no source directory")
    bvhs = sorted(rig_dir.glob("*.bvh"))
    named = REST_POSE_FILE.get(rig)
    if named is not None:
        path = rig_dir / named
    else:
        by_path = {p: BVH.read_names(p) for p in bvhs}
        modal, _ = Counter(by_path.values()).most_common(1)[0]
        modal_paths = [p for p in bvhs if by_path[p] == modal]
        path = next((p for p in modal_paths if "tpos" in p.name.lower()), modal_paths[0])
    return BVH.read(path).to_animation()


def _height_fraction(anim, joint: int) -> float:
    """Where a joint sits between the skeleton's lowest and highest point."""
    y = anim.global_positions()[0][:, 1]
    lo, hi = float(y.min()), float(y.max())
    return 0.0 if hi <= lo else (float(y[joint]) - lo) / (hi - lo)


@pytest.mark.parametrize("rig", sorted(PROMOTE_ROOT))
def test_the_promoted_joint_exists_and_leaves_the_locator_band(rig):
    anim = _rest_anim(rig)
    target = PROMOTE_ROOT[rig]
    assert target in anim.names, f"{rig}: no joint named {target}"

    assert _height_fraction(anim, 0) < LOCATOR_BAND, (
        f"{rig} is in the table but its root is not a ground locator"
    )
    promoted = _height_fraction(anim, anim.names.index(target))
    assert promoted > LOCATOR_BAND, (
        f"{rig}: promoting to {target} leaves the root at height fraction "
        f"{promoted:.3f}, still inside the locator band"
    )


@pytest.mark.parametrize("rig", ["Lynx", "BrownBear", "Tyranno", "PolarBear"])
def test_healthy_rigs_are_absent_from_the_table(rig):
    """These have a correct pelvis root; some also have a chain with a real
    offset, which is precisely why the offset is not the gate."""
    assert rig not in PROMOTE_ROOT
    anim = _rest_anim(rig)
    assert _height_fraction(anim, 0) > LOCATOR_BAND


def test_promotion_moves_no_surviving_joint_on_a_real_rig():
    """World positions, on real data -- not a joint count, which would restate
    the table."""
    from poseydon.build.prepare import PromoteRoot

    rig = "Camel"
    anim = _rest_anim(rig)
    stage = PromoteRoot(target=PROMOTE_ROOT[rig])
    params = stage.fit(anim, None)
    promoted = stage.apply(anim, params)

    before = anim.global_positions()
    after = promoted.global_positions()
    for index, name in enumerate(promoted.names):
        np.testing.assert_allclose(
            after[:, index], before[:, anim.names.index(name)], atol=1e-7,
            err_msg=f"{rig}/{name} moved",
        )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `docker compose run --rm test pytest tests/build/test_promote_table.py -v`
Expected: FAIL with `ImportError: cannot import name 'PROMOTE_ROOT'`.

- [ ] **Step 3: Add the table**

In `scripts/process_dataset_truebones.py`, immediately after `REST_POSE_FILE`:

```python
#: Joint promoted to root, for the 14 rigs that root at a ground locator.
#:
#: Measured on the rest pose as height fraction (rootY - minY) / (maxY - minY),
#: the corpus splits absolutely: these 14 at f <= 0.005, the other 59 at
#: f >= 0.189, nothing between. Thirteen entries are the first branching joint
#: along the root's single-child chain. Tukan is authored because its root
#: branches straight into the real skeleton (N_ALL -> locator) and a dead MESH
#: subtree of geometry-holder nodes, so no chain rule reaches it.
#:
#: The gate is the locator classification, NEVER the presence of an offset:
#: Lynx and BrownBear have a correct root on the pelvis and a chain continuing
#: to Bip01_Spine at 0.79 and 0.89 bone lengths, and an offset-keyed rule would
#: promote their root onto the spine. A rig absent from this table is untouched.
PROMOTE_ROOT: dict[str, str] = {
    "Bear": "NPC_Pelvis",
    "Camel": "Bip01",
    "Crow": "_00",
    "Dog": "Bip01_Pelvis",
    "Dog-2": "Bip01_Pelvis",
    "Horse": "Bip01_Pelvis",
    "Pirrana": "locator",
    "Pteranodon": "jt_Cog_C",
    "Raptor3": "jt_Cog_C",
    "SabreToothTiger": "Sabrecat__pelv_",
    "Scorpion-2": "jt_Cog_C",
    "Spider": "_body_",
    "Trex": "jt_Cog_C",
    "Tukan": "locator",
}
```

- [ ] **Step 4: Insert the stage into the chain**

Change `_chain_for` to take the rig name and place `PromoteRoot` second:

```python
def _chain_for(source_channels, rig: str = "") -> PrepareChain:
    return PrepareChain(
        (
            RestRelative(),
            # Second, and the position is contractual. After RestRelative,
            # whose _check requires the clip to carry exactly the joints its
            # rest pose declares -- a promoted clip would fail it. Before
            # EnforceRigid, so the translation channel EnforceRigid preserves
            # is the PROMOTED root's rather than the locator's. And before
            # ScaleToMeanBoneLength: the connectors this removes are long
            # (6.47 bone lengths on Pirrana, 4.95 on Bear) and currently enter
            # the mean, so promoting first changes the canonical scale for
            # these 14 rigs -- a correction, since a locator-to-body connector
            # is not an anatomical bone.
            PromoteRoot(target=PROMOTE_ROOT.get(rig)),
            FaceAxis(axis=TARGET_AXIS),
            EnforceRigid(joint_translation="drop", source_channels=source_channels),
            CentreXZ(),
            ScaleToMeanBoneLength(),
            PutOnGround(),
        )
    )
```

Add `PromoteRoot` to the `from poseydon.build.prepare import (...)` block, and update the call in `process_species`:

```python
    chain = _chain_for(rest_bvh.channels, manifest.name)
```

- [ ] **Step 5: Run the tests**

Run: `docker compose run --rm test pytest tests/build/test_promote_table.py -v`
Expected: all pass — 14 parametrized promotion cases, 4 healthy-rig cases, and the Camel world-position check.

- [ ] **Step 6: Prepare one promoted rig end to end**

```bash
docker compose run --rm test sh -c '
  mkdir -p /tmp/promo/rigs/Camel
  cp data/truebones/rigs/Camel/manifest.yaml /tmp/promo/rigs/Camel/
  cp data/truebones/rigs/_base.yaml /tmp/promo/rigs/ 2>/dev/null || true
  python scripts/process_dataset_truebones.py --out-root /tmp/promo --rigs Camel
  python - <<PY
from poseydon.io.bvh import BVH
a = BVH.read("/tmp/promo/clips/Camel/walk.bvh").to_animation()
print("root joint:", a.names[0], " joints:", a.n_joints)
y = a.global_positions()[0][:, 1]
print("root height fraction:", (y[0] - y.min()) / (y.max() - y.min()))
PY'
```

Expected: root joint is `Bip01`, and the height fraction is well above 0.05 (≈0.68). Before this task it would print `Hips` and ≈0.000.

- [ ] **Step 7: `ruff` and commit**

```bash
docker compose run --rm test ruff check .
git add scripts/process_dataset_truebones.py tests/build/test_promote_table.py
git commit -m "feat(prepare): wire the root-promotion table into stage 1

PromoteRoot is stage 2 and the position is contractual: after RestRelative,
whose check a promoted clip would fail; before EnforceRigid, so the preserved
translation channel is the promoted root's; and before ScaleToMeanBoneLength,
whose mean the removed connectors currently enter.

The table gates on the ground-locator classification and never on the presence
of an offset -- Lynx and BrownBear have a correct pelvis root and a chain
continuing to Bip01_Spine at a real offset, and an offset-keyed rule would
break both. The test asserts they are absent."
```

---

### Task 6: `SAMPLE_RIGS` gains Camel

Spec §8 test 3. None of Flamingo, BrownBear, Crab or Scorpion is a ground-locator rig, so promotion is covered nowhere by the round trip. Camel specifically: it exercises both new tables at once — no T-pose file, so Task 3 authors `__IdleLoop.bvh`, and a two-step promotion `Hips → C_ctrl → Bip01` rather than the single-step majority.

**Files:**
- Modify: `tests/conftest.py:18`

**Interfaces:**
- Consumes: `SAMPLE_RIGS`, used by `test_roundtrip.py` and `test_bvh_fbx_agreement.py`.
- Produces: `SAMPLE_RIGS = ("Flamingo", "BrownBear", "Crab", "Scorpion", "Camel")`.

- [ ] **Step 1: Add Camel**

```python
# One biped, one quadruped, one milliped, the rig whose reduction differs from
# the reference's (Scorpion keeps a zero-offset joint with siblings), and one
# ground-locator rig. Camel covers both stage-1 tables at once: it has no
# T-pose file, so REST_POSE_FILE authors __IdleLoop.bvh for it, and it is a
# two-step promotion (Hips -> C_ctrl -> Bip01) rather than the single-step
# majority.
SAMPLE_RIGS = ("Flamingo", "BrownBear", "Crab", "Scorpion", "Camel")
```

- [ ] **Step 2: Run the round trip**

Run: `docker compose run --rm test pytest tests/build/test_roundtrip.py -v -rs`

Expected: 10 cases, all PASSED — including `[Camel]` for both tests. `test_prepared_clips_face_plus_z_and_stand_on_the_ground[Camel]` asserts the mean bone length equals `HML_MEAN_BONE_LENGTH` on a promoted rig, which is the independent check that promotion-before-scaling works.

If `[Camel]` fails on the facing assertion, the manifest's `facing` joints may not survive promotion — check that `resolve(manifest, anim.names)` is called on the *source* names (it is, in `process_species`) and that no facing joint is in the removed chain. Camel's chain is `Hips`, `C_ctrl`; if a manifest names either as a facing joint, that is a genuine data problem to report, not to work around.

- [ ] **Step 3: Run the whole suite**

Run: `docker compose run --rm test pytest -q -rs`
Expected: green. `test_bvh_fbx_agreement.py[Camel]` skips — no `mesh.npz` for Camel until the FBX pass runs; that is the fixture's designed behaviour.

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add Camel to SAMPLE_RIGS so promotion is round-tripped

None of the existing four is a ground-locator rig, so PromoteRoot was covered
by unit tests only. Camel exercises both stage-1 tables at once: no T-pose
file, so REST_POSE_FILE authors its rest pose, and a two-step promotion."
```

---

### Task 7: Extend the round trip through features

Spec §8 test 2. The existing test walks `source → prepare → reduce → expand → unprepare` and stops. The loop the application actually runs passes through feature extraction and reconstruction, and that half is untested.

**Files:**
- Modify: `tests/build/test_roundtrip.py` — add one test beside the existing one

**Interfaces:**
- Consumes: `extract_features(anim, resolved, names) -> (np.ndarray, FeatureSpec)`, `reconstruct(name, features, spec, template) -> (positions, RigidBodyAnimation | None)`, `build_reduction`, `apply_reduction`, `invert_reduction`.
- Produces: nothing consumed later.

- [ ] **Step 1: Write the failing test**

Add to `tests/build/test_roundtrip.py`:

```python
@pytest.mark.parametrize("rig", SAMPLE_RIGS)
def test_round_trip_through_features_returns_the_source_rig(rig, raw_clips):
    """The loop the application runs: features are in the middle of it.

    The model stage is the identity -- a real model changes the values, and
    what is under test is that the plumbing survives the trip, which is what
    spec §9 promises. `fk` is the reconstruction here because the features
    carry the exact rot6d that was extracted; `positions_ik` is a
    generation-quality choice for motion whose rotations and positions
    disagree, and is not what closes this loop.
    """
    clips = raw_clips(rig)
    manifest = _manifest(rig)
    rest_path = _rest_path(clips)
    rest = BVH.read(rest_path).to_animation()
    rig_params = CHAIN.fit_rig(rest, resolve(manifest, rest.names))

    source = BVH.read(next(p for p in clips if p != rest_path)).to_animation()
    if tuple(source.names) != tuple(rest.names):
        pytest.skip(f"{rig}: this clip is rigged differently from its own rest pose")

    prepared, params = CHAIN.apply(source, resolve(manifest, source.names), rig_params)
    reduction = build_reduction(prepared)
    reduced = apply_reduction(prepared, reduction)

    resolved_reduced = resolve(manifest, reduced.names)
    features, spec = extract_features(reduced, resolved_reduced, DEFAULT_FEATURES)

    # The model stage, as the identity.
    _positions, rebuilt = reconstruct("fk", features, spec, reduced)
    assert rebuilt is not None, "`fk` must produce rotations, hence a BVH"

    expanded = invert_reduction(rebuilt, reduction)
    restored = CHAIN.invert(expanded, params)

    # Spec §9: structure and reference frame return.
    assert restored.names == source.names
    assert list(restored.parents) == list(source.parents)
    np.testing.assert_allclose(restored.offsets, source.offsets, atol=1e-9)

    # And the frame: re-preparing the returned asset reproduces the prepared
    # clip, in world space, over the joints that survived reduction. A velocity
    # feature costs the last frame, so compare the frames features kept.
    reprepared = _apply_with(CHAIN, restored, params)
    kept = features.shape[0]
    surviving = [prepared.names.index(name) for name in reduced.names]
    np.testing.assert_allclose(
        reprepared.global_positions()[:kept][:, surviving],
        prepared.global_positions()[:kept][:, surviving],
        rtol=0,
        atol=1e-6,
    )
```

Add to that file's imports:

```python
from poseydon.features import DEFAULT_FEATURES, extract_features, reconstruct
```

- [ ] **Step 2: Run it to verify it fails or reveals a real gap**

Run: `docker compose run --rm test pytest tests/build/test_roundtrip.py -k through_features -v`
Expected: FAIL initially with `NameError`/`ImportError` on the new imports. After the import is added, the test may still fail — if it does, **do not loosen the tolerance to make it pass**. A genuine mismatch here is the gap this test exists to find. Report the measured error and its shape before changing anything.

- [ ] **Step 3: Make it pass**

Add the import. If a real discrepancy appears, diagnose before adjusting: the likely causes in order are (a) `resolve` on reduced names failing because a manifest facing/foot joint was reduced away — in which case `augment/topology.py::_reindex_resolved` is the right tool and the test should route through it rather than re-resolving; (b) the frame-count mismatch from velocity features, handled by the `kept` slice above; (c) `fk` reconstructing from `rot6d` where the block ordering differs from extraction.

- [ ] **Step 4: Run the tests**

Run: `docker compose run --rm test pytest tests/build/test_roundtrip.py -v -rs`
Expected: 15 cases, all PASSED (5 rigs × 3 tests).

- [ ] **Step 5: `ruff` and commit**

```bash
docker compose run --rm test ruff check .
git add tests/build/test_roundtrip.py
git commit -m "test: carry the round trip through features

The existing test stopped at reduce -> expand. The loop the application runs
passes through extraction and reconstruction, with the model in the middle;
that half was untested. The model stage is the identity here, because what
spec §9 promises is that structure and reference frame survive the trip, not
that a model preserves values."
```

---

### Task 8: The FBX arm of the round trip

Spec §8 test 4. Blender-gated, because the test image deliberately has none.

**Files:**
- Create: `tests/build/test_roundtrip_fbx.py`

**Interfaces:**
- Consumes: `poseydon.io.fbx`, `CHAIN`, `SAMPLE_RIGS`.
- Produces: nothing consumed later.

- [ ] **Step 1: Check what the FBX writer offers**

```bash
docker compose run --rm test python -c "
import inspect, poseydon.io.fbx as m
print([n for n in dir(m) if not n.startswith('_')])
"
```

Note the read and write entry points; the test below uses them by the names this prints. If the module exposes no writer, the round trip cannot close through FBX — record that in the test as an explicit `pytest.skip` naming the missing capability, and report it, rather than inventing one.

- [ ] **Step 2: Write the test**

Create `tests/build/test_roundtrip_fbx.py`:

```python
"""The round trip must return an FBX, not only a BVH.

Blender-gated: Dockerfile.compat and the test image deliberately ship no
Blender, so this runs against the `fbx` compose service:

    docker compose run --rm fbx blender -b --python-expr \\
        "import pytest, sys; sys.exit(pytest.main(['tests/build/test_roundtrip_fbx.py','-v']))"

Skipping when Blender is absent is correct here -- the data is genuinely not
available -- unlike the skip this suite carried for Crab, which hid a rig the
code could handle.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import CORPUS, SAMPLE_RIGS

bpy = pytest.importorskip("bpy", reason="FBX round trip needs Blender")


@pytest.mark.parametrize("rig", SAMPLE_RIGS)
def test_round_trip_returns_the_source_fbx(rig, tmp_path):
    source_dir = CORPUS / "source" / rig
    if not source_dir.is_dir():
        pytest.skip(f"{rig}: no source directory")
    fbxs = sorted(p for p in source_dir.glob("*.fbx") if "ALL" not in p.name.upper())
    if not fbxs:
        pytest.skip(f"{rig}: no per-clip source FBX")

    from poseydon.io import fbx as fbx_io

    source = fbx_io.read_animation(str(fbxs[0]))

    out = tmp_path / "restored.fbx"
    fbx_io.write_animation(str(out), source)
    restored = fbx_io.read_animation(str(out))

    # Spec §9: structure and reference frame, not motion values.
    assert restored.names == source.names
    assert list(restored.parents) == list(source.parents)
    np.testing.assert_allclose(restored.offsets, source.offsets, atol=1e-5)
```

Adjust `read_animation` / `write_animation` to the real names printed in Step 1.

- [ ] **Step 3: Run it in the FBX container**

```bash
docker compose run --rm fbx blender -b --python-expr \
  "import pytest, sys; sys.exit(pytest.main(['tests/build/test_roundtrip_fbx.py','-v']))"
```

Expected: PASSED for the rigs with per-clip FBX; skips where the source has none.

- [ ] **Step 4: Confirm it skips cleanly in the normal image**

Run: `docker compose run --rm test pytest tests/build/test_roundtrip_fbx.py -v -rs`
Expected: SKIPPED with "FBX round trip needs Blender". It must not error at collection.

- [ ] **Step 5: `ruff` and commit**

```bash
docker compose run --rm test ruff check .
git add tests/build/test_roundtrip_fbx.py
git commit -m "test(fbx): close the round trip through FBX, gated on Blender

Skips in the normal image, which ships no Blender by design, and runs against
the fbx compose service."
```

---

### Task 9: Rebuild the full corpus

Spec §1. Expected yield ~1145 clips over 73 rigs, against 1153 raw and the 81-row index on disk. Exactly one file per rig is lost for Ant, Centipede, Crab, Deer, Elephant, HermitCrab, Jaguar and Trex — each rig's own outlier.

**Files:**
- Modify: `data/truebones/clips/**`, `data/truebones/rigs/*/prepare.npz` (all gitignored)
- Test: `tests/build/test_corpus_yield.py` (create)

**Interfaces:**
- Consumes: everything above.
- Produces: the prepared corpus A2 reads.

- [ ] **Step 1: Record what is on disk now**

```bash
find data/truebones/clips -name '*.bvh' | wc -l
ls data/truebones/clips | wc -l
```

Expected before: 5 rig directories. Note the numbers — Step 4 compares against them.

- [ ] **Step 2: Clear the stale prepared corpus**

The prepared BVH and `prepare.npz` on disk predate `b2b0151` and every change above. Leaving them risks a partial rebuild reading a mix.

```bash
docker compose run --rm test sh -c '
  rm -rf data/truebones/clips
  find data/truebones/rigs -name prepare.npz -delete'
```

`mesh.npz` files are **kept** — they come from the Blender pass, are not affected by this plan, and regenerating them needs the FBX container.

- [ ] **Step 3: Run stage 1 over all 73 rigs**

```bash
docker compose run --rm test python scripts/process_dataset_truebones.py 2>&1 \
  | tee /tmp/rebuild.log | tail -40
```

This takes several minutes. Expected tail: `total: ~1145 clips written across 73 rigs`, followed by the warning list.

- [ ] **Step 4: Check the yield and the warnings**

```bash
grep -c '' /dev/null; find data/truebones/clips -name '*.bvh' | wc -l
ls data/truebones/clips | wc -l
grep -E 'ValueError|Error' /tmp/rebuild.log | sed 's/.*: //' | sort | uniq -c | sort -rn
```

Expected: ~1145 BVH across 73 directories. The error list should contain **only** one rejected clip each for Ant, Centipede, Crab, Deer, Elephant, HermitCrab, Jaguar and Trex. Any other rejection is a regression from this plan — investigate before continuing, do not accept a lower count.

- [ ] **Step 5: Write the yield test**

Create `tests/build/test_corpus_yield.py`:

```python
"""The recovered rigs stay recovered, and the promoted rigs stay promoted.

b2b0151 recovered 60 clips across four rigs by choosing the rest-pose file by
modal joint set, and nothing guarded it. A regression in find_tpose would
quietly cost them again -- quietly being the point: the clips are skipped with
a warning, not an error.
"""

from __future__ import annotations

import pytest

from poseydon.io.bvh import BVH
from scripts.process_dataset_truebones import PROMOTE_ROOT
from tests.conftest import CORPUS

RECOVERED = {"Ant": 17, "Crab": 10, "Deer": 20, "Jaguar": 13}


@pytest.mark.parametrize("rig, expected", sorted(RECOVERED.items()))
def test_the_recovered_rigs_keep_their_clips(rig, expected):
    clips = CORPUS / "clips" / rig
    if not clips.is_dir():
        pytest.skip(f"{rig}: corpus not built -- run scripts/process_dataset_truebones.py")
    assert len(sorted(clips.glob("*.bvh"))) == expected


@pytest.mark.parametrize("rig", sorted(PROMOTE_ROOT))
def test_promoted_rigs_are_rooted_on_the_body(rig):
    """Measured on the PREPARED corpus, so it checks what stage 1 wrote rather
    than what the table says."""
    clips = CORPUS / "clips" / rig
    if not clips.is_dir():
        pytest.skip(f"{rig}: corpus not built")
    written = sorted(clips.glob("*.bvh"))
    if not written:
        pytest.skip(f"{rig}: no prepared clips")

    anim = BVH.read(written[0]).to_animation()
    assert anim.names[0] == PROMOTE_ROOT[rig]

    y = anim.global_positions()[0][:, 1]
    fraction = (float(y[0]) - float(y.min())) / (float(y.max()) - float(y.min()))
    assert fraction > 0.05, (
        f"{rig}: prepared root sits at height fraction {fraction:.3f}, still a "
        "ground locator"
    )
```

- [ ] **Step 6: Run the full suite against the rebuilt corpus**

Run: `docker compose run --rm test pytest -q -rs`

Expected: green. Specifically confirm:
- `test_corpus_yield.py` — 4 recovered + 14 promoted cases pass.
- `test_roundtrip.py` — 15 pass, **no skips**.
- `test_bvh_fbx_agreement.py` — cases that previously skipped for a missing rest clip now resolve one via the recorded action; the recorded rest-geometry `xfail` stays `xfail`.

- [ ] **Step 7: Re-run the FBX pass for the sample rigs**

`mesh.npz` was kept but its joint order corresponds to the *un-promoted* skeleton for the 14 promoted rigs. None of `SAMPLE_RIGS` is promoted except Camel, which has no `mesh.npz`; so nothing is stale today. Record the hazard rather than fix it here — A2 consumes `mesh.npz` and must re-derive it after promotion.

```bash
docker compose run --rm test python - <<'PY'
from pathlib import Path
from scripts.process_dataset_truebones import PROMOTE_ROOT
stale = [p.parent.name for p in Path("data/truebones/rigs").glob("*/mesh.npz")
         if p.parent.name in PROMOTE_ROOT]
print("mesh.npz predating promotion, must be regenerated in A2:", stale or "none")
PY
```

- [ ] **Step 8: `ruff` and commit**

```bash
docker compose run --rm test ruff check .
git add tests/build/test_corpus_yield.py
git commit -m "test: pin the corpus yield after the full rebuild

Ant, Crab, Deer and Jaguar keep 17/10/20/13 clips, and the 14 promoted rigs
are rooted on the body in the PREPARED corpus rather than merely in the table.
b2b0151's 60 recovered clips were guarded by nothing; a regression would cost
them silently, since a rejected clip is a warning and not an error."
```

- [ ] **Step 9: Record the outcome**

Append an `# Outcome` section to this plan: the measured clip total, the rig count, the exact rejection list, and anything that differed from the expectations above. Phase 1's outcome section is what made this plan possible; write the one that makes A2 possible.

```bash
git add docs/superpowers/plans/2026-09-08-stage-1-corrections-and-corpus-rebuild.md
git commit -m "docs(plan): record the A1 outcome"
```

---

## Self-Review

**Spec coverage.** §1 corpus rebuild → Task 9. §1.1 authored rest poses → Task 3, with the recorded-choice consolidation in Task 2. §1.2 root promotion → Tasks 4 and 5, ordering pinned in Task 5 Step 4. §8 test 1 (Crab un-skips) → Task 2. Test 2 (features leg) → Task 7. Test 3 (promoted rig) → Task 6. Test 4 (FBX arm) → Task 8. Test 5 (promotion moves nothing) → Task 4 Step 1 and Task 5 Step 1. Test 6 (authored rest table) → Task 3 Step 1. Test 7 (recovered rigs) → Task 9 Step 5. §9's contract is asserted by Tasks 4, 6 and 7. The compose `UID` defect from §1 → Task 1.

Not covered here, by design: §8 tests 8–10 (golden feature parity, normalization policy, build-then-train smoke) belong to A2, which builds the artefacts they check.

**Type consistency.** `find_tpose(clip_paths, rig="")` is used with two arguments in Task 3 Step 4 and one in `test_roundtrip._rest_path` (Task 2 Step 4) — the default covers it. `PromoteRoot(target=...)` is constructed in Task 4's tests, Task 5's chain and Task 5's test with the same keyword. `_chain_for(source_channels, rig="")` gains its parameter in Task 5 Step 4 and is called there. `prepare.npz`'s `rig/rest/clip` is written in Task 2 Step 3 and read in Task 2 Step 4 and Task 9's hazard check with the same key path.

**Known risk.** Task 7 Step 2 may reveal a real mismatch in the features leg — the manifest's facing and foot joints can be reduced away, and `resolve` on reduced names would then fail. The step says to diagnose rather than loosen the tolerance, and names `_reindex_resolved` as the likely tool. Task 8 Step 1 verifies the FBX module's actual entry points before the test assumes them.
