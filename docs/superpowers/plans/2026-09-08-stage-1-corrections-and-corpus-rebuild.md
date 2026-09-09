# Stage 1 Corrections and Corpus Rebuild — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct stage-1 preparation — authored rest poses, root promotion, and the round-trip test gaps — then rebuild the full 73-rig corpus.

**Architecture:** Each rig's manifest gains a `rest_pose` declaration, replacing a selection rule duplicated across three files and fixed in one — which is why the round trip has been silently skipping Crab. A single hardcoded table stays in `scripts/process_dataset_truebones.py`: which joint to promote to root for the 14 rigs that root at a ground locator. Promotion is a new `PrepareStage` in `src/poseydon/build/prepare.py`, placed second in the chain. The round-trip test is extended through feature extraction and reconstruction.

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

### Task 2: Declare the rest pose in the manifest

A rig's rest pose is a fact about the rig, so it belongs in the rig's manifest —
not in a table inside one script, and not re-derived by each consumer. Today it
is re-derived by three copies of one rule, and only the script has `b2b0151`'s
modal-set fix:

| location | rule | consequence |
|---|---|---|
| `scripts/process_dataset_truebones.py::find_tpose` | modal set, then name | correct |
| `tests/build/test_roundtrip.py::_rest_path` | name only | **Crab skips today** |
| `tests/build/test_bvh_fbx_agreement.py::_rest_path` | name only | will skip once Crab's rest clip is `walk.bvh` |

There is already a manifest key for this, and it is a trap. `SkeletonManifest.tpose`
is parsed, and read by `ingest/pipeline.py:94` and `data/dataset.py:145` — both
guarded by `if manifest.tpose is not None and manifest.tpose.is_file()`. **No
manifest declares it.** So both consumers have always taken their silent
fallback: `_rest_frame` uses "the first frame of the skeleton's first clip",
which means the rest frame the model is shown as a rig's identity is arbitrary
for all 73 rigs. The key reads as working, which is worse than the dead
`strip_joint_prefix` the parity spec removed.

`tpose` is also the wrong shape. It resolves relative to the manifest
(`source.parent / tpose`), but stage 1 needs a **raw source** file while stage 2
and the dataset need the **prepared** clip. So the manifest declares a
*filename* and each consumer resolves it in its own domain.

**Files:**
- Modify: `src/poseydon/core/skeleton.py` — add `rest_pose`, remove `tpose`
- Modify: `src/poseydon/ingest/pipeline.py:94-95` — drop the dead branch
- Modify: `src/poseydon/data/dataset.py:145-148` — drop the dead branch
- Modify: `scripts/process_dataset_truebones.py` — `find_tpose` becomes `rest_source`
- Modify: `tests/build/test_roundtrip.py`, `tests/build/test_bvh_fbx_agreement.py`
- Test: `tests/build/test_rest_selection.py`

**Interfaces:**
- Consumes: `SkeletonManifest.load(path) -> SkeletonManifest`, `BVH.read_names(path)`.
- Produces:
  - `SkeletonManifest.rest_pose: str | None` — a bare filename, e.g. `"__IdleLoop.bvh"`, relative to the rig's raw source directory.
  - `rest_source(manifest: SkeletonManifest, clip_paths: list[Path]) -> Path` in `scripts/process_dataset_truebones.py`. Raises `ValueError` when the manifest declares nothing, when the named file is absent, or when it does not carry the rig's modal skeleton. Never returns a fallback.
  - `rest_action(manifest: SkeletonManifest) -> str` — the prepared-clip action slug for that rest pose, for consumers working on `clips/<Rig>/`.

- [ ] **Step 1: Write the failing tests**

Replace the contents of `tests/build/test_rest_selection.py` below its existing synthetic test with:

```python
def test_manifest_round_trips_rest_pose(tmp_path):
    """A rig's rest pose is data about the rig, so it lives in the manifest."""
    from poseydon.core.skeleton import SkeletonManifest

    path = tmp_path / "manifest.yaml"
    path.write_text(
        "skeleton: Testy\n"
        "rest_pose: __IdleLoop.bvh\n"
        "facing:\n"
        "  hips: {right: R, left: L}\n"
        "contact: {max_height: 0.3, max_speed: 0.04}\n"
    )
    assert SkeletonManifest.load(path).rest_pose == "__IdleLoop.bvh"


def test_the_dead_tpose_key_is_gone(tmp_path):
    """`tpose` was parsed, read by two consumers, and declared by no manifest,
    so both consumers silently took their fallback forever. A key that reads as
    working is worse than one that is obviously dead."""
    from poseydon.core.skeleton import ManifestError, SkeletonManifest

    path = tmp_path / "manifest.yaml"
    path.write_text(
        "skeleton: Testy\n"
        "tpose: ../tposes/Testy.bvh\n"
        "facing:\n"
        "  hips: {right: R, left: L}\n"
        "contact: {max_height: 0.3, max_speed: 0.04}\n"
    )
    with pytest.raises(ManifestError, match="tpose"):
        SkeletonManifest.load(path)


def test_rest_source_refuses_a_rig_that_declares_nothing(tmp_path):
    """No silent fallback. A rig with no declared rest pose is a data error
    that must stop the build, not a rig that quietly gets an arbitrary one."""
    from poseydon.core.skeleton import SkeletonManifest
    from scripts.process_dataset_truebones import rest_source

    path = tmp_path / "manifest.yaml"
    path.write_text(
        "skeleton: Testy\n"
        "facing:\n"
        "  hips: {right: R, left: L}\n"
        "contact: {max_height: 0.3, max_speed: 0.04}\n"
    )
    manifest = SkeletonManifest.load(path)
    clip = _write(tmp_path, "__Walk.bvh", ["Hips", "Spine", "Head"])
    with pytest.raises(ValueError, match="declares no `rest_pose`"):
        rest_source(manifest, [clip])


def test_rest_source_refuses_a_declaration_outside_the_modal_set(tmp_path):
    """Trex's natural neutral pick, __STILL.bvh, IS that rig's outlier file.
    An authored value must never override the corpus's own evidence."""
    from poseydon.core.skeleton import SkeletonManifest
    from scripts.process_dataset_truebones import rest_source

    path = tmp_path / "manifest.yaml"
    path.write_text(
        "skeleton: Testy\n"
        "rest_pose: __Odd.bvh\n"
        "facing:\n"
        "  hips: {right: R, left: L}\n"
        "contact: {max_height: 0.3, max_speed: 0.04}\n"
    )
    manifest = SkeletonManifest.load(path)
    majority = ["Hips", "Spine", "Head"]
    clips = [
        _write(tmp_path, "__Walk.bvh", majority),
        _write(tmp_path, "__Run.bvh", majority),
        _write(tmp_path, "__Odd.bvh", ["Hips", "Spine"]),
    ]
    with pytest.raises(ValueError, match="modal"):
        rest_source(manifest, clips)


def test_rest_source_returns_the_declared_file(tmp_path):
    from poseydon.core.skeleton import SkeletonManifest
    from scripts.process_dataset_truebones import rest_source

    path = tmp_path / "manifest.yaml"
    path.write_text(
        "skeleton: Testy\n"
        "rest_pose: __Run.bvh\n"
        "facing:\n"
        "  hips: {right: R, left: L}\n"
        "contact: {max_height: 0.3, max_speed: 0.04}\n"
    )
    manifest = SkeletonManifest.load(path)
    majority = ["Hips", "Spine", "Head"]
    clips = [
        _write(tmp_path, "__Walk.bvh", majority),
        _write(tmp_path, "__Run.bvh", majority),
    ]
    assert rest_source(manifest, clips).name == "__Run.bvh"
```

Add `import pytest` to that file's imports.

- [ ] **Step 2: Run them to verify they fail**

Run: `docker compose run --rm test pytest tests/build/test_rest_selection.py -v`
Expected: FAIL — `rest_pose` is an unknown manifest key, and `rest_source` does not exist.

- [ ] **Step 3: Add `rest_pose`, remove `tpose`**

In `src/poseydon/core/skeleton.py`:

In `_KNOWN_KEYS`, replace `"tpose"` with `"rest_pose"`.

In `SkeletonManifest`, replace the `tpose` field:

```python
    #: Filename of the clip whose first frame is this rig's rest pose, relative
    #: to the rig's RAW source directory -- e.g. "__IdleLoop.bvh". A filename
    #: rather than a path because consumers resolve it in different domains:
    #: stage 1 wants `source/<Rig>/<file>`, everything downstream wants the
    #: prepared clip `clips/<Rig>/<action>.bvh`.
    rest_pose: str | None = None
```

In `_build`, replace the `tpose` block:

```python
    rest_pose = data.get("rest_pose")
```

and the constructor argument `tpose=tpose_path,` with `rest_pose=None if rest_pose is None else str(rest_pose),`.

- [ ] **Step 4: Delete the two dead branches**

In `src/poseydon/ingest/pipeline.py`, remove lines 94-95 — the `if manifest.tpose ...` branch and its body — leaving whatever fallback follows as the only path. It already was the only path in practice.

In `src/poseydon/data/dataset.py::_rest_frame`, remove the equivalent branch at 145-148. Update the docstring, which currently promises "From the manifest's T-pose when it names one":

```python
        """One frame describing the skeleton, cached per skeleton.

        The first frame of the skeleton's first clip. `SkeletonManifest.tpose`
        used to be consulted here, but no manifest ever declared it, so this
        fallback has always been the only path -- a guard that reads as working
        is worse than no guard. A3 replaces this with the rest frame recorded in
        `rigs/<Rig>/stats.npz`, chosen by `manifest.rest_pose`.
        """
```

- [ ] **Step 5: Replace `find_tpose` with `rest_source`**

In `scripts/process_dataset_truebones.py`, delete `_pick_by_name` and `find_tpose`, and add:

```python
def rest_action(manifest: SkeletonManifest) -> str:
    """The prepared-clip action slug for this rig's declared rest pose.

    Consumers working on `clips/<Rig>/` need the slug, not the raw filename;
    deriving it here keeps one source of truth in the manifest.
    """
    if manifest.rest_pose is None:
        raise ValueError(f"{manifest.name}: manifest declares no `rest_pose`")
    stem = Path(manifest.rest_pose).stem
    return strip_skeleton_prefix(action_slug(stem), manifest.name)


def rest_source(manifest: SkeletonManifest, clip_paths: list[Path]) -> Path:
    """The raw file supplying this rig's rest pose, validated against the corpus.

    The choice is authored -- it is a judgement about which clip's first frame
    is a neutral pose, and no rule gets that right (matching "idle" as a
    substring picks Lion's __DeathIdle.bvh, Jaguar's __LieIdle.bvh and Trex's
    __idle_attack.bvh). But an authored value never overrides the corpus's own
    evidence: the declared file must carry the rig's MODAL joint set, because
    a rest pose disagreeing with the clips makes `RestRelative` reject every
    one of them. Trex is why this guard is not ceremony -- its natural neutral
    pick, __STILL.bvh, is that rig's outlier file (66 joints against 78).

    Raises rather than falling back. A rig whose rest pose cannot be resolved
    is a data error the build must stop on; the previous behaviour returned
    "the first file in the modal set", which is what gave Crab a rest geometry
    fitted from __Attack1.bvh.
    """
    if manifest.rest_pose is None:
        raise ValueError(
            f"{manifest.name}: manifest declares no `rest_pose`; every rig must "
            "name the clip whose first frame is its rest pose"
        )

    by_name = {path.name: path for path in clip_paths}
    chosen = by_name.get(manifest.rest_pose)
    if chosen is None:
        raise ValueError(
            f"{manifest.name}: declared rest_pose `{manifest.rest_pose}` is not "
            f"among this rig's {len(clip_paths)} raw clips"
        )

    names_by_path = {path: BVH.read_names(path) for path in clip_paths}
    modal_names, modal_count = Counter(names_by_path.values()).most_common(1)[0]
    if names_by_path[chosen] != modal_names:
        raise ValueError(
            f"{manifest.name}: declared rest_pose `{manifest.rest_pose}` has "
            f"{len(names_by_path[chosen])} joints but the rig's modal skeleton "
            f"has {len(modal_names)} ({modal_count}/{len(clip_paths)} clips); "
            "a rest pose disagreeing with the clips rejects every one of them"
        )
    return chosen
```

Add `from poseydon.core.skeleton import SkeletonManifest, resolve` if `SkeletonManifest` is not already imported, and `from pathlib import Path` if absent.

In `process_species`, replace the `find_tpose` call and its fallback block with:

```python
    rest_path = rest_source(manifest, clip_paths)
```

Delete the `if rest_path is None:` fallback that followed — there is no fallback any more. Keep `warnings` as a list; `rest_source` raises instead of warning, and `process_species`'s caller reports the exception per rig.

Wrap the call so one bad manifest does not abort the corpus, matching how clip errors are already handled:

```python
    try:
        rest_path = rest_source(manifest, clip_paths)
    except ValueError as error:
        return 0, [f"{manifest.name}: {error}"]
```

- [ ] **Step 6: Point the two test copies at the manifest**

In `tests/build/test_roundtrip.py`, replace `_rest_path`:

```python
def _rest_path(clips):
    """The rig's rest-pose file, from its manifest.

    This used to carry its own copy of a filename-matching rule -- the rule
    b2b0151 fixed in the script and not here. Crab therefore resolved to its
    54-joint __TPOSE.bvh, every 64-joint clip disagreed, and the case skipped.
    """
    return rest_source(_manifest(_rig_of(clips)), list(clips))


def _rig_of(clips) -> str:
    return clips[0].parent.name
```

Add `from scripts.process_dataset_truebones import rest_source`.

In `tests/build/test_bvh_fbx_agreement.py`, replace its `_rest_path` — this one selects among *prepared* clips, so it uses the slug:

```python
def _rest_path(bvhs):
    """The rig's prepared rest clip, named by its manifest.

    `FaceAxis` is clip-scoped and `rotate_rig` turns OFFSETS with the motion, so
    every prepared clip carries a differently-rotated offset block -- only the
    rest clip's offsets correspond to mesh.npz's bind pose. Matching on the
    filename was a stale copy of a rule that no longer holds: Crab's rest clip
    is `walk`.
    """
    rig = bvhs[0].parent.name
    manifest = SkeletonManifest.load(CORPUS / "rigs" / rig / "manifest.yaml")
    path = bvhs[0].parent / f"{rest_action(manifest)}.bvh"
    if not path.is_file():
        pytest.skip(f"{rig}: declared rest clip {path.name} is not in the prepared corpus")
    return path
```

Add `from poseydon.core.skeleton import SkeletonManifest` and `from scripts.process_dataset_truebones import rest_action`.

- [ ] **Step 7: Run the tests**

Run: `docker compose run --rm test pytest tests/build/ tests/features/ -v -rs`

Expected: `test_rest_selection.py` passes. Every `test_roundtrip.py` case now **errors or fails** with "declares no `rest_pose`" — that is correct and expected, because no manifest declares one yet. Task 3 populates them. Do not add a fallback to make this green.

- [ ] **Step 8: `ruff` and commit**

```bash
docker compose run --rm test ruff check .
git add src/poseydon/core/skeleton.py src/poseydon/ingest/pipeline.py src/poseydon/data/dataset.py scripts/process_dataset_truebones.py tests/build/
git commit -m "feat(manifest): declare the rest pose per rig, and delete the dead tpose key

A rig's rest pose is a fact about the rig, so it belongs in the manifest rather
than in three copies of a selection rule -- of which only the script carried
b2b0151's modal-set fix, which is why the round trip silently skipped Crab.

SkeletonManifest.tpose was parsed and read by two consumers, and declared by
none of the 73 manifests, so both had always taken their fallback: the rest
frame the model sees as a rig identity has been the first frame of an arbitrary
clip. A key that reads as working is worse than one that is obviously dead, so
it is removed rather than populated -- it also resolved relative to the
manifest, while stage 1 needs a raw file and everything downstream needs the
prepared clip.

rest_source raises instead of falling back. The modal-set check stays as the
guard over the authored value: Trex's natural pick, __STILL.bvh, is that rig's
outlier file."
```

---

### Task 3: Populate `rest_pose` in all 73 manifests

Spec §1.1. Every rig declares its rest pose; nothing is discovered at runtime.
Fourteen land on an idle clip, three on a gait because they have no idle at all,
two on a fly loop, and the remaining 56 on their T-pose.

**Files:**
- Create: `tools/write_rest_pose.py`
- Modify: `data/truebones/rigs/*/manifest.yaml` (73 files)
- Test: `tests/build/test_rest_selection.py`

**Interfaces:**
- Consumes: `rest_source`, `SkeletonManifest`, `BVH.read_names`.
- Produces: every manifest carries `rest_pose: <filename>`. Task 5's promotion test and Task 9 read it.

- [ ] **Step 1: Write the failing test**

Append to `tests/build/test_rest_selection.py`:

```python
def test_every_rig_declares_a_rest_pose_that_exists_and_is_modal():
    """The declaration is a judgement about pose content and cannot be tested.
    What can be: it exists, and it agrees with the rig's own clips."""
    from poseydon.core.skeleton import SkeletonManifest
    from scripts.process_dataset_truebones import rest_source

    source = CORPUS / "source"
    if not source.is_dir():
        pytest.skip("Truebones corpus not present")

    manifests = sorted((CORPUS / "rigs").glob("*/manifest.yaml"))
    assert manifests, "no rig manifests found"

    for path in manifests:
        manifest = SkeletonManifest.load(path)
        rig_dir = source / manifest.name
        if not rig_dir.is_dir():
            continue
        clips = sorted(rig_dir.glob("*.bvh"))
        if not clips:
            continue
        # Raises on: no declaration, missing file, or non-modal skeleton.
        rest_source(manifest, clips)
```

Add `from tests.conftest import CORPUS` to that file's imports.

- [ ] **Step 2: Run it to verify it fails**

Run: `docker compose run --rm test pytest tests/build/test_rest_selection.py -k every_rig -v`
Expected: FAIL — `Alligator: manifest declares no rest_pose`.

- [ ] **Step 3: Write the generator**

Create `tools/write_rest_pose.py`:

```python
"""Emit a `rest_pose:` line for every rig, for review before it is applied.

The rule is T-pose, then idle, then walk -- or fly, for a flying creature --
applied WITHIN the rig's modal joint set. The 17 rigs whose named T-pose is
missing or unusable are listed explicitly below, because no rule gets them
right: matching "idle" as a substring picks Lion's __DeathIdle.bvh, Jaguar's
__LieIdle.bvh and Trex's __idle_attack.bvh.

Run, review the output, then re-run with --apply.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from poseydon.io.bvh import BVH

RAW = Path("data/truebones/source")
RIGS = Path("data/truebones/rigs")

#: Rigs whose named T-pose is absent or declares a minority skeleton.
AUTHORED: dict[str, str] = {
    "Anaconda": "__Idle.bvh",
    "Ant": "__Idle.bvh",
    "Bird": "__IdleLoop.bvh",
    "Camel": "__IdleLoop.bvh",
    "Crab": "__Walk.bvh",
    "Cricket": "__Idle.bvh",
    "Deer": "__Idle.bvh",
    "Dog": "__Idle.bvh",
    "Goat": "__Idle.bvh",
    "Jaguar": "__Idle.bvh",
    "Lion": "__SlowIdle.bvh",
    "Monkey": "__Idle1.bvh",
    "Pteranodon": "__FlyLoop.bvh",
    "Rat": "__Trottle.bvh",
    "SabreToothTiger": "__Startwalk.bvh",
    "Scorpion-2": "__Idle.bvh",
    "Trex": "__walk_loop.bvh",
}


def choose(rig: str) -> tuple[str | None, str]:
    rig_dir = RAW / rig
    bvhs = sorted(rig_dir.glob("*.bvh"))
    if not bvhs:
        return None, "no raw clips"

    names = {p: BVH.read_names(p) for p in bvhs}
    modal, count = Counter(names.values()).most_common(1)[0]
    modal_paths = [p for p in bvhs if names[p] == modal]

    authored = AUTHORED.get(rig)
    if authored is not None:
        match = next((p for p in modal_paths if p.name == authored), None)
        if match is None:
            return None, f"AUTHORED {authored} absent or non-modal"
        return match.name, f"authored ({count}/{len(bvhs)} modal)"

    for key in ("tpos",):
        for p in modal_paths:
            if key in p.name.lower():
                return p.name, f"tpose ({count}/{len(bvhs)} modal)"
    return None, "no T-pose in the modal set and no authored entry"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    problems = []
    for manifest_path in sorted(RIGS.glob("*/manifest.yaml")):
        rig = manifest_path.parent.name
        filename, why = choose(rig)
        print(f"{rig:<18} {str(filename):<24} {why}")
        if filename is None:
            problems.append(rig)
            continue
        if not args.apply:
            continue
        text = manifest_path.read_text()
        if "rest_pose:" in text:
            continue
        lines = text.splitlines()
        # After `skeleton:`, so the file reads identity-first.
        at = next(i for i, line in enumerate(lines) if line.startswith("skeleton:"))
        lines.insert(at + 1, f"rest_pose: {filename}")
        manifest_path.write_text("\n".join(lines) + "\n")

    if problems:
        print(f"\nUNRESOLVED: {problems}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Review the proposed values**

```bash
docker compose run --rm test python tools/write_rest_pose.py
```

Expected: 73 lines, no `UNRESOLVED`. Check that the 17 authored rigs show `authored` and the rest show `tpose`. If any rig is unresolved, it needs an `AUTHORED` entry — decide it by looking at that rig's clip list, and do not let the generator invent one.

- [ ] **Step 5: Apply**

```bash
docker compose run --rm test python tools/write_rest_pose.py --apply
git diff --stat data/truebones/rigs/
```

Expected: 73 manifests changed, one line added each.

- [ ] **Step 6: Spot-check the diff**

```bash
git diff data/truebones/rigs/Camel/manifest.yaml data/truebones/rigs/Crab/manifest.yaml data/truebones/rigs/Trex/manifest.yaml
```

Expected: `rest_pose: __IdleLoop.bvh`, `rest_pose: __Walk.bvh`, `rest_pose: __walk_loop.bvh` — and specifically **not** `__STILL.bvh` for Trex.

- [ ] **Step 7: Run the tests**

Run: `docker compose run --rm test pytest tests/build/ -v -rs`
Expected: `test_rest_selection.py` fully green, and `test_roundtrip.py` back to passing — including `[Crab]`, which is the point of Task 2 and this one together.

- [ ] **Step 8: Verify Crab actually round-trips**

Run: `docker compose run --rm test pytest "tests/build/test_roundtrip.py::test_round_trip_returns_the_source_rig[Crab]" -v -rs`
Expected: PASSED, not SKIPPED.

- [ ] **Step 9: `ruff` and commit**

```bash
docker compose run --rm test ruff check .
git add tools/write_rest_pose.py data/truebones/rigs tests/build/test_rest_selection.py
git commit -m "feat(manifest): declare rest_pose for all 73 rigs

Nothing is discovered at runtime any more. 56 rigs land on their T-pose, 14 on
an idle clip, three on a gait because they have no idle at all (Crab,
SabreToothTiger, Rat) and two on a fly loop (Bird, Pteranodon).

The values are authored because no rule gets them right -- matching idle as a
substring picks a dying pose, a lying pose and a crouched attack -- but every
one is checked against its rig's modal joint set, which is what stops Trex
resting on __STILL.bvh, its outlier file."
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
- Consumes: `PromoteRoot(target=...)` from Task 4, `SkeletonManifest.rest_pose` from Tasks 2 and 3.
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

import numpy as np
import pytest

from poseydon.core.skeleton import SkeletonManifest
from poseydon.io.bvh import BVH
from scripts.process_dataset_truebones import PROMOTE_ROOT, rest_source
from tests.conftest import CORPUS

LOCATOR_BAND = 0.05


def _rest_anim(rig: str):
    rig_dir = CORPUS / "source" / rig
    if not rig_dir.is_dir():
        pytest.skip(f"{rig}: no source directory")
    bvhs = sorted(rig_dir.glob("*.bvh"))
    manifest = SkeletonManifest.load(CORPUS / "rigs" / rig / "manifest.yaml")
    return BVH.read(rest_source(manifest, bvhs)).to_animation()


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

In `scripts/process_dataset_truebones.py`, immediately after `rest_source`:

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
# T-pose file, so its manifest declares __IdleLoop.bvh as its rest_pose, and
# it is a two-step promotion (Hips -> C_ctrl -> Bip01) rather than the
# single-step majority.
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
file, so its manifest declares one, and a two-step promotion."
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
modal joint set, and nothing guarded it. A regression in rest selection would
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

**Type consistency.** `rest_source(manifest, clip_paths) -> Path` is defined in Task 2 Step 5 and called in Task 2 Step 6, Task 3 Steps 1 and 3, and Task 5's `_rest_anim` — same two positional arguments throughout. `rest_action(manifest) -> str` is defined beside it and used only by `test_bvh_fbx_agreement._rest_path`. `SkeletonManifest.rest_pose` is added in Task 2 Step 3, populated in Task 3, and read in Tasks 2, 3 and 5. `PromoteRoot(target=...)` is constructed in Task 4's tests, Task 5's chain and Task 5's test with the same keyword. `_chain_for(source_channels, rig="")` gains its parameter in Task 5 Step 4 and is called there.

**Known risk.** Task 7 Step 2 may reveal a real mismatch in the features leg — the manifest's facing and foot joints can be reduced away, and `resolve` on reduced names would then fail. The step says to diagnose rather than loosen the tolerance, and names `_reindex_resolved` as the likely tool. Task 8 Step 1 verifies the FBX module's actual entry points before the test assumes them.

# Outcome

Completed 2026-09-09, one review round. 3 commits in the original pass,
`0486820..6058542` (on top of the 17 commits already on the branch for
Tasks 1–8, `315ad61..af7522f`), plus a fix round addressing a review that
found a real structural round-trip breach. See "Fix round 1" below for the
fix-round commits and verification. Suite after the fix round: 124 passed,
11 skipped, 4 xfailed, 0 xpassed. `ruff` clean.

**Measured yield:** 1145 clips across 73 rig directories — exactly the
plan's prediction. `find data/truebones/clips -name '*.bvh' | wc -l` → 1145;
`ls data/truebones/clips | wc -l` → 73.

**Rejection list (8 clips, one per rig, exactly as predicted):**

| Rig | File | Reason |
|---|---|---|
| Ant | `__TPOSE.bvh` | 44 vs 51 joints against the rest pose it was fitted against |
| Crab | `__TPOSE.bvh` | 54 vs 64 joints |
| Deer | `__TPOSE.bvh` | 46 vs 51 joints |
| Jaguar | `__TPOSE.bvh` | 53 vs 59 joints |
| HermitCrab | `__Take_001.bvh` | 71 vs 83 joints |
| Centipede | `__Take_001.bvh` | this one clip lacks `BN_Toe01_L_01` — verified present in the other 10/11 raw clips; the manifest names the right joint |
| Elephant | `__Take_001.bvh` | this one clip lacks `Bip01_R_Toe0` — verified present in the other 14/15 raw clips; the manifest names the right joint |
| Trex | `__STILL.bvh` | this one clip lacks `jt_ClawMiddle_R` — verified present in the other 70/71 raw clips; the manifest names the right joint |

All eight rejections are the same underlying cause: the outlier clip carries
a reduced joint set relative to the rig's modal skeleton. For Ant/Crab/
Deer/Jaguar/HermitCrab it surfaces as a joint-count mismatch at
`rest_relative` fitting; for Centipede/Elephant/Trex it surfaces earlier, at
manifest resolution, because the specific joint the manifest's `foot_joints`
names happens to be one the outlier clip is missing. **The manifest is
correct in all three cases** — do not "fix" `Centipede`/`Elephant`/`Trex`'s
`foot_joints` to a joint the outlier clip happens to have; that would be
fitted to the wrong (1-in-11, 1-in-15, 1-in-71) clip and break resolution
for the many good ones.

No other rejection occurred. `RECOVERED` counts from Task 9 Step 5 match
exactly: Ant 17, Crab 10, Deer 20, Jaguar 13.

## A regression the plan didn't anticipate: Tukan crashed the whole rebuild

Running stage 1 over all 73 rigs (rather than the 5-rig stale corpus or the
5-sample-rig test suite) surfaced a bug in `PromoteRoot.fit` (Task 5) that no
existing test exercised: `test_promote_table.py` checks every `PROMOTE_ROOT`
entry's height fraction but never calls `.fit()` except for Camel, so
Tukan's case was never run before this task.

Tukan's `Hips` has two children: `N_ALL` (the real skeleton, leading to the
promotion target `locator`) and `MESH` (an FBX geometry-holder subtree —
`ESI1_Body`, `body01` — with offset `(0,0,0)` and rotation std ≈1e-15 across
every one of its 9 raw clips; verified, not assumed). `PromoteRoot.fit`'s
single-child-chain guard correctly refused to promote, since naively slicing
away `Hips` would have left `MESH` pointing at a parent index that no longer
existed — but the guard had no way to say "this sibling is provably dead,
drop it" versus "this sibling is a real, silently-losable joint." The
uncaught `ValueError` propagated out of `process_species` (which only
wraps per-clip and `rest_source` failures, not the rig-level `fit_rig`
call) and killed `main()`'s loop entirely — 15 of 73 rigs were never
attempted in the first run.

Fixed in `0486820`: `PromoteRoot.fit` now tolerates an extra child if every
joint in its subtree has an exactly-zero offset (so it occupies no space and
nothing surviving can depend on it), records the dropped indices as
`dead_indices`, and `apply` generalizes from a contiguous index slice to an
explicit keep-list so the subtree is actually removed rather than silently
misindexed. Anything else with more than one child (a real, non-dead
sibling) still raises the original error — no other rig in the corpus hit
this path, and `test_promote_table.py` / `test_prepare.py`'s existing
PromoteRoot cases still passed unchanged.

**This first version of the fix was itself a spec violation, caught in
review** (see "Fix round 1"): `invert` didn't reinsert the dropped subtree,
so Tukan silently round-tripped 26 source joints down to 19 and back up to
only 21 — a structural loss, not one of spec §9's four documented
value-level losses. Fixed for real in the fix round: `fit` now records the
dead joints' names, offsets and original parent indices (not just their
indices), and `invert` splices them back at their exact original position
with identity rotation and their own zero rest offset — the same treatment
already given to the promoted chain above the target. `Tukan` is now in
`SAMPLE_RIGS` specifically to keep this honest going forward.

This was necessary, not optional: without some fix, Tukan's 9 clips are
unreachable and the full rebuild cannot complete at all, let alone hit the
predicted 1145. It was not on the plan's list of files to modify
(`src/poseydon/build/prepare.py` wasn't expected to need a change in Task
9), so it is its own commit with its own rationale, ahead of the test-only
commit the task brief specified.

## `tests/build/` against the rebuilt corpus

`test_corpus_yield.py`: all 4 recovered-rig cases and all 14 promoted-rig
cases pass. `test_roundtrip.py`: 15 pass, no skips, as expected.

`test_bvh_fbx_agreement.py::test_bvh_and_fbx_agree_on_rest_geometry` changed
status for 3 of its 4 sample rigs, and this is worth flagging rather than
glossing over. The rest-pose selection fix (`b2b0151`) means every sample
rig's declared rest clip now actually exists under `clips/<rig>/`, so
`_rest_path` no longer skips ("declared rest clip is not in the prepared
corpus") for any of them — the case the brief asked to watch for. Measuring
the actual disagreement now that it runs:

| Rig | Rest clip used | Disagreement (bone-length fraction) | Status |
|---|---|---|---|
| Flamingo | `tpose.bvh` | 2.4e-5 | PASS (was blanket-xfailed at a claimed 6.8e-2) |
| BrownBear | `tpose.bvh` | 1.8e-5 | PASS (was blanket-xfailed at a claimed 5.5e-2) |
| Scorpion | `tpose.bvh` | 1.0e-5 | PASS (was blanket-xfailed at a claimed 3.3e-2) |
| Crab | `walk.bvh` | **0.998** | XFAIL, rig-specific now (recorded xfail reason claimed 2.0e-5, "passes") |

**Fixed in the fix round, not left as a stale XPASS.** The original pass of
this task left the blanket `@pytest.mark.xfail(strict=False)` marker in
place across all four sample rigs, which meant Flamingo/BrownBear/Scorpion
showed as XPASS — asserting nothing, so a regression back to 6.8e-2 would
have stayed green. Review caught this as the exact failure mode this whole
plan exists to close. Fixed: the blanket marker is gone; the test now
enforces the real assertion for Flamingo/BrownBear/Scorpion (which pass at
2-3 orders of magnitude better than the old recorded numbers, likely from
this plan's other fixes — the zero-length-bone scale corrections mentioned
in the same test file) and calls `pytest.xfail(...)` only for Crab, with a
corrected reason. Crab's real disagreement is ~1.0 bone lengths at
`BN_Leg_R_12`, not the 2e-5 the old blanket text claimed (that number
predates the rest-pose-selection fix). Likely cause: Crab's `mesh.npz` was
built by the Blender/FBX pass against whatever rest reference it used
*before* the rest-pose-selection fix, and now that the BVH side correctly
resolves to `walk.bvh` (Task 2/3's fix), the two sides are comparing
geometry from different source clips. This is the same underlying hazard as
the promoted-rig `mesh.npz` staleness below, just for a rig outside
`PROMOTE_ROOT` — **Crab's `mesh.npz` also needs regenerating in A2**, and
the two hardcoded `pytest.xfail(...)` calls in
`test_bvh_and_fbx_agree_on_the_skeleton_structure` and
`test_bvh_and_fbx_clips_share_basenames` for Crab describe a "1 clip
survives" state that Task 9 no longer produces (Crab now keeps 10 clips) —
their reasoning text is stale documentation, not a live check, and should
be re-examined in the next plan rather than trusted as current.

Full suite (`pytest -q -rs`, 139 collected after Tukan joined `SAMPLE_RIGS`):
124 passed, 11 skipped (Blender-only and missing-FBX/mesh.npz cases, all
pre-existing and expected, now including Tukan since it has no `mesh.npz`),
4 xfailed, **0 xpassed**, 0 failed. No test in the suite asserts nothing
anymore.

## Step 7 hazard check

```
mesh.npz predating promotion, must be regenerated in A2: none
```

None of the 14 `PROMOTE_ROOT` rigs currently has a `mesh.npz` on disk (only
`SAMPLE_RIGS` do, and only Flamingo/BrownBear/Crab/Scorpion have one —
Camel has none and is the only promoted rig in that sample), so there is
nothing stale from promotion specifically today. This is a **now** answer,
not a standing guarantee: the moment A2 or a future task runs the FBX
container for a promoted rig from source (not from a currently-committed
`mesh.npz`), the joint order must be derived post-promotion, and the code
comment in `scripts/process_dataset_truebones.py` at `PROMOTE_ROOT` already
says so.

## What a follow-on plan needs to know

- The corpus at `data/truebones/clips/` and `data/truebones/rigs/*/prepare.npz`
  is now complete and current: 1145 clips, 73 rigs. **The `prepare.npz`
  files on disk are stale against the fix-round code** — the fix round
  changed what `PromoteRoot.fit` stores (`dead_names`/`dead_offsets`/
  `dead_parents` alongside `dead_indices`), and the corpus was *not*
  rebuilt after that change (rebuilding was explicitly out of scope for the
  fix round). Both `clips/` and `prepare.npz` are gitignored — re-derive
  them with `python scripts/process_dataset_truebones.py` before trusting
  `prepare.npz` for anything that reads `promote_root`'s params (Tukan's
  entry specifically; the other 72 rigs' params are unaffected in content,
  only Tukan gained new keys).
- `PromoteRoot` has a new, narrower escape hatch (dead-subtree dropping,
  with full structural reinsertion on invert) that only Tukan currently
  exercises. If a future rig needs the same treatment, the criterion is
  "every joint in the extra subtree has an exactly-zero offset" — not "the
  joint is named like a mesh helper." Spec §9's structure guarantee (joint
  count, names, parent array, hierarchy order) holds for Tukan's round trip
  exactly as it does for every other rig; `SAMPLE_RIGS` includes Tukan now
  specifically to keep that true going forward.
- **Crab's `mesh.npz` needs regenerating**, same as the 14 promoted rigs',
  before `test_bvh_and_fbx_agree_on_rest_geometry[Crab]` and the two
  hardcoded Crab `xfail`s in `test_bvh_fbx_agreement.py` can be trusted —
  right now they describe corpus states (1 surviving clip) that no longer
  exist. This is the one open item this plan is handing forward; everything
  else it set out to fix (yield, promotion, round-trip structure, the
  rest-geometry regression guard) is done.
- No rig other than the 8 predicted ones lost a clip; the yield is exactly
  the plan's prediction, and all 8 share one root cause (the rejected clip
  carries a reduced joint set relative to the rig's modal skeleton) even
  though it surfaces at two different points in the pipeline. No manifest
  is missing `rest_pose` — the "declares no `rest_pose`" raise path was
  never hit. No manifest's `foot_joints`/`facing` entries are wrong for
  Centipede, Elephant or Trex — do not "fix" them.

## Fix round 1

Review found three issues in the pass above. All three are fixed, in
`19ac4a4`.

**Finding 1 (Critical) — structural round-trip breach.** The first version
of the Tukan fix (`0486820`) dropped `MESH`'s subtree in `apply` but never
reinserted it in `invert`: source 26 joints -> prepared 19 -> restored 21,
missing `MESH`/`ESI1_Body`/`ESI1_Body_End`/`body01`/`body01_End`. Silent —
no error, and `test_roundtrip.py` didn't catch it only because `Tukan`
wasn't in `SAMPLE_RIGS`. Spec §9's structure guarantee is non-negotiable;
this was a fifth, undocumented, structural loss against its four documented
value-level ones.

Fixed: `PromoteRoot.fit` now records the dead joints' `dead_names`,
`dead_offsets` and `dead_parents` (original absolute parent indices)
alongside the existing `dead_indices`; `invert` splices them back at their
exact original position with identity rotation and their own zero offset —
the treatment already given to the ancestor chain above the target, exact
because the dead subtree never moved. `Tukan` was added to
`tests/conftest.py`'s `SAMPLE_RIGS` to guard this going forward.

Verified:
```
$ docker compose run --rm test python - <<'PY'
from poseydon.io.bvh import BVH
from poseydon.build.prepare import PromoteRoot
b = BVH.read("data/truebones/source/Tukan/__Fly.bvh")
anim = b.to_animation()
stage = PromoteRoot(target="locator")
params = stage.fit(anim, None)
applied = stage.apply(anim, params)
restored = stage.invert(applied, params)
print("source", anim.n_joints, "prepared", applied.n_joints, "restored", restored.n_joints)
print("names match:", restored.names == anim.names)
print("parents match:", list(restored.parents) == list(anim.parents))
PY
source 26 prepared 19 restored 26
names match: True
parents match: True
```
`test_round_trip_returns_the_source_rig[Tukan]` passes: all 26 joints, in
`tests/build/`'s green run below. As with every other promoted rig, only
the ancestor chain (`Hips`/`N_ALL`, and now `MESH` which shares `Hips`'s
world-position ambiguity) does not return to its exact world position —
that is spec §9's accepted value-level loss, unchanged, and `locator` and
its descendants (the real skeleton) return exactly.

**Finding 2 (Important) — three rejections misdiagnosed.** The original
Outcome table attributed Trex/Centipede/Elephant's rejections to the
manifest naming the wrong `foot_joints`. Verified directly: the named joint
(`jt_ClawMiddle_R`, `BN_Toe01_L_01`, `Bip01_R_Toe0`) is present in all but
one raw clip per rig (70/71, 10/11, 14/15) — the manifests are correct, and
it is the outlier clip (the same one already causing the other five
rejections) that carries a reduced joint set. Table and surrounding text
corrected above; explicitly flagged as "do not fix the manifest" in the
follow-on notes.

**Finding 3 (Minor) — an assertion-free XPASS.** The blanket
`@pytest.mark.xfail(strict=False)` on
`test_bvh_and_fbx_agree_on_rest_geometry` meant Flamingo/BrownBear/Scorpion
showed as XPASS after this task's other fixes closed their disagreement
from ~5-7% of a bone length to ~1-2e-5 — asserting nothing, so a regression
back to 6.8e-2 would have stayed green. Fixed: the marker is now
rig-specific (Crab only, via an in-body `pytest.xfail(...)` matching this
file's existing per-rig pattern), with a corrected reason reflecting Crab's
actual ~1.0 bone-length disagreement rather than the stale 2e-5 claim.

**Verification (real output):**
```
$ docker compose run --rm test pytest tests/build/ -v -rs
...
105 passed, 11 skipped, 4 xfailed in 2.23s
```
`test_round_trip_returns_the_source_rig[Tukan]` PASSED (26/26 joints).
`test_bvh_and_fbx_agree_on_rest_geometry[Flamingo/BrownBear/Scorpion]` now
PASS (not XPASS); `[Crab]` XFAILs with the corrected reason.

```
$ docker compose run --rm test pytest -q -rs
...
124 passed, 11 skipped, 4 xfailed in 2.30s
```
0 failed, **0 xpassed** (down from 3).

```
$ docker compose run --rm test ruff check .
All checks passed!
```

**The corpus rebuild was NOT re-run**, per instruction. This means
**`data/truebones/rigs/Tukan/prepare.npz` is now stale against the fix-round
code**: `PromoteRoot.fit`'s `promote_root` params gained three new keys
(`dead_names`, `dead_offsets`, `dead_parents`) that the on-disk file for
Tukan does not have, since it was written by the pre-fix-round code. The
other 72 rigs' `prepare.npz` are unaffected in content (their `dead_*`
arrays were always empty and the shape/keys for non-promoted-with-dead-
subtree rigs didn't change). **A rebuild is required before Tukan's
`prepare.npz` can be used to invert a prepared clip** — flagged explicitly
per instruction, and already carried into "What a follow-on plan needs to
know" above.

Commits: `19ac4a4` (the fix, `src/poseydon/build/prepare.py`,
`tests/conftest.py`, `tests/build/test_bvh_fbx_agreement.py`) plus this
outcome-doc update.

## Addendum — three rigs do not face +Z (found by the controller, after the fix round)

Regenerating the corpus after Task 9's fix (needed because `PromoteRoot.fit` gained
`dead_names`/`dead_offsets`/`dead_parents`, making every pre-fix `prepare.npz` stale)
surfaced a warning class the Outcome above does not mention. Stage 1's own sanity
check reports the prepared clip's forward axis against +Z, and three rigs miss the
0.99 threshold badly:

| rig | forward · +Z | rest file |
|---|---|---|
| Tukan | 0.7579 | `tpose.bvh` |
| Trex | 0.8207 | `walk_loop.bvh` |
| Crow | 0.9433 | `tpose.bvh` |

Both the frame-0 pose and the rest skeleton are off by the same amount in each case,
so this is a rig-level facing problem, not a per-clip one. All three are promoted
rigs, which is suggestive but not conclusive — the other eleven promoted rigs are
clean, so promotion alone does not explain it.

**These are newly surfaced, not newly caused.** Before this plan only five rigs had
ever been prepared (BrownBear, Crab, Flamingo, Goat, Scorpion) and none of these
three was among them, so no run had ever measured them. The likely cause is the
`facing:` joint pairs in each rig's manifest — `FaceAxis` derives its rotation from
those pairs at frame 0, so a pair naming near-collinear or mis-sided joints yields a
rotation that does not actually square the character up. Tukan and Crow are
particularly odd because their rest file IS a T-pose, which should be the easiest
case to align.

It matters: those rigs' clips teach the model a systematically wrong orientation, and
the facing convention is what the whole cross-topology representation is anchored to.
It is out of scope here — A1 corrects stage-1 code and rebuilds; the `facing:` pairs
are authored manifest data, and fixing them needs the rigs inspected in a DCC tool.

**A follow-on plan should:** inspect Tukan, Trex and Crow in Blender, correct their
manifest `facing:` pairs (or `extra_yaw_deg`), re-run stage 1 for those three, and
consider promoting the sanity check from a warning to a hard failure once the corpus
is clean — a warning in a 73-rig run scrolls past, which is how this went unnoticed.
