"""The data round trip, on every prepared clip of one dataset.

    raw source --(stage 1, on disk)--> prepared clip --reduce--> model view
        model view --expand--> prepared clip                   (reduction leg)
        model view --features--> --fk--> model view            (features leg)
        prepared --invert each stage--> unprepared             (preparation leg)
        model view --RigRecord.restore, BVH.from_animation--> file    (export leg)

Each leg is checked on its own, so a failure names the step that broke: the
reduction, one preparation stage by name, the features, or the written file.
The file is then compared against the RAW SOURCE: the one reference the
pipeline did not produce. It is read through the dataset's own `source:` (a
BVH file, or UniMate's FBX cache) and cut to the frames the clip kept
(`clip_params[action]["frames"]["window"]`); what only a BVH has (CHANNELS, the
OFFSET block) is asserted for `bvh_files` alone.

`test_round_trip_fbx` closes the same loop in FBX, the format animators use.
It needs the Autodesk FBX SDK, which only the `fbxsdk` container has, and runs
there alongside the BVH test. One test per clip, so `-k Camel` or
`-k Camel/bite` narrows the probe:

    docker compose run --rm -T test python -m pytest -q tests/test_roundtrip.py --dataset truebones
    docker compose run --rm -T fbxsdk python -m pytest -q tests/test_roundtrip.py --dataset truebones

What is asserted is what the round trip promises (docs/guides/concepts/data/roundtrip.md):
structure exact, root trajectory exact, every observable world rotation exact,
bones rigid. What it loses by design is not asserted: per-joint translation
(`EnforceRigid`, one rigid geometry per rig), the locator chain above a
promoted root and its dead subtree (constant pose), and the root's own
rotation where the reduction folded a joint into it (End Sites, the only
joints it drops, carry none). Joints `prune_unskinned` removed come back at
their bind pose with no motion: their world rotation is not asserted and their
position only to the bound `_pruned_tolerance` states. The features leg
checks both rotation layouts; only `rot6d_joint` brings every joint's own
rotation back (`rot6d_bone` has no slot for a leaf's).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scripts.process_dataset_truebones import chain_for

from poseydon.build.corpus import rest_source
from poseydon.build.pipeline import compose_build
from poseydon.build.sources import SOURCES
from poseydon.core.animation import RigidBodyAnimation
from poseydon.core.rotations import quat_between, quat_inverse, quat_mul
from poseydon.core.skeleton import SkeletonManifest, contact_defaults, resolve
from poseydon.export.rig import RigRecord
from poseydon.features import extract_features, reconstruct
from poseydon.features.reduce import (
    COLLAPSE,
    DROP,
    apply_reduction,
    invert_reduction,
    load_still,
    reduction_layout,
)
from poseydon.ingest.align import rotate_rig
from poseydon.io.bvh import BVH

# BVH stores six decimals, so a write/read cycle is exact only to that.
POS_TOL = 1e-5  # fraction of the rig's extent
EXACT_TOL = 1e-9  # same, for legs that never touch a file
# FBX keys are float32, and their rounding adds up along a chain: Anaconda's
# 27-joint spine reaches 1.35e-5 in its tip's frame. Any real error is far larger.
FBX_TOL = 1e-4
ROT_TOL_DEG = 1e-3
#: Removed joints whose position is not checked, with the reason. None yet.
PRUNED_SKIP: dict[tuple[str, str], str] = {}


def pytest_generate_tests(metafunc):
    if "clip" not in metafunc.fixturenames:
        return
    dataset = metafunc.config.getoption("--dataset")
    # The clips training reads: the index's (a one-frame rest pose is left out of it).
    root = Path("data") / dataset
    clips = sorted(root / json.loads(line)["path"] for line in (root / "index.jsonl").read_text().splitlines() if line)
    metafunc.parametrize(
        "clip", clips, ids=[f"{p.parent.name}/{p.stem}" for p in clips]
    )


_records: dict[tuple[str, str], RigRecord] = {}
_configs: dict = {}


def _config(dataset: str):
    """`configs/build.yaml` for this dataset (`data=<dataset>`)."""
    if dataset not in _configs:
        _configs[dataset] = compose_build([f"data={dataset}"])
    return _configs[dataset]


def _record(corpus: Path, raw: Path, rig: str, layout: str = "rot6d_joint") -> RigRecord:
    if (rig, layout) not in _records:
        _records[rig, layout] = RigRecord.load(corpus, raw, rig, layout=layout)
    return _records[rig, layout]


_sources: dict = {}


def _raw_source(dataset: str, manifest: SkeletonManifest, action: str, window=None):
    """The raw clip as its source gives it, cut to the frames the prepared clip kept.

    Returns the `SourceClip` (its `channels` are the file's, BVH only) with its
    animation windowed to ``(start, stop, step)``; no window is the whole clip."""
    data = _config(dataset).data
    if dataset not in _sources:
        _sources[dataset] = SOURCES.build(data.source)
    try:
        clip = _sources[dataset].clip(data, manifest, action)
    except ValueError as error:
        pytest.fail(f"{manifest.name}/{action}: the source cannot read this clip: {error}")
    if window is not None:
        start, stop, step = (int(x) for x in window)
        a = clip.anim
        clip = replace(clip, anim=replace(
            a, rotations=a.rotations[start:stop:step], translations=a.translations[start:stop:step],
            fps=a.fps / step,
        ))
    return clip


def _pruned_tolerance(source, pruned: set[int]) -> dict[int, tuple[int, float]]:
    """For each joint `prune_unskinned` removed whose nearest kept ancestor k has a kept
    child (else k is a leaf of the prepared rig, and a leaf's rotation is stored nowhere,
    End Site style, so the joint hangs off an unknown turn): k, and how far the vector from k to the joint may sit from the source's,
    in world units.

    Measured from k, not the world origin: a kept joint's position is only as exact
    as its chain's translations, which `EnforceRigid` drops by design, while its world
    rotation is exact. A removed joint comes back at its bind pose, motionless, so the
    vector k to j is off through the removed joints between them, j included:

    - rotation: a removed joint a above j turns less than 2 degrees from its first
      frame, and turning a point at distance d by 2 degrees moves it 2 d sin(1 deg),
      with d the distance from a to j in the source's frame 0;
    - translation: the pruning also removes joints that translate or never carry
      a translation (a source node with none, a stretching leaf), and the restored
      one sits at its bind offset: the largest distance the source's local
      translation strays from it, over the clip.

    The bound is the sum of both over the chain. What only the pruning's own tests
    can say (a control bone that moves far, a leaf that stretches) shows as a miss
    over the bound and goes in `PRUNED_SKIP` by name, with the reason.
    """
    pos = source.global_positions()[0]
    turn = 2 * np.sin(np.radians(1.0))
    stray = np.linalg.norm(source.translations - source.offsets, axis=-1).max(axis=0)
    parents = source.parents
    out = {}
    for j in pruned:
        total, a = stray[j], int(parents[j])
        while a in pruned:
            total += turn * np.linalg.norm(pos[j] - pos[a]) + stray[a]
            a = int(parents[a])
        if a >= 0 and any(c not in pruned for c in np.flatnonzero(parents == a)):
            out[j] = (a, total)
    return out


def _bvh_only(dataset: str, why: str) -> None:
    if _config(dataset).data.source.type != "bvh_files":
        pytest.skip(f"{dataset}'s source is not raw BVH files: {why}")


def _angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = quat_mul(quat_inverse(a), b)
    return np.degrees(2 * np.arccos(np.clip(np.abs(d[..., 3]), 0.0, 1.0)))


def _assert_same_world(got, want, tol: float, what: str, skip=()) -> None:
    """World positions agree to `tol` of the rig's extent; the message names the joint.

    Joints named in `skip` are not compared. The same joints in another order
    compare by name (re-applying `bfs_order` puts them in BFS order again)."""
    assert sorted(got.names) == sorted(want.names), f"{what}: joints differ"
    a, b = got.global_positions()[:, [got.names.index(n) for n in want.names]], want.global_positions()
    err = np.linalg.norm(a - b, axis=-1).max(axis=0) / np.ptp(b[0], axis=0).max()
    err[[j for j, name in enumerate(want.names) if name in skip]] = 0.0
    worst = int(err.argmax())
    assert err[worst] <= tol, f"{what}: `{want.names[worst]}` off by {err[worst]:.2e} of extent"


def test_round_trip(clip: Path, tmp_path, request):
    dataset = request.config.getoption("--dataset")
    corpus, raw = Path("data") / dataset, Path("data/_raw") / dataset
    rig, action = clip.parent.name, clip.stem
    record = _record(corpus, raw, rig)
    manifest_rig = SkeletonManifest.load(corpus / "rigs" / rig / "manifest.yaml")
    clip_source = _raw_source(dataset, manifest_rig, action, record.clip_params[action].get("frames", {}).get("window"))
    source = clip_source.anim

    # The model's view, read the way training reads it (data/dataset.py).
    prepared = RigidBodyAnimation.load(clip).as_rigid_body(joint_translation="drop")
    model = apply_reduction(prepared, record.reduction)

    # Reduction leg: expanding the model view puts every joint back where the
    # prepared clip had it. World-exact, not rotation-exact: a folded joint's
    # rotation stays on its parent, and the two are coincident.
    # A still joint comes back at its rig's constant rotation, which a clip
    # holds only to the dataset's `drop_still` degrees; it is checked to that.
    config = _config(dataset)
    manifest = SkeletonManifest.load(corpus / "rigs" / rig / "manifest.yaml", contact_defaults(config.data.labels.contact))
    with np.load(corpus / "rigs" / rig / "skeleton.npz", allow_pickle=True) as data:
        still = load_still(data)
    expanded = invert_reduction(model, record.reduction)
    _assert_same_world(expanded, prepared, EXACT_TOL, "reduction leg", skip=still)
    if still:
        held = [prepared.names.index(n) for n in still]
        turned = _angle_deg(prepared.rotations[:, held], expanded.rotations[:, held]).max()
        degrees = next(r.degrees for r in record.reduction_rules if r.name == "drop_still")
        assert turned <= degrees, f"reduction leg: a still joint turned {turned:.3f} deg"
    # A fold loses its parent's own world rotation, so it may only target the root.
    folded = [op.parent for op in record.reduction.ops
              if op.kind == COLLAPSE and prepared.parents[prepared.names.index(op.parent)] >= 0]
    assert not folded, f"reduction leg: folded into non-root {folded[:3]}, losing its rotation"

    # Features leg: extracting and rebuilding (the model as identity) gives the
    # model view back. Features are root-invariant in XZ and a velocity block
    # costs the last frame; both are by design, so they are factored out.
    # Both rotation layouts are checked, the dataset's and the other one, so
    # switching `schema:` can't break recovery unnoticed.
    schema = list(config.features.blocks)
    swap = {"rot6d_joint": ["rot6d_bone"], "rot6d_bone": ["rot6d_joint", "heading"], "heading": []}
    for names in (schema, [m for n in schema for m in swap.get(n, [n])]):
        layout = reduction_layout(names)
        view = apply_reduction(prepared, _record(corpus, raw, rig, layout).reduction)
        features, spec = extract_features(view, resolve(manifest, view.names), names)
        _, rebuilt = reconstruct("fk", features, spec, view)
        want = view.global_positions()[: len(features)]
        got = rebuilt.global_positions()
        scale = np.ptp(want[0], axis=0).max()
        shift = want[0, 0] - got[0, 0]
        assert abs(shift[1]) <= POS_TOL * scale, f"features leg ({layout}): root height did not come back"
        shift[1] = 0.0
        err = np.abs(got + shift - want).max() / scale
        assert err <= POS_TOL, f"features leg ({layout}): joints off by {err:.2e} of extent"
        # Every rotation that orients something comes back: under `rot6d_joint`
        # every joint's, leaves included; under `rot6d_bone` every joint with a
        # child (its reduction keeps the leaf that carries a parent's rotation,
        # so the only leaves left are zero-length and orient nothing).
        if layout == "rot6d_bone":
            made = [n for j, n in enumerate(view.names) if j not in view.parents
                    and prepared.names.index(n) in prepared.parents and n not in still]
            assert not made, f"features leg ({layout}): reduction made {made[:3]} leaves, losing their rotation"
        else:
            # Only End Sites and still joints may drop: any other joint has a
            # rotation of its own.
            dropped = [op.name for op in _record(corpus, raw, rig, layout).reduction.ops
                       if op.kind == DROP and prepared.names.index(op.name) in prepared.parents
                       and op.name not in still]
            assert not dropped, f"features leg ({layout}): reduction dropped joints {dropped[:3]}, losing their rotation"
        held = [j for j in range(view.n_joints) if layout == "rot6d_joint" or j in view.parents]
        turn = quat_mul(quat_inverse(view.rotations[: len(features), held]), rebuilt.rotations[:, held])
        angle = np.degrees(2 * np.arccos(np.clip(np.abs(turn[..., 3]), 0.0, 1.0))).max()
        assert angle <= ROT_TOL_DEG, f"features leg ({layout}): a joint rotation off by {angle:.2e} deg"

    # Preparation leg: undo the stages one at a time with the recorded
    # parameters, and check that re-applying each reproduces its input.
    params = {name: dict(value) for name, value in record.params.items()}
    params.update(record.clip_params[action])
    unprepared = expanded
    for stage in reversed(record.chain.stages):
        undone = stage.invert(unprepared, params[stage.name])
        _assert_same_world(
            stage.apply(undone, params[stage.name]), unprepared, EXACT_TOL,
            f"preparation leg: `{stage.name}`",
        )
        unprepared = undone

    # Export leg: back onto the user's rig, through a real file. It must hold
    # what the preparation leg unprepared in memory, then match the source.
    restored = record.restore(model, facing_from=action)
    BVH.from_animation(restored, channels=record.source_channels).write(tmp_path / "out.bvh")
    back = BVH.read(tmp_path / "out.bvh")
    _assert_same_world(
        back.to_animation(), unprepared, POS_TOL,
        "export leg: the file differs from the in-memory unprepared clip "
        "(BVH writer or RigRecord)",
    )

    assert back.names == source.names, "export leg: joint names differ from the source"
    np.testing.assert_array_equal(back.parents, source.parents, "export leg: hierarchy differs")
    if clip_source.channels is not None:   # a BVH source only
        assert back.channels == clip_source.channels, "export leg: CHANNELS layout differs"
    assert back.rotations.shape == source.rotations.shape, "export leg: frame count differs"
    assert back.fps == pytest.approx(source.fps), "export leg: frame time differs"

    src_pos, src_rot = source.global_transforms()
    back_pos, back_rot = back.to_animation().global_transforms()
    extent = np.ptp(src_pos[0], axis=0).max()
    if clip_source.channels is not None:   # the OFFSET block of a BVH
        np.testing.assert_allclose(
            back.offsets, source.offsets, rtol=0, atol=POS_TOL * extent,
            err_msg="export leg: OFFSET block differs from the source",
        )

    # Joints the round trip drops by design: the locator chain above a promoted
    # root and its dead subtree, then whatever the reduction removed.
    promote = record.params.get("promote_root")   # data=unimate has none
    root = int(promote["n_removed"]) if promote else 0
    locators = set(range(root)) | ({int(i) for i in promote["dead_indices"]} if promote else set())
    # Reduction names joints on the promoted rig, so map them back by name.
    names = source.names
    ops = record.reduction.ops
    removed = {names.index(op.name) for op in ops if op.kind == DROP}
    removed |= {names.index(op.parent) for op in ops if op.kind == COLLAPSE}
    # Joints `prune_unskinned` removed: motionless at their bind pose.
    prune = record.params.get("prune_unskinned", {})
    pruned = {names.index(str(n)) for n in prune.get("removed", ())}
    removed |= pruned
    # `prune_unskinned` also takes off the roots above the real one (`kind` root),
    # the unimate counterpart of the locator chain `promote_root` strips.
    above = {names.index(str(n)) for n, kind in zip(prune.get("removed", ()), prune.get("kind", ())) if kind == "root"}
    locators |= above
    if above:   # the one joint left under them
        root = next(j for j in range(len(names)) if j not in pruned and int(source.parents[j]) in above | {-1})

    err = np.linalg.norm(back_pos[:, root] - src_pos[:, root], axis=-1).max() / extent
    assert err <= POS_TOL, f"export leg: root `{source.names[root]}` off by {err:.2e} of extent"

    # A joint's world rotation is observable only through its children.
    # (a removed child, back at its bind pose, shows nothing of it).
    has_child = np.zeros(len(names), dtype=bool)
    kept = np.array([j not in pruned for j in range(len(names))])
    has_child[source.parents[(source.parents >= 0) & kept]] = True
    checked = [
        j for j in range(len(names))
        if has_child[j] and j not in locators and j not in removed
    ]
    angles = _angle_deg(src_rot[:, checked], back_rot[:, checked]).max(axis=0)
    worst = int(angles.argmax())
    assert angles[worst] <= ROT_TOL_DEG, (
        f"export leg: `{source.names[checked[worst]]}` world rotation off by "
        f"{angles[worst]:.4f} deg"
    )

    # A removed joint is where its kept ancestors put it, within the bound its
    # removed ancestors' turns allow (`_pruned_tolerance`).
    off = []
    for j, (k, bound) in _pruned_tolerance(source, pruned).items():
        miss = np.linalg.norm((back_pos[:, j] - back_pos[:, k]) - (src_pos[:, j] - src_pos[:, k]), axis=-1).max()
        if (rig, names[j]) not in PRUNED_SKIP and miss > bound + POS_TOL * extent:
            off.append(f"`{names[j]}` {miss:.2e} from `{names[k]}`'s frame (bound {bound:.2e})")
    assert not off, f"export leg: removed joints off the source: {'; '.join(off)}"

    # The documented loss, asserted rather than assumed: bones are rigid.
    moving = np.ptp(back.translations[:, 1:], axis=0).max()
    assert moving <= POS_TOL * extent, "export leg: a non-root joint translates"


def test_up_axis(request):
    """A rig authored Z-up and declared `up_axis: +Z` prepares to the same clip
    as its Y-up original, and inverts back to its own Z-up convention.

    The corpus is all Y-up, so this turns one real rig to Z-up to exercise
    `UpAxis`.
    """
    dataset = request.config.getoption("--dataset")
    _bvh_only(dataset, "needs the rest and a clip as raw .bvh files")
    corpus, raw = Path("data") / dataset, Path("data/_raw") / dataset
    rig = sorted(p.name for p in (corpus / "rigs").iterdir() if p.is_dir())[0]
    manifest = SkeletonManifest.load(corpus / "rigs" / rig / "manifest.yaml")
    paths = sorted((raw / "source" / rig).glob("*.bvh"))
    rest_path = rest_source(manifest, paths)
    clip_path = next(p for p in paths if p != rest_path)
    rest_bvh = BVH.read(rest_path)
    chain = chain_for(_config(dataset).data.prepare, rest_bvh.channels)

    def prepare(man, rest, clip):
        rig_params = chain.fit_rig(rest, resolve(man, rest.names))
        return chain.apply(clip, resolve(man, clip.names), rig_params)

    rest, clip = rest_bvh.to_animation(), BVH.read(clip_path).to_animation()
    to_z = quat_between(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    want, want_params = prepare(manifest, rest, clip)
    got, got_params = prepare(
        replace(manifest, up_axis="+Z"), rotate_rig(rest, to_z), rotate_rig(clip, to_z)
    )
    _assert_same_world(got, want, EXACT_TOL, f"{rig}: Z-up prepared")
    _assert_same_world(
        chain.invert(got, got_params), rotate_rig(chain.invert(want, want_params), to_z),
        EXACT_TOL, f"{rig}: Z-up inverted",
    )


def test_round_trip_fbx(clip: Path, tmp_path, request):
    """The FBX ends of the loop, proved as a chain of equalities.

    1. The rig's calibration: its vendor FBX and raw BVH of one clip are the
       same motion under one similarity and a constant per-joint frame change.
       That is checked against every frame of the pair when it is fitted.
    2. `FBX.from_animation` writes exactly the restored clip onto that asset: the
       file, read back, holds the calibrated world transforms of every joint.
    3. The prepared clip and the model view, written onto the same asset in
       canonical space, read back with every joint where the prepared clip
       (for the model view: its expansion) has it.

    With `test_round_trip` showing the restored clip IS the source (up to the
    losses by design), 1 and 2 put the source back into the user's own FBX.
    """
    fbx = pytest.importorskip(
        "poseydon.io.fbx", reason="needs the fbxsdk container", exc_type=ImportError
    )
    dataset = request.config.getoption("--dataset")
    _bvh_only(dataset, "calibrates against the vendor FBX and raw BVH of one clip")
    corpus, raw = Path("data") / dataset, Path("data/_raw") / dataset
    rig, action = clip.parent.name, clip.stem
    record = _record(corpus, raw, rig)
    prepared = RigidBodyAnimation.load(clip).as_rigid_body(joint_translation="drop")
    model = apply_reduction(prepared, record.reduction)

    calibration = record.calibration("source")
    assert calibration.residual <= 1e-4, f"calibration off by {calibration.residual:.2e}"

    restored = record.restore(model, facing_from=action)
    fbx.FBX.from_animation(restored, calibration).write(tmp_path / "out.fbx")
    back = fbx.FBX.read(tmp_path / "out.fbx")
    asset = fbx.FBX.read(calibration.asset)
    assert back.fps == pytest.approx(prepared.fps), "original fbx: frame rate differs"
    assert back.names == asset.names, "original fbx: node hierarchy differs from the asset"
    np.testing.assert_array_equal(back.parents, asset.parents)

    want = calibration.drive(restored)
    world = back.world()
    first = next(iter(want.values()))
    assert len(world) == len(first), "original fbx: frame count differs"
    got = np.stack([world[:, back.names.index(n)] for n in want], axis=1)
    want = np.stack(list(want.values()), axis=1)
    extent = np.ptp(want[0, :, :3, 3], axis=0).max()
    err = np.linalg.norm(got[..., :3, 3] - want[..., :3, 3], axis=-1).max() / extent
    assert err <= FBX_TOL, f"original fbx: joints off by {err:.2e} of extent"
    # Relative to each node's own frame: nodes carry their own scale.
    size = np.linalg.norm(want[..., :3, :3], axis=-2, keepdims=True)
    err = (np.abs(got[..., :3, :3] - want[..., :3, :3]) / size).max()
    assert err <= FBX_TOL, f"original fbx: joint frames off by {err:.2e}"

    # 3. Prepared and model FBX: the same asset, mesh and all, placed in
    #    canonical space with every joint where the prepared clip has it.
    in_prepared = record.calibration("prepared")
    for name, target in (("prepared", prepared), ("model", record.expand(model))):
        fbx.FBX.from_animation(target, in_prepared).write(tmp_path / f"{name}.fbx")
        read = fbx.FBX.read(tmp_path / f"{name}.fbx")
        assert read.names == asset.names, f"{name} fbx: node hierarchy differs from the asset"
        assert read.fps == pytest.approx(target.fps), f"{name} fbx: frame rate differs"
        world = read.world()[:, [read.names.index(n) for n in in_prepared.joints]]
        want = target.global_positions()[:, [target.names.index(n) for n in in_prepared.joints]]
        err = np.linalg.norm(world[..., :3, 3] - want, axis=-1).max() / np.ptp(want[0], axis=0).max()
        assert err <= FBX_TOL, f"{name} fbx: joints off by {err:.2e} of extent"
