"""The data round trip, on every prepared clip of one dataset.

    raw source --(stage 1, on disk)--> prepared clip --reduce--> model view
        model view --expand--> prepared clip                   (reduction leg)
        model view --features--> --fk--> model view            (features leg)
        prepared --invert each stage--> unprepared             (preparation leg)
        model view --CharacterExporter.original_bvh--> file    (export leg)

Each leg is checked on its own, so a failure names the step that broke: the
reduction, one preparation stage by name, the features, or the written file.
The file is then compared against the RAW SOURCE FILE: the one reference the
pipeline did not produce.

`test_round_trip_fbx` closes the same loop in FBX, the format animators use.
It needs the Autodesk FBX SDK, which only the `fbxsdk` container has, and runs
there alongside the BVH test. One test per clip, so `-k Camel` or
`-k Camel/bite` narrows the probe:

    docker compose run --rm -T test python -m pytest -q tests/test_roundtrip.py --dataset truebones
    docker compose run --rm -T fbxsdk python -m pytest -q tests/test_roundtrip.py --dataset truebones

What is asserted is what the round trip promises (docs/guides/roundtrip.md):
structure exact, root trajectory exact, every observable world rotation exact,
bones rigid. What it loses by design is not asserted: per-joint translation
(`EnforceRigid`, one rigid geometry per rig), the locator chain above a
promoted root and its dead subtree (constant pose), and the rotation of
joints the reduction removed or folded into their parent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from poseydon.build.index import action_slug, strip_skeleton_prefix
from poseydon.core.animation import RigidBodyAnimation
from poseydon.core.rotations import quat_inverse, quat_mul
from poseydon.core.skeleton import SkeletonManifest, resolve
from poseydon.export.character import CharacterExporter
from poseydon.features import extract_features, reconstruct
from poseydon.features.reduce import COLLAPSE, DROP, apply_reduction, invert_reduction
from poseydon.io.bvh import BVH

# BVH stores six decimals, so a write/read cycle is exact only to that.
POS_TOL = 1e-5  # fraction of the rig's extent
EXACT_TOL = 1e-9  # same, for legs that never touch a file
# FBX keys are float32, and their rounding adds up along a chain: Anaconda's
# 27-joint spine reaches 1.35e-5 in its tip's frame. Any real error is far larger.
FBX_TOL = 1e-4
ROT_TOL_DEG = 1e-3


def pytest_generate_tests(metafunc):
    if "clip" not in metafunc.fixturenames:
        return
    dataset = metafunc.config.getoption("--dataset")
    clips = sorted((Path("data") / dataset / "clips").glob("*/*.npz"))
    metafunc.parametrize(
        "clip", clips, ids=[f"{p.parent.name}/{p.stem}" for p in clips]
    )


_exporters: dict[str, CharacterExporter] = {}


def _exporter(corpus: Path, raw: Path, rig: str) -> CharacterExporter:
    if rig not in _exporters:
        _exporters[rig] = CharacterExporter.load(corpus, raw, rig)
    return _exporters[rig]


def _raw_source(raw: Path, rig: str, action: str) -> Path:
    for path in sorted((raw / "source" / rig).glob("*.bvh")):
        if strip_skeleton_prefix(action_slug(path.stem), rig) == action:
            return path
    pytest.fail(f"{rig}/{action}: no raw .bvh under {raw / 'source' / rig} maps to this clip")


def _angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = quat_mul(quat_inverse(a), b)
    return np.degrees(2 * np.arccos(np.clip(np.abs(d[..., 3]), 0.0, 1.0)))


def _assert_same_world(got, want, tol: float, what: str) -> None:
    """World positions agree to `tol` of the rig's extent; the message names the joint."""
    assert got.names == want.names, f"{what}: joints differ"
    a, b = got.global_positions(), want.global_positions()
    err = np.linalg.norm(a - b, axis=-1).max(axis=0) / np.ptp(b[0], axis=0).max()
    worst = int(err.argmax())
    assert err[worst] <= tol, f"{what}: `{want.names[worst]}` off by {err[worst]:.2e} of extent"


def test_round_trip(clip: Path, tmp_path, request):
    dataset = request.config.getoption("--dataset")
    corpus, raw = Path("data") / dataset, Path("data/_raw") / dataset
    rig, action = clip.parent.name, clip.stem
    exporter = _exporter(corpus, raw, rig)
    source = BVH.read(_raw_source(raw, rig, action))

    # The model's view, read the way training reads it (data/dataset.py).
    prepared = RigidBodyAnimation.load(clip).as_rigid_body(joint_translation="drop")
    model = apply_reduction(prepared, exporter.reduction)

    # Reduction leg: expanding the model view puts every joint back where the
    # prepared clip had it. World-exact, not rotation-exact: a folded joint's
    # rotation stays on its parent, and the two are coincident.
    expanded = invert_reduction(model, exporter.reduction)
    _assert_same_world(expanded, prepared, EXACT_TOL, "reduction leg")

    # Features leg: extracting and rebuilding (the model as identity) gives the
    # model view back. Features are root-invariant in XZ and a velocity block
    # costs the last frame; both are by design, so they are factored out.
    schema = yaml.safe_load(Path(f"configs/dataset/{dataset}.yaml").read_text())["schema"]
    manifest = SkeletonManifest.load(corpus / "rigs" / rig / "manifest.yaml")
    features, spec = extract_features(model, resolve(manifest, model.names), schema)
    _, rebuilt = reconstruct("fk", features, spec, model)
    want = model.global_positions()[: len(features)]
    got = rebuilt.global_positions()
    scale = np.ptp(want[0], axis=0).max()
    shift = want[0, 0] - got[0, 0]
    assert abs(shift[1]) <= POS_TOL * scale, "features leg: root height did not come back"
    shift[1] = 0.0
    err = np.abs(got + shift - want).max() / scale
    assert err <= POS_TOL, f"features leg: joints off by {err:.2e} of extent"

    # Preparation leg: undo the stages one at a time with the recorded
    # parameters, and check that re-applying each reproduces its input.
    params = {name: dict(value) for name, value in exporter.params.items()}
    params.update(exporter.clip_params[action])
    unprepared = expanded
    for stage in reversed(exporter.chain.stages):
        undone = stage.invert(unprepared, params[stage.name])
        _assert_same_world(
            stage.apply(undone, params[stage.name]), unprepared, EXACT_TOL,
            f"preparation leg: `{stage.name}`",
        )
        unprepared = undone

    # Export leg: back onto the user's rig, through a real file. It must hold
    # what the preparation leg unprepared in memory, then match the source.
    exporter.original_bvh(model, facing_from=action).write(tmp_path / "out.bvh")
    back = BVH.read(tmp_path / "out.bvh")
    _assert_same_world(
        back.to_animation(), unprepared, POS_TOL,
        "export leg: the file differs from the in-memory unprepared clip "
        "(BVH writer or CharacterExporter)",
    )

    assert back.names == source.names, "export leg: joint names differ from the source"
    np.testing.assert_array_equal(back.parents, source.parents, "export leg: hierarchy differs")
    assert back.channels == source.channels, "export leg: CHANNELS layout differs"
    assert back.rotations.shape == source.rotations.shape, "export leg: frame count differs"
    assert back.fps == pytest.approx(source.fps), "export leg: frame time differs"

    src_pos, src_rot = source.to_animation().global_transforms()
    back_pos, back_rot = back.to_animation().global_transforms()
    extent = np.ptp(src_pos[0], axis=0).max()
    np.testing.assert_allclose(
        back.offsets, source.offsets, rtol=0, atol=POS_TOL * extent,
        err_msg="export leg: OFFSET block differs from the source",
    )

    # Joints the round trip drops by design: the locator chain above a promoted
    # root and its dead subtree, then whatever the reduction removed.
    promote = exporter.params["promote_root"]
    root = int(promote["n_removed"])
    locators = set(range(root)) | {int(i) for i in promote["dead_indices"]}
    # Reduction names joints on the promoted rig, so map them back by name.
    names = source.names
    ops = exporter.reduction.ops
    removed = {names.index(op.name) for op in ops if op.kind == DROP}
    removed |= {names.index(op.parent) for op in ops if op.kind == COLLAPSE}

    err = np.linalg.norm(back_pos[:, root] - src_pos[:, root], axis=-1).max() / extent
    assert err <= POS_TOL, f"export leg: root `{source.names[root]}` off by {err:.2e} of extent"

    # A joint's world rotation is observable only through its children.
    has_child = np.zeros(len(names), dtype=bool)
    has_child[source.parents[source.parents >= 0]] = True
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

    # The documented loss, asserted rather than assumed: bones are rigid.
    moving = np.ptp(back.translations[:, 1:], axis=0).max()
    assert moving <= POS_TOL * extent, "export leg: a non-root joint translates"


def test_round_trip_fbx(clip: Path, tmp_path, request):
    """The FBX ends of the loop, proved as a chain of equalities.

    1. The rig's calibration: its vendor FBX and raw BVH of one clip are the
       same motion under one similarity and a constant per-joint frame change.
       That is checked against every frame of the pair when it is fitted.
    2. `original_fbx` writes exactly the restored clip onto that asset: the
       file, read back, holds the calibrated world transforms of every joint.
    3. The prepared clip and the model view, written as skeleton FBX, read
       back as themselves.

    With `test_round_trip` showing the restored clip IS the source (up to the
    losses by design), 1 and 2 put the source back into the user's own FBX.
    """
    fbx = pytest.importorskip(
        "poseydon.io.fbx", reason="needs the fbxsdk container", exc_type=ImportError
    )
    dataset = request.config.getoption("--dataset")
    corpus, raw = Path("data") / dataset, Path("data/_raw") / dataset
    rig, action = clip.parent.name, clip.stem
    exporter = _exporter(corpus, raw, rig)
    prepared = RigidBodyAnimation.load(clip).as_rigid_body(joint_translation="drop")
    model = apply_reduction(prepared, exporter.reduction)

    calibration = exporter.calibration()
    assert calibration.residual <= 1e-4, f"calibration off by {calibration.residual:.2e}"

    exporter.original_fbx(model, facing_from=action).write(tmp_path / "out.fbx")
    back = fbx.FBX.read(tmp_path / "out.fbx")
    asset = fbx.FBX.read(calibration.asset)
    assert back.fps == pytest.approx(prepared.fps), "original fbx: frame rate differs"
    assert back.names == asset.names, "original fbx: node hierarchy differs from the asset"
    np.testing.assert_array_equal(back.parents, asset.parents)

    want = calibration.drive(exporter._restore(model, facing_from=action))
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

    for name, anim in (("prepared", prepared), ("model", model)):
        fbx.FBX.from_animation(anim).write(tmp_path / f"{name}.fbx")
        read = fbx.FBX.read(tmp_path / f"{name}.fbx")
        assert read.fps == pytest.approx(anim.fps), f"{name} fbx: frame rate differs"
        world = read.world()[:, [read.names.index(n) for n in anim.names]]
        target = anim.global_positions()
        err = np.linalg.norm(world[..., :3, 3] - target, axis=-1).max() / np.ptp(target[0], axis=0).max()
        assert err <= FBX_TOL, f"{name} fbx: joints off by {err:.2e} of extent"
