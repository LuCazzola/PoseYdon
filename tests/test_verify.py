"""The motion verifier (tools/verify, scripts/verify_motion.py).

The reference skin must reproduce the SDK reading of the same file, the asset must
name the vendor file it came from, and every mutation of a built round trip must turn
the verifier red on the check that owns it. Needs the FBX SDK, so it runs in the
`fbxsdk` container only.
"""

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

fbx = pytest.importorskip("poseydon.io.fbx", exc_type=ImportError)
import fbx as sdk  # noqa: E402 - the Autodesk module, present whenever poseydon.io.fbx imported
from scripts.verify_motion import build, judge  # noqa: E402 - needs the SDK too
from tools.verify.case import load_case, sample_frames  # noqa: E402
from poseydon.export import RigRecord  # noqa: E402
from tools.verify import laws  # noqa: E402
from tools.verify.report import Report, Status  # noqa: E402
from tools.verify.skin import deform, extent  # noqa: E402


def test_reference_skin_matches_the_sdk_reading():
    scene = fbx.FBX.read("data/truebones/cache/rigs/Camel/rest.fbx")
    for twist in (0.0, 0.4):  # at rest the world equals the bind, which hides the matrix order
        if twist:
            rot = np.eye(4)
            rot[:2, :2] = [[np.cos(twist), -np.sin(twist)], [np.sin(twist), np.cos(twist)]]
            scene.pose({n: scene.world(scene.frames()[:1])[0, j] @ rot for j, n in enumerate(scene.names)}, 30.0)
        posed = scene.world(scene.frames()[:1])
        world = {n: posed[:, j] for j, n in enumerate(scene.names)}
        got = np.concatenate([deform(s, world)[0] for s in scene.skin_data()])
        want = scene.skinned_vertices(0)
        assert got.shape == want.shape
        assert np.abs(got - want).max() / extent(want) < 1e-9
    with np.load("data/truebones/cache/rigs/Camel/rest.npz") as z:
        assert Path(str(z["source_fbx"])).is_file()
    scene.close()


def _amatrix(m: np.ndarray):
    """`_amatrix` of 89bff67: a column-vector 4x4 ``T @ R @ S`` as an FbxAMatrix (`fbx._matrix` inverted)."""
    scale = np.linalg.norm(m[:3, :3], axis=0)
    x, y, z, w = Rotation.from_matrix(m[:3, :3] / scale).as_quat()
    out = sdk.FbxAMatrix()
    out.SetTQS(sdk.FbxVector4(*map(float, m[:3, 3])), sdk.FbxQuaternion(x, y, z, w), sdk.FbxVector4(*map(float, scale)))
    return out


def _mutate(path: Path, edit) -> None:
    scene = fbx.FBX.read(path)
    edit(scene)
    scene.write(path)
    scene.close()


def strip_mesh(path: Path) -> None:
    def edit(scene):
        for node, kind in zip(scene.nodes, scene.node_types()):
            if kind == "eMesh":
                node.GetParent().RemoveChild(node)
                node.Destroy()

    _mutate(path, edit)


def delete_one_cluster(path: Path) -> None:
    def edit(scene):
        skin = scene._skins()[0]
        skin.RemoveCluster(skin.GetCluster(0))

    _mutate(path, edit)


def delete_skin_deformers(path: Path) -> None:
    def edit(scene):
        for skin in scene._skins():
            skin.GetGeometry().RemoveDeformer(0)

    _mutate(path, edit)


def truncate_file(path: Path) -> None:
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])


def swap_two_cluster_links(path: Path) -> None:
    def edit(scene):
        skin = scene._skins()[0]
        a, b = skin.GetCluster(0), skin.GetCluster(1)
        link_a, link_b = a.GetLink(), b.GetLink()
        a.DisconnectSrcObject(link_a)  # SetLink alone leaves the first link in place
        b.DisconnectSrcObject(link_b)
        a.SetLink(link_b)
        b.SetLink(link_a)

    _mutate(path, edit)


def perturb_one_link_bind(path: Path) -> None:
    """Move the bind of the cluster that weighs most vertices by 5% of the mesh extent."""
    def edit(scene):
        skin = scene._skins()[0]
        big = max(range(skin.GetClusterCount()), key=lambda c: skin.GetCluster(c).GetControlPointIndicesCount())
        cluster, bind = skin.GetCluster(big), sdk.FbxAMatrix()
        cluster.GetTransformLinkMatrix(bind)
        move = fbx._matrix(bind)
        move[:3, 3] += 0.05 * extent(scene.skin_data()[0].points)
        cluster.SetTransformLinkMatrix(_amatrix(move))

    _mutate(path, edit)


def nan_control_point(path: Path) -> None:
    def edit(scene):
        scene._skins()[0].GetGeometry().SetControlPointAt(sdk.FbxVector4(np.nan, np.nan, np.nan, 1.0), 0)

    _mutate(path, edit)


def add_a_control_point(path: Path) -> None:
    def edit(scene):
        geometry = scene._skins()[0].GetGeometry()
        points = [tuple(p)[:3] for p in geometry.GetControlPoints()]
        geometry.InitControlPoints(len(points) + 1)
        for i, p in enumerate(points + [(0.0, 0.0, 0.0)]):
            geometry.SetControlPointAt(sdk.FbxVector4(*p, 1.0), i)

    _mutate(path, edit)


def rename_a_cluster_link(path: Path) -> None:
    _mutate(path, lambda scene: scene._skins()[0].GetCluster(0).GetLink().SetName("renamed_joint"))


def rebind_89bff67(path: Path) -> None:
    """`FBX.rebind` as of commit 89bff67 (positions only): make the rest asset's pose the bind
    pose and the mesh its skinned self, then give the file its animation back."""
    def edit(scene):
        rest = fbx.FBX.read(f"data/truebones/cache/rigs/{path.parent.name.split('__')[0]}/rest.fbx")
        world, fps, motion = rest.world(rest.frames()[:1])[0], scene.fps, scene.world()
        new = dict(zip(rest.names, world))
        for skin in scene._skins():
            geometry = skin.GetGeometry()
            points = np.array([[p[0], p[1], p[2], 1.0] for p in geometry.GetControlPoints()])
            blend, total = np.zeros((len(points), 4, 4)), np.zeros(len(points))
            for c in range(skin.GetClusterCount()):
                cluster = skin.GetCluster(c)
                link_bind, mesh_bind = sdk.FbxAMatrix(), sdk.FbxAMatrix()
                cluster.GetTransformLinkMatrix(link_bind)
                cluster.GetTransformMatrix(mesh_bind)
                name = cluster.GetLink().GetName()
                mesh = fbx._matrix(mesh_bind)
                deform_ = np.linalg.inv(mesh) @ new[name] @ np.linalg.inv(fbx._matrix(link_bind)) @ mesh
                where = np.array(cluster.GetControlPointIndices(), dtype=int)
                weight = np.array(cluster.GetControlPointWeights(), dtype=float)
                if where.size:
                    np.add.at(blend, where, weight[:, None, None] * deform_)
                    np.add.at(total, where, weight)
                cluster.SetTransformLinkMatrix(_amatrix(new[name]))
            weighted = total > 0
            blend[weighted] /= total[weighted, None, None]
            blend[~weighted] = np.eye(4)
            for i, point in enumerate(np.einsum("vij,vj->vi", blend, points)):
                geometry.SetControlPointAt(sdk.FbxVector4(*point[:3], 1.0), i)
        scene.animate({name: motion[:, j] for j, name in enumerate(scene.names)}, fps)

    _mutate(path, edit)


def _largest_cluster_joint(scene) -> int:
    skin = scene._skins()[0]
    big = max(range(skin.GetClusterCount()), key=lambda c: skin.GetCluster(c).GetControlPointIndicesCount())
    return scene.names.index(skin.GetCluster(big).GetLink().GetName())


def offset_one_joint_rotation(path: Path) -> None:
    """Turn the joint weighing most vertices by 25 degrees about Z, every joint's position kept."""
    def edit(scene):
        world, j = scene.world(), _largest_cluster_joint(scene)
        turn = np.eye(4)
        turn[:3, :3] = Rotation.from_euler("z", 25, degrees=True).as_matrix()
        world[:, j, :3, :3] = world[:, j, :3, :3] @ turn[:3, :3]
        scene.animate({n: world[:, k] for k, n in enumerate(scene.names)}, scene.fps)

    _mutate(path, edit)


def write_with_wrong_offset(path: Path, factor: float = 1.1) -> None:
    """Stand-in for a bug in `EnforceRigid`: write the restored clip again with the longest bone 10% longer
    (``factor`` 1.0 is the same rewrite with nothing wrong, the control)."""
    from poseydon.build.pipeline import compose_build
    from poseydon.core.animation import RigidBodyAnimation
    from poseydon.features.reduce import apply_reduction

    meta = json.loads((path.parent / "case.json").read_text())
    root = Path(compose_build([f"data={meta['data']}"]).data.paths.root)
    record = RigRecord.load(root, meta["rig"])
    prepared = RigidBodyAnimation.load(root / "clips" / meta["rig"] / f"{meta['action']}.npz").as_rigid_body(joint_translation="drop")
    restored = record.restore(apply_reduction(prepared, record.reduction), facing_from=meta["action"])
    j = int(np.linalg.norm(restored.offsets, axis=-1).argmax())
    offsets, translations = restored.offsets.copy(), restored.translations.copy()
    offsets[j] *= factor
    translations[:, j] = offsets[j] if j else translations[:, j]
    wrong = type(restored)(restored.rotations, translations, offsets, restored.parents, restored.names, restored.fps)
    fbx.FBX.from_animation(wrong, record.asset, fixed=record.fixed).write(path, embed_textures=True)


def write_placement_yawed_20deg(path: Path) -> None:
    """The canonical stage files written 20 degrees off the placement the record gives (a wrong record is C2b's catch, see `wrong_facing`)."""
    turn = np.eye(4)
    turn[:3, :3] = Rotation.from_euler("y", 20, degrees=True).as_matrix()
    write_placement(path, turn)


def write_placement_scaled_5(path: Path) -> None:
    """The canonical stage files written 5% larger than the placement the record gives."""
    write_placement(path, np.diag([1.05, 1.05, 1.05, 1.0]))


def turn_roundtrip_20deg(path: Path) -> None:
    """Stand-in for a wrong recorded facing: the restored clip turned 20 degrees about the vertical through its root at frame 0."""
    def edit(scene):
        world = scene.world()
        pivot = world[0, 0, :3, 3].copy()
        turn = np.eye(4)
        turn[:3, :3] = Rotation.from_euler("y", 20, degrees=True).as_matrix()
        turn[:3, 3] = pivot - turn[:3, :3] @ pivot
        scene.animate({n: turn @ world[:, k] for k, n in enumerate(scene.names)}, scene.fps)

    _mutate(path, edit)


def drift_yaw_over_the_clip(path: Path) -> None:
    """Turn the canonical stage files about +Y by 0 to 20 degrees across the clip: one similarity cannot absorb it, a per-frame fit would."""
    def edit(scene):
        world = scene.world()
        turn = np.tile(np.eye(4), (len(world), 1, 1))
        turn[:, :3, :3] = Rotation.from_euler("y", np.linspace(0, 20, len(world))[:, None], degrees=True).as_matrix()
        scene.animate({n: turn @ world[:, k] for k, n in enumerate(scene.names)}, scene.fps)

    for name in ("2_prepared", "3_model"):
        _mutate(path.parent / f"{name}.fbx", edit)


def write_placement(path: Path, edit: np.ndarray) -> None:
    """Rewrite 2_prepared.fbx and 3_model.fbx as `scripts/build_stage_samples.py` does, with ``edit @ space`` as the placement."""
    from poseydon.build.pipeline import compose_build
    from poseydon.core.animation import RigidBodyAnimation
    from poseydon.features.reduce import apply_reduction

    meta = json.loads((path.parent / "case.json").read_text())
    root = Path(compose_build([f"data={meta['data']}"]).data.paths.root)
    record = RigRecord.load(root, meta["rig"])
    prepared = RigidBodyAnimation.load(root / "clips" / meta["rig"] / f"{meta['action']}.npz").as_rigid_body(joint_translation="drop")
    for name, anim in (("2_prepared", prepared), ("3_model", apply_reduction(prepared, record.reduction))):
        placed, space = record.placement(anim, facing_from=meta["action"])
        fbx.FBX.from_animation(placed, record.asset, edit @ space, record.fixed).write(path.parent / f"{name}.fbx", embed_textures=True)


def shift_zero_joint(path: Path) -> None:
    """The zero file with the joint weighing most vertices moved by 5% of the mesh extent (the output, not
    `rest.npz`: a wrong rest pose written into the cache is what C4-joints is for)."""
    def edit(scene):
        world, j = scene.world(), _largest_cluster_joint(scene)
        world[:, j, 0, 3] += 0.05 * extent(scene.skin_data()[0].points)
        scene.animate({n: world[:, k] for k, n in enumerate(scene.names)}, scene.fps)

    _mutate(path.parent / "5_zero.fbx", edit)


def turn_zero_half_degree(path: Path) -> None:
    """The zero file turned 0.5 degrees about +Y: one similarity absorbs it (C4-mesh, C4-joints stay green),
    the frame's definition (facing +Z) does not."""
    def edit(scene):
        world = scene.world()
        turn = np.eye(4)
        turn[:3, :3] = Rotation.from_euler("y", 0.5, degrees=True).as_matrix()
        scene.animate({n: turn @ world[:, k] for k, n in enumerate(scene.names)}, scene.fps)

    _mutate(path.parent / "5_zero.fbx", edit)


def drop_enforce_rigid_law(path: Path) -> None:
    """The stage stays in the file's record, its law leaves the verifier (restored by the `laws` fixture)."""
    laws.LAWS.pop("enforce_rigid")
    _mutate(path, lambda scene: None)


def scale_one_rest_bone(path: Path) -> None:
    """The rest input the laws read (`RigRecord.asset`) with its longest bone 50% longer, the file under test
    untouched but for a rewrite."""
    from poseydon.build.pipeline import compose_build

    meta = json.loads((path.parent / "case.json").read_text())
    root = Path(compose_build([f"data={meta['data']}"]).data.paths.root)
    scene = fbx.FBX.read(RigRecord.load(root, meta["rig"]).asset)
    world = scene.world(scene.frames()[:1])
    reach = np.linalg.norm(world[0, 1:, :3, 3] - world[0, scene.parents[1:], :3, 3], axis=-1)
    j = 1 + int(reach.argmax())
    world[0, j, :3, 3] = world[0, scene.parents[j], :3, 3] + 1.5 * (world[0, j, :3, 3] - world[0, scene.parents[j], :3, 3])
    scene.animate({n: world[:, k] for k, n in enumerate(scene.names)}, scene.fps)
    scene.write(path.parent / "rest_scaled.fbx")
    scene.close()
    _MP.setattr(RigRecord, "asset", property(lambda self: path.parent / "rest_scaled.fbx"))
    _mutate(path, lambda scene: None)


_MP = None


@pytest.fixture(autouse=True)
def keep_laws(monkeypatch):
    global _MP
    _MP, saved = monkeypatch, dict(laws.LAWS)
    yield
    laws.LAWS.clear()
    laws.LAWS.update(saved)


@dataclass
class Mutation:
    id: str
    rig: str
    clip: str | None
    mutate: Callable[[Path], None]
    expected: dict[str, str]
    rise: float = 0.0
    base_red: bool = False  # the clip fails on the unmutated run already
    names: str = ""  # a word the C2a line must carry (the stage or joint it names)
    keeps_joints: bool = False  # the edit leaves every node's world as it was
    bounds: tuple[float, float] | None = None  # the C2a value the edit must land in
    control: Callable[[Path], None] | None = None  # the same edit with nothing wrong: must stay green


MUTATIONS = [
    Mutation("strip_mesh", "Goat", None, strip_mesh, {"C1": "FAIL"}),
    Mutation("drop_cluster", "Goat", None, delete_one_cluster, {"C1": "FAIL"}),
    Mutation("no_skin", "Goat", None, delete_skin_deformers, {"C1": "FAIL", "C2a": "NOT RUN"}),
    Mutation("truncated", "Goat", None, truncate_file, {"C1": "FAIL", "C2a": "NOT RUN"}, names="4_roundtrip"),
    Mutation("nan_point", "Goat", None, nan_control_point, {"C2a": "FAIL"}),
    Mutation("topology", "Goat", None, add_a_control_point, {"C2a": "FAIL"}),
    Mutation("missing_link", "Goat", None, rename_a_cluster_link, {"C2a": "FAIL"}, names="Bip01_Pelvis"),
    Mutation("swap_cluster_link", "Goat", None, swap_two_cluster_links, {"C2a": "FAIL"}, keeps_joints=True),
    Mutation("perturb_bind", "Goat", None, perturb_one_link_bind, {"C2a": "FAIL"}, keeps_joints=True),
    Mutation("rebind", "Goat", None, rebind_89bff67, {"C2a": "FAIL"}, bounds=(2e-2, 1e-1)),
    Mutation("rotate_joint", "Goat", None, offset_one_joint_rotation, {"C2b-untraced": "FAIL", "C2a": "PASS"}),
    Mutation("rigid_wrong_offset", "Goat", None, write_with_wrong_offset, {"C2b-untraced": "FAIL"},
             control=lambda path: write_with_wrong_offset(path, 1.0)),
    Mutation("rest_bone_scaled", "Goat", None, scale_one_rest_bone, {"C2b-rest": "FAIL"}),
    Mutation("missing_law", "Goat", None, drop_enforce_rigid_law, {"C2b-untraced": "NOT RUN", "C2b-rest": "PASS"}),
    Mutation("wrong_yaw", "Goat", None, write_placement_yawed_20deg, {"C3-placement": "FAIL", "C3-shape": "PASS"}),
    Mutation("wrong_facing", "Goat", None, turn_roundtrip_20deg, {"C2b-untraced": "FAIL"}),
    Mutation("drifting_yaw", "Goat", None, drift_yaw_over_the_clip, {"C3-shape": "FAIL"}),
    Mutation("shift_rest", "Goat", None, shift_zero_joint, {"C4-mesh": "FAIL", "C4-joints": "FAIL"}),
    Mutation("turn_zero", "Goat", None, turn_zero_half_degree, {"C4-pose": "FAIL", "C4-mesh": "PASS", "C4-joints": "PASS"}),
    Mutation("wrong_scale", "Crow", None, write_placement_scaled_5, {"C3-placement": "FAIL", "C3-shape": "PASS"}),
]


def _judged_lines(report):
    """Every line but `C4-symmetry` NOT RUN, which stays until the owner sets limits (`configs/verify/limits.yaml`)."""
    return [c for c in report.checks if (c.id, c.status) != ("C4-symmetry", Status.NOT_RUN)]


def assert_green(report):
    left = [c for c in _judged_lines(report) if c.status is not Status.PASS]
    assert not left, left


def verdict(report):
    return Report(report.run, _judged_lines(report)).verdict


def _status(report, check):
    """The worst line of ``check`` (one per stage file for C2a)."""
    order = [Status.PASS, Status.ALLOWED, Status.NOT_RUN, Status.FAIL]
    assert any(c.id == check for c in report.checks), f"no check line `{check}` in the report"
    return max((c for c in report.checks if c.id == check), key=lambda c: (order.index(c.status), c.value or 0))


def assert_red(base, after, check, want, rise):
    was, now = _status(base, check), _status(after, check)
    if was.status.value == want and want != "PASS":  # already red on the unmutated run: the evidence must grow
        assert rise > 0 and now.value >= was.value + rise, f"{check}: {was.value} -> {now.value}"
    else:
        assert now.status.value == want, f"{check}: {now.status.value} {now.note}"


def _world(path: Path) -> np.ndarray:
    scene = fbx.FBX.read(path)
    try:
        return scene.world()
    finally:
        scene.close()


@pytest.mark.parametrize("m", MUTATIONS, ids=lambda m: m.id)
def test_verifier_goes_red(m, tmp_path):
    run = build("truebones", m.rig, m.clip, tmp_path)
    base = judge(run)
    if not m.base_red:
        assert_green(base)
    before = _world(run / "4_roundtrip.fbx") if m.keeps_joints else None
    if m.control:
        m.control(run / "4_roundtrip.fbx")
        assert _status(judge(run, accept_changed=True), "C2b-untraced").status is Status.PASS
    m.mutate(run / "4_roundtrip.fbx")
    if before is not None:
        assert np.abs(before - _world(run / "4_roundtrip.fbx")).max() < 1e-4
    with pytest.raises(ValueError, match="changed"):
        judge(run)
    after = judge(run, accept_changed=True)
    for check, want in m.expected.items():
        assert_red(base, after, check, want, m.rise)
    assert verdict(after) is not Status.PASS
    assert m.names in _status(after, "C2a").note
    if m.bounds:
        assert m.bounds[0] <= _status(after, "C2a").value <= m.bounds[1]


def test_judge_refuses_what_it_cannot_read(tmp_path, monkeypatch):
    from scripts import verify_motion

    for argv in (["judge", str(tmp_path)], ["judge", str(tmp_path / "missing")]):
        monkeypatch.setattr("sys.argv", ["verify_motion", *argv])
        with pytest.raises(SystemExit) as stop:
            verify_motion.main()
        assert stop.value.code not in (0, None)
    # a truncated file against a truncated reference is two unreadable files, not two equal empty ones
    rest = Path("data/truebones/cache/rigs/Camel/rest.fbx")
    for name in ("a.fbx", "b.fbx"):
        (tmp_path / name).write_bytes(rest.read_bytes()[: rest.stat().st_size // 2])
    run = verify_motion.file_case(tmp_path / "a.fbx", tmp_path / "b.fbx", tmp_path)
    assert judge(run).verdict is Status.FAIL
    # `file` on a readable skinned FBX samples its own frames and reports a C2a value
    run = verify_motion.file_case(rest, rest, tmp_path)
    assert _status(judge(run), "C2a").value is not None


def test_source_frames_follow_the_recorded_window(tmp_path):
    """Scorpion2's clip is trimmed by a head frame (window (1, 91, 1) over keys 0-90), so source
    frame i+1 is round-trip frame i; Bear keeps all 24 keys though the vendor's time span ends
    one early, so its C1 frame range is green (C1 compares frame counts against the window). A one-frame clip cannot be built
    (`--clip tpose` writes a 1-key take under the asset's 31-frame span), so `sample_frames`
    is driven on a one-frame array directly."""
    run = build("unimate", "Scorpion2", None, tmp_path)
    case = load_case(run)
    assert RigRecord.load(Path("data/unimate"), "Scorpion2").clip_params[case.action]["frames"]["window"].tolist() == [1, 91, 1]
    assert list(case.source_frames) == [int(f) + 1 for f in case.frames]
    assert _status(judge(run), "C2a").status is Status.PASS
    assert _status(judge(run), "C1").status is Status.PASS  # the head trim is the window, not a frame-range change
    assert _status(judge(run), "C2b-untraced").status is Status.PASS  # prune_unskinned and the bind rest pose
    bear = build("truebones", "Bear", None, tmp_path)
    assert list(load_case(bear).source_frames) == list(load_case(bear).frames)
    assert _status(judge(bear), "C1").status is Status.PASS
    assert list(sample_frames(np.tile(np.eye(4), (1, 3, 1, 1)), 8)) == [0]


def test_laws_import_nothing_from_the_pipeline():
    tree = ast.parse(Path("tools/verify/laws.py").read_text())
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not {m for m in imported if m.startswith(("poseydon.build", "poseydon.export", "poseydon.features"))}


@pytest.mark.parametrize("data, rig", [("truebones", "Goat"), ("truebones", "Dog")])
def test_laws_hold_on_a_folded_and_a_still_rig(data, rig, tmp_path):
    """Goat folds `Bip01_Pelvis` into its root (the fold is seen only through joint rotations), Dog drops still
    joints; C2b reports vertices and joints apart, and a floor run is judged against the source itself."""
    run = build(data, rig, None, tmp_path)
    params = laws_params(load_case(run))
    assert params["fold_into_root"]["names"] if rig == "Goat" else params["drop_still"]["names"]
    report = judge(run)
    assert_green(report)
    where = _status(report, "C2b-untraced").where
    assert {"vertex", "joint", "rotation"} <= where.keys()
    assert {c.status for c in report.checks if c.id.startswith("C3")} == {Status.PASS}


def laws_params(case):
    from tools.verify.case import loss_record

    return loss_record(case).params


def test_a_floor_run_is_judged_against_the_source_itself(tmp_path):
    run = build("truebones", "Camel", None, tmp_path, floor=True)
    line = _status(judge(run), "C2b-untraced")
    assert line.status is Status.PASS and 0 < line.value < 1e-4, line
