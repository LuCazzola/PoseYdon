"""The motion verifier (tools/verify, scripts/verify_motion.py).

The reference skin must reproduce the SDK reading of the same file, the asset must
name the vendor file it came from, and every mutation of a built round trip must turn
the verifier red on the check that owns it. Needs the FBX SDK, so it runs in the
`fbxsdk` container only.
"""

import ast
import json
import re
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
from tools.verify import checks, laws  # noqa: E402
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


def zero_on_wrong_rest(path: Path) -> None:
    """A rest cache whose pose is wrong, with the corpus built from it: one bone's subtree turned 5 degrees in a
    COPY of `rest.npz` (the real cache is untouched; `rig_cache` is pointed at the copy) and the zero file turned
    the same way. File and reference agree, so C4-mesh stays green; only the rest-pose BVH (C4-joints) can tell."""
    from poseydon.build.prepare import similarity

    zero = path.parent / "5_zero.fbx"
    cache = path.parent / "wrong_cache"
    cache.mkdir(exist_ok=True)
    with np.load(f"data/truebones/cache/rigs/{path.parent.name.split('__')[0]}/rest.npz", allow_pickle=True) as z:
        kept = {k: z[k] for k in z.files}
    scene = fbx.FBX.read(zero)
    names, world = scene.names, scene.world()[:1]
    here = [names.index(str(n)) for n in kept["names"]]
    scale, rotation, shift = similarity(kept["rest"][:, :3, 3], world[0, here, :3, 3])
    place = np.eye(4)
    place[:3, :3], place[:3, 3] = scale * rotation, shift
    j = _largest_cluster_joint(scene)
    turn, pivot = Rotation.from_euler("z", 5, degrees=True).as_matrix(), world[0, j, :3, 3].copy()
    move = np.eye(4)
    move[:3, :3], move[:3, 3] = turn, pivot - turn @ pivot
    for k in range(len(names)):
        chain, a = {k}, k
        while scene.parents[a] >= 0:
            a = int(scene.parents[a])
            chain.add(a)
        if j in chain:
            world[0, k] = move @ world[0, k]
    scene.animate({n: world[:, k] for k, n in enumerate(names)}, scene.fps)
    scene.write(zero)
    scene.close()
    kept["rest"] = np.linalg.inv(place) @ world[0, here]
    np.savez(cache / "rest.npz", **kept)
    _MP.setattr("poseydon.build.sources.rig_cache", lambda root, rig: cache)


def turn_zero_half_degree(path: Path) -> None:
    """The zero file turned 0.5 degrees about +Y: one similarity absorbs it (C4-mesh, C4-joints stay green),
    the frame's definition (facing +Z) does not."""
    def edit(scene):
        world = scene.world()
        turn = np.eye(4)
        turn[:3, :3] = Rotation.from_euler("y", 0.5, degrees=True).as_matrix()
        scene.animate({n: turn @ world[:, k] for k, n in enumerate(scene.names)}, scene.fps)

    _mutate(path.parent / "5_zero.fbx", edit)


def turn_a_limb_180deg(path: Path) -> None:
    """The subtree of the biggest-weighing joint that is less than half the rig turned 180 degrees about
    its own origin, as if a generated clip had flipped a limb: the mesh tears at the joint."""
    def edit(scene):
        world, skin = scene.world(), scene._skins()[0]
        order = sorted(range(skin.GetClusterCount()), key=lambda c: -skin.GetCluster(c).GetControlPointIndicesCount())
        for c in order:
            j = scene.names.index(skin.GetCluster(c).GetLink().GetName())
            under = [k for k in range(len(scene.names)) if j in _chain(scene.parents, k)]
            if scene.parents[j] >= 0 and len(under) <= len(scene.names) // 2:
                break
        turn, pivot = Rotation.from_euler("x", 180, degrees=True).as_matrix(), world[:, j, :3, 3].copy()
        for k in under:
            world[:, k, :3, :3] = turn @ world[:, k, :3, :3]
            world[:, k, :3, 3] = pivot + (turn @ (world[:, k, :3, 3] - pivot).T).T
        scene.animate({n: world[:, k] for k, n in enumerate(scene.names)}, scene.fps)

    _mutate(path, edit)


def fling_first_frame(path: Path) -> None:
    """The subtree of a mid-sized joint thrown out by 5 rig extents on frame 0 only: a clip that starts exploded.
    Normalised by its own first frame (the old yardstick) the range would not notice."""
    def edit(scene):
        world = scene.world()
        j = len(scene.names) // 2
        under = [k for k in range(len(scene.names)) if j in _chain(scene.parents, k)]
        world[0, under, 0, 3] += 5 * extent(world[0, :, :3, 3])
        scene.animate({n: world[:, k] for k, n in enumerate(scene.names)}, scene.fps)

    _mutate(path, edit)


def twist_calf_30deg(path: Path) -> None:
    """The left calf and everything under it turned 30 degrees about the calf's local X, which is the bone axis of a
    Biped calf: a TWIST, not a bend, frame by frame. The global mesh lines do not move for it. A bend that stays inside the
    vendor's range is out of C5-joints' reach (tooling.md)."""
    def edit(scene):
        world = scene.world()
        j = next(k for k, n in enumerate(scene.names) if "Bip01_L_Calf" in n)
        under = [k for k in range(len(scene.names)) if j in _chain(scene.parents, k)]
        for f in range(len(world)):
            pivot = world[f, j, :3, 3].copy()
            axis = world[f, j, :3, 0] / np.linalg.norm(world[f, j, :3, 0])
            turn = Rotation.from_rotvec(np.radians(30) * axis).as_matrix()
            for k in under:
                world[f, k, :3, :3] = turn @ world[f, k, :3, :3]
                world[f, k, :3, 3] = pivot + turn @ (world[f, k, :3, 3] - pivot)
        scene.animate({n: world[:, k] for k, n in enumerate(scene.names)}, scene.fps)

    _mutate(path, edit)


def _chain(parents, k: int) -> list[int]:
    out = [k]
    while parents[out[-1]] >= 0:
        out.append(int(parents[out[-1]]))
    return out


def write_nan_rotation(path: Path) -> None:
    """One rotation key of the first animated joint is NaN, as a diverged sampler would leave it
    (`FBX.animate` cannot write one, so the curve is edited)."""
    def edit(scene):
        layer = scene._stack().GetMember(sdk.FbxCriteria.ObjectType(sdk.FbxAnimLayer.ClassId), 0)
        curve = next(c for n in scene.nodes if (c := n.LclRotation.GetCurve(layer, "X")) is not None)
        curve.KeyModifyBegin()
        curve.KeySetValue(curve.KeyGetCount() // 2, float("nan"))
        curve.KeyModifyEnd()

    _mutate(path, edit)


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
    judged: tuple[str, str] | None = None  # the (joint, part) the C5-joints line must name
    stability_min: float = 0.0  # how far above C5's `stability_tol` (1e-5) the edit must land
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
    Mutation("rotate_joint", "Goat", None, offset_one_joint_rotation, {"C2b-untraced": "FAIL", "C2a": "PASS", "C5-stability": "FAIL"}, stability_min=1e-4),
    Mutation("rigid_wrong_offset", "Goat", None, write_with_wrong_offset, {"C2b-untraced": "FAIL"},
             control=lambda path: write_with_wrong_offset(path, 1.0)),
    Mutation("rest_bone_scaled", "Goat", None, scale_one_rest_bone, {"C2b-rest": "FAIL"}),
    Mutation("missing_law", "Goat", None, drop_enforce_rigid_law, {"C2b-untraced": "NOT RUN", "C2b-rest": "PASS"}),
    Mutation("wrong_yaw", "Goat", None, write_placement_yawed_20deg, {"C3-placement": "FAIL", "C3-shape": "PASS"}),
    Mutation("wrong_facing", "Goat", None, turn_roundtrip_20deg, {"C2b-untraced": "FAIL"}),
    Mutation("drifting_yaw", "Goat", None, drift_yaw_over_the_clip, {"C3-shape": "FAIL"}),
    Mutation("shift_rest", "Goat", None, shift_zero_joint, {"C4-mesh": "FAIL", "C4-joints": "FAIL"}),
    Mutation("wrong_rest_cache", "Goat", None, zero_on_wrong_rest, {"C4-joints": "FAIL", "C4-mesh": "PASS"}),
    Mutation("turn_zero", "Goat", None, turn_zero_half_degree, {"C4-pose": "FAIL", "C4-placement": "FAIL", "C4-mesh": "PASS", "C4-joints": "PASS"}),
    Mutation("flip_limb", "Goat", None, turn_a_limb_180deg, {"C5-stretch": "FAIL"}),
    Mutation("twist_30", "Goat", None, twist_calf_30deg, {"C5-joints": "FAIL", "C5-stretch": "PASS"}, judged=("Bip01_L_Calf", "twist")),
    Mutation("fling_frame0", "Goat", None, fling_first_frame, {"C5-range": "FAIL"}),
    Mutation("nan_key", "Goat", None, write_nan_rotation, {"C5-range": "FAIL"}),
    Mutation("wrong_scale", "Crow", None, write_placement_scaled_5, {"C3-placement": "FAIL", "C3-shape": "PASS"}),
]


def _judged_lines(report):
    """Every line but `C6` NOT RUN: Blender cannot run inside `fbxsdk`, so the real C6 run is `scripts/verify_motion.sh`."""
    return [c for c in report.checks if (c.id, c.status) != ("C6", Status.NOT_RUN)]


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
    drawn = m.id == "wrong_yaw"  # sheets are slow under QEMU: one row draws them
    base = judge(run, sheets=drawn)
    if not m.base_red:
        assert_green(base)
    if drawn:
        sheets = base.run["sheets"]
        assert {"3_model-vs-expected_first", "4_roundtrip-vs-expected_first", "5_zero-vs-rest_first"} <= {Path(x).stem for x in sheets}  # placement, not only skin
        assert len(sheets) >= 8 and not any(c.id == "C7-sheets" for c in base.checks)
        assert all((run / s).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" for s in sheets)
        first = [(run / s).read_bytes() for s in sheets]
        again = judge(run)  # the same files give the same bytes
        assert first == [(run / s).read_bytes() for s in again.run["sheets"]]
    before = _world(run / "4_roundtrip.fbx") if m.keeps_joints else None
    if m.control:
        m.control(run / "4_roundtrip.fbx")
        assert _status(judge(run, accept_changed=True, sheets=False), "C2b-untraced").status is Status.PASS
    m.mutate(run / "4_roundtrip.fbx")
    if before is not None:
        assert np.abs(before - _world(run / "4_roundtrip.fbx")).max() < 1e-4
    with pytest.raises(ValueError, match="changed"):
        judge(run)
    after = judge(run, accept_changed=True, sheets=False)
    assert not after.run["sheets"] and not any(c.id == "C7-sheets" for c in after.checks)
    for check, want in m.expected.items():
        assert_red(base, after, check, want, m.rise)
    assert verdict(after) is not Status.PASS
    assert m.names in _status(after, "C2a").note
    assert (_status(after, "C5-stability").value or 0) >= m.stability_min
    if m.judged:
        assert (_status(after, "C5-joints").where["joint"], _status(after, "C5-joints").where["part"]) == m.judged
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
    # `file --rig` (the generated-motion path): C5 on the file, stability NOT RUN until a clip is passed
    report = judge(verify_motion.file_case(rest, None, tmp_path, "truebones", "Camel"))
    assert _status(report, "C5-stretch").value is not None and _status(report, "C5-stability").status is Status.NOT_RUN
    assert _status(report, "C6").status is Status.NOT_RUN and report.verdict is not Status.PASS  # no Blender output yet
    assert "C2b" in report.run["not_applicable"]
    assert np.isnan(checks._worst_ratio(np.zeros(0), np.zeros(0), 0.0))  # nothing to measure is NaN (NOT RUN), not 1.0
    # `--clip`: the generator's clip is the stability reference; another clip is not what the file should be
    goat = build("truebones", "Goat", None, tmp_path)
    clips = Path("data/truebones/clips/Goat")
    same = verify_motion.file_case(goat / "4_roundtrip.fbx", None, tmp_path, "truebones", "Goat", clips / f"{load_case(goat).action}.npz")
    other = verify_motion.file_case(goat / "4_roundtrip.fbx", None, tmp_path / "o", "truebones", "Goat", clips / "bleat.npz")
    assert _status(judge(same), "C5-stability").status is Status.PASS
    assert _status(judge(other), "C5-stability").status is Status.FAIL
    # C6 on a stand-in for Blender: the SDK's own deformed vertices in blender_read.py's format, in metres; the unit factor passes, a 2.5x scale fails
    for scale, want in ((1.0, Status.PASS), (2.5, Status.FAIL)):
        _fake_blender(goat, scale)
        c6 = [c for c in judge(goat, sheets=False).checks if c.id == "C6"]
        assert c6 and {c.status for c in c6} == {want} and all(c.value is not None for c in c6), c6


def _fake_blender(run: Path, scale: float) -> None:
    """`blender_read.py`'s npz per file, from the SDK's deformed vertices times the file's unit factor (cm -> m) times ``scale``."""
    case = load_case(run)
    (run / "blender").mkdir(exist_ok=True)
    for name, path in case.files.items():
        scene = fbx.FBX.read(path)
        try:
            frames = scene.frames()[np.zeros(1, dtype=int) if name == "5_zero" else case.frames]
            world, unit = scene.world(frames), scene.inventory().unit_cm / 100
            joints = {n: world[:, j] for j, n in enumerate(scene.names)}
            meshes = {s.mesh: deform(s, joints) * unit * scale for s in scene.skin_data()}
        finally:
            scene.close()
        np.savez(run / "blender" / f"{name}.npz", names=np.array(list(meshes)), **meshes)


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


def turn_hind_leg_10deg(path: Path) -> None:
    """The left thigh and everything under it turned 10 degrees about world X in the zero file."""
    def edit(scene):
        world = scene.world()
        j = next(k for k, n in enumerate(scene.names) if "Bip01_L_Thigh" in n)
        turn = Rotation.from_rotvec(np.radians(10) * np.array([1.0, 0, 0])).as_matrix()
        for k in (k for k in range(len(scene.names)) if j in _chain(scene.parents, k)):
            world[:, k, :3, :3] = turn @ world[:, k, :3, :3]
            world[:, k, :3, 3] = world[:, j, :3, 3] + (turn @ (world[:, k, :3, 3] - world[:, j, :3, 3]).T).T
        scene.animate({n: world[:, k] for k, n in enumerate(scene.names)}, scene.fps)

    _mutate(path, edit)


def test_symmetry_lines_exist_only_where_declared(tmp_path, monkeypatch):
    """No limit, no line; a declared skeleton and mesh each give their own PASS line; a hind leg turned 10 degrees fails the skeleton."""
    from scripts import verify_motion

    run = build("truebones", "Goat", None, tmp_path)
    assert not [c for c in judge(run, sheets=False).checks if c.id.startswith("C4-symmetry")]
    conf = tmp_path / "conf"
    conf.mkdir()
    for f in Path("configs/verify").glob("*.yaml"):
        (conf / f.name).write_text(f.read_text())
    (conf / "limits.yaml").write_text("Goat: {skeleton: {position: 0.1, rotation_deg: 110.0}, mesh: 0.04}\n")
    monkeypatch.setattr(verify_motion, "CONFIG", conf)
    base = judge(run, sheets=False)
    assert_green(base)
    lines = {c.id: c for c in base.checks if c.id.startswith("C4-symmetry")}
    assert lines.keys() == {"C4-symmetry-skeleton", "C4-symmetry-mesh"} and {c.status for c in lines.values()} == {Status.PASS}
    assert {c.evidence.value for c in lines.values()} == {"derived"} and all(c.value is not None for c in lines.values()), lines
    turn_hind_leg_10deg(run / "5_zero.fbx")
    after = judge(run, accept_changed=True, sheets=False)
    assert _status(after, "C4-symmetry-skeleton").status is Status.FAIL, _status(after, "C4-symmetry-skeleton")


def test_bvh_files_with_and_without_a_mesh(tmp_path):
    """A vendor BVH through `bvh_checks` (directly, against its own source) and through `file` with no mesh anywhere."""
    from scripts import verify_motion
    from poseydon.build.pipeline import compose_build
    from tools.verify.bvh import bvh_checks
    from tools.verify.case import Case, rig_skin

    cfg = verify_motion.load_config()
    vendor = Path("data/_raw/truebones/bvh/Camel/__DownLoop.bvh")
    other = Path("data/_raw/truebones/bvh/Camel/__SlowWalk.bvh")
    text, channels = vendor.read_text(), "Xposition Yposition Zposition Zrotation Yrotation Xrotation"
    # a joint with no position channels, so its OFFSET is where it stands
    offset = next(m.group(1) for m in re.finditer(r"(OFFSET \S+ \S+ \S+)\s+CHANNELS 3", text) if m.group(1).split() != ["OFFSET", *["0.000000"] * 3])
    head, _, motion = text.partition("MOTION")
    skin = rig_skin(compose_build(["data=truebones"]).data, "Camel")

    def copy(name, body=text):
        (tmp_path / name).write_text(body)
        return tmp_path / name

    def lines(path, source=vendor, kind="bvh", case=None):
        return {c.id: c for c in bvh_checks(path, source, kind, case, cfg)}

    def seen(path, **kw):
        return {k: c.status.value for k, c in lines(path, **kw).items()}

    green = {"B1-layout": "PASS", "B2-joints": "PASS", "B4-reader": "NOT RUN"}  # no Blender in `fbxsdk`
    assert seen(copy("ok.bvh")) == green
    assert seen(copy("channels.bvh", text.replace(channels, "Xposition Yposition Zposition Xrotation Yrotation Zrotation", 1)))["B1-layout"] == "FAIL"
    moved = copy("offset.bvh", text.replace(offset, "OFFSET " + " ".join(f"{1.1 * float(v):.6f}" for v in offset.split()[1:]), 1))
    assert seen(moved) == {**green, "B1-layout": "PASS", "B2-joints": "FAIL"}
    assert seen(copy("nan.bvh", head + "MOTION" + motion.replace("0.", "nan", 1)))["B2-joints"] != "PASS"
    assert seen(copy("short.bvh", head + "MOTION\nFrames: 1\n" + motion.split("\n", 3)[2] + "\n" + motion.split("\n", 4)[3]))["B2-joints"] == "FAIL"
    assert seen(copy("empty.bvh", ""))["B2-joints"] == "FAIL"
    # source FBX: no layout to match; the joints compare with the clip's own file; another clip's is no source
    assert seen(vendor, source=skin, kind="fbx") == {k: v for k, v in green.items() if k != "B1-layout"}
    assert seen(other, source=skin, kind="fbx")["B2-joints"] == "NOT RUN"
    assert seen(moved, source=skin, kind="fbx")["B2-joints"] == "FAIL"
    # B3 only with a rig's skin
    case = Case("truebones", "Camel", "DownLoop", {"file": vendor}, skin, skin, np.zeros(0, dtype=int), np.zeros(0, dtype=int))
    assert "B3-mesh" in lines(vendor, case=case) and lines(vendor, case=case)["B3-mesh"].evidence.value == "driven"
    # `file`: no mesh anywhere. The mesh checks are left out and named, the verdict rests on the joints.
    report = judge(verify_motion.file_case(copy("again.bvh"), vendor, tmp_path))
    assert "no mesh in this input" in report.run["not_applicable"]
    assert {c.id for c in report.checks} == set(green) and not any(c.id.startswith(("C", "B3")) for c in report.checks)
    assert [c.id for c in report.checks if c.status is Status.NOT_RUN] == ["B4-reader"] and report.verdict is Status.NOT_RUN
    assert judge(verify_motion.file_case(moved, vendor, tmp_path)).verdict is Status.FAIL
    rig_only = judge(verify_motion.file_case(vendor, None, tmp_path, "truebones", "Camel"))  # the rig's own FBX clip of that name is the source
    assert {c.id: c.status.value for c in rig_only.checks} == {k: v for k, v in green.items() if k != "B1-layout"}
    assert "B1-layout" in rig_only.run["not_applicable"]
