"""The motion verifier (tools/verify, scripts/verify_motion.py).

The reference skin must reproduce the SDK reading of the same file, the asset must
name the vendor file it came from, and every mutation of a built round trip must turn
the verifier red on the check that owns it. Needs the FBX SDK, so it runs in the
`fbxsdk` container only.
"""

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
from tools.verify.report import Status  # noqa: E402
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


MUTATIONS = [
    Mutation("strip_mesh", "Camel", None, strip_mesh, {"C1": "FAIL"}),
    Mutation("drop_cluster", "Camel", None, delete_one_cluster, {"C1": "FAIL"}),
    Mutation("no_skin", "Camel", None, delete_skin_deformers, {"C1": "FAIL", "C2a": "NOT RUN"}),
    Mutation("truncated", "Camel", None, truncate_file, {"C1": "FAIL", "C2a": "NOT RUN"}, names="4_roundtrip"),
    Mutation("nan_point", "Camel", None, nan_control_point, {"C2a": "FAIL"}),
    Mutation("topology", "Camel", None, add_a_control_point, {"C2a": "FAIL"}),
    Mutation("missing_link", "Camel", None, rename_a_cluster_link, {"C2a": "FAIL"}, names="Bip01_Pelvis"),
    Mutation("swap_cluster_link", "Camel", None, swap_two_cluster_links, {"C2a": "FAIL"}, keeps_joints=True),
    Mutation("perturb_bind", "Camel", None, perturb_one_link_bind, {"C2a": "FAIL"}, keeps_joints=True),
    Mutation("rebind", "Goat", None, rebind_89bff67, {"C2a": "FAIL"}, bounds=(2e-2, 1e-1)),
]


def _status(report, check):
    """The worst line of ``check`` (one per stage file for C2a)."""
    order = [Status.PASS, Status.ALLOWED, Status.NOT_RUN, Status.FAIL]
    return max((c for c in report.checks if c.id == check), key=lambda c: (order.index(c.status), c.value or 0))


def assert_red(base, after, check, want, rise):
    was, now = _status(base, check), _status(after, check)
    if was.status.value == want:  # already red on the unmutated run: the evidence must grow
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
        assert base.verdict is Status.PASS, [c for c in base.checks if c.status is not Status.PASS]
    before = _world(run / "4_roundtrip.fbx") if m.keeps_joints else None
    m.mutate(run / "4_roundtrip.fbx")
    if before is not None:
        assert np.abs(before - _world(run / "4_roundtrip.fbx")).max() < 1e-4
    with pytest.raises(ValueError, match="changed"):
        judge(run)
    after = judge(run, accept_changed=True)
    for check, want in m.expected.items():
        assert_red(base, after, check, want, m.rise)
    assert after.verdict is not Status.PASS
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
    one early, so its C1 frame range is green (C1 compares key ranges). A one-frame clip cannot be built
    (`--clip tpose` writes a 1-key take under the asset's 31-frame span), so `sample_frames`
    is driven on a one-frame array directly."""
    run = build("unimate", "Scorpion2", None, tmp_path)
    case = load_case(run)
    assert RigRecord.load(Path("data/unimate"), "Scorpion2").clip_params[case.action]["frames"]["window"].tolist() == [1, 91, 1]
    assert list(case.source_frames) == [int(f) + 1 for f in case.frames]
    assert _status(judge(run), "C2a").status is Status.PASS  # C1 still reads the head trim as a frame-range change
    bear = build("truebones", "Bear", None, tmp_path)
    assert list(load_case(bear).source_frames) == list(load_case(bear).frames)
    assert judge(bear).verdict is Status.PASS
    assert list(sample_frames(np.tile(np.eye(4), (1, 3, 1, 1)), 8)) == [0]
