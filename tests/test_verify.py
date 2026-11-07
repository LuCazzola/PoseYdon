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

fbx = pytest.importorskip("poseydon.io.fbx", exc_type=ImportError)
from scripts.verify_motion import build, judge  # noqa: E402 - needs the SDK too
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


@dataclass
class Mutation:
    id: str
    rig: str
    clip: str | None
    mutate: Callable[[Path], None]
    expected: dict[str, str]
    rise: float = 0.0


MUTATIONS = [
    Mutation("strip_mesh", "Camel", None, strip_mesh, {"C1": "FAIL"}),
    Mutation("drop_cluster", "Camel", None, delete_one_cluster, {"C1": "FAIL"}),
    Mutation("no_skin", "Camel", None, delete_skin_deformers, {"C1": "FAIL"}),
    Mutation("truncated", "Camel", None, truncate_file, {"C1": "FAIL"}),
]


def _status(report, check):
    return next(c for c in report.checks if c.id == check)


def assert_red(base, after, check, want, rise):
    was, now = _status(base, check), _status(after, check)
    if was.status.value == want:  # already red on the unmutated run: the evidence must grow
        assert now.value >= was.value + rise, f"{check}: {was.value} -> {now.value}"
    else:
        assert now.status.value == want, f"{check}: {now.status.value} {now.note}"


@pytest.mark.parametrize("m", MUTATIONS, ids=lambda m: m.id)
def test_verifier_goes_red(m, tmp_path):
    run = build("truebones", m.rig, m.clip, tmp_path)
    base = judge(run)
    assert base.verdict is Status.PASS, [c for c in base.checks if c.status is not Status.PASS]
    m.mutate(run / "4_roundtrip.fbx")
    with pytest.raises(ValueError, match="changed"):
        judge(run)
    after = judge(run, accept_changed=True)
    for check, want in m.expected.items():
        assert_red(base, after, check, want, m.rise)
    assert after.verdict is not Status.PASS
