"""The verifier's reference skin, read from a vendor rest asset.

`deform` (tools/verify/skin.py) must reproduce the SDK reading of the same
file (`FBX.skinned_vertices`), and the asset must name the vendor file it came from.
Needs the FBX SDK, so it runs in the `fbxsdk` container only.
"""

from pathlib import Path

import numpy as np
import pytest

fbx = pytest.importorskip("poseydon.io.fbx", exc_type=ImportError)
from tools.verify.skin import deform, extent  # noqa: E402 - needs the SDK too


def test_reference_skin_matches_the_sdk_reading():
    scene = fbx.FBX.read("data/truebones/cache/rigs/Camel/rest.fbx")
    for twist in (0.0, 0.4):  # at rest the world equals the bind, which hides the matrix order
        if twist:
            rot = np.eye(4)
            rot[:2, :2] = [[np.cos(twist), -np.sin(twist)], [np.sin(twist), np.cos(twist)]]
            scene.pose({n: scene.world(scene.frames()[:1])[0, j] @ rot for j, n in enumerate(scene.names)}, 30.0)
        world = {n: scene.world(scene.frames()[:1])[:, j] for j, n in enumerate(scene.names)}
        got = np.concatenate([deform(s, world)[0] for s in scene.skin_data()])
        want = scene.skinned_vertices(0)
        assert got.shape == want.shape
        assert np.abs(got - want).max() / extent(want) < 1e-9
    with np.load("data/truebones/cache/rigs/Camel/rest.npz") as z:
        assert Path(str(z["source_fbx"])).is_file()
