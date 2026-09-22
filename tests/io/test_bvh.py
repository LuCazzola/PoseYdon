"""``BVH.read_names`` must agree with a full ``read().to_animation().names``."""

from __future__ import annotations

from pathlib import Path

import pytest

from poseydon.io.bvh import BVH
from tests.conftest import CORPUS, SAMPLE_RIGS


def test_read_names_matches_full_read():
    clips = sorted((Path("data/_raw/truebones/source") / SAMPLE_RIGS[0]).glob("*.bvh"))
    if not clips:
        pytest.skip(f"no raw clips for {SAMPLE_RIGS[0]}")
    path = clips[0]

    names_only = BVH.read_names(path)
    full = BVH.read(path).to_animation().names

    assert names_only == full


def test_write_honours_each_joints_declared_rotation_order(tmp_path):
    """A declared order is a promise about the numbers written under it.

    The writer used to emit one order for every joint regardless of what the
    struct declared, so a rig restored to a source file's ``ZXY`` layout was
    labelled ``ZYX`` -- self-consistent as a file, but not the layout the
    caller asked for, and wrong for anything reading the block back onto a rig
    that expects the source order.
    """
    import numpy as np

    from poseydon.core.animation import Animation
    from poseydon.core.rotations import euler_to_quat

    angles = np.array([[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]])
    offsets = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    rotations = np.concatenate(
        [
            euler_to_quat(angles[:, 0], "ZXY")[:, None],
            euler_to_quat(angles[:, 1], "ZXY")[:, None],
            np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (1, 1, 1)),
        ],
        axis=1,
    )
    anim = Animation(
        rotations=rotations,
        translations=np.tile(offsets, (1, 1, 1)),
        offsets=offsets,
        parents=np.array([-1, 0, 1], dtype=np.int32),
        names=("root", "child", "leaf"),
        fps=30.0,
    )
    declared = (
        ("Xposition", "Yposition", "Zposition", "Zrotation", "Xrotation", "Yrotation"),
        ("Zrotation", "Xrotation", "Yrotation"),
        (),
    )
    path = tmp_path / "zxy.bvh"
    BVH.from_animation(anim, channels=declared).write(path)

    text = path.read_text()
    assert "Zrotation Xrotation Yrotation" in text
    assert "Zrotation Yrotation Xrotation" not in text

    reread = BVH.read(path)
    assert reread.channels == declared
    np.testing.assert_allclose(reread.rotations, anim.rotations, atol=1e-9)


def test_default_channels_name_the_order_the_writer_emits(tmp_path):
    """The struct's own declaration and the file it writes must agree."""
    import numpy as np

    from poseydon.core.animation import Animation

    anim = Animation(
        rotations=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (1, 2, 1)),
        translations=np.tile(np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), (1, 1, 1)),
        offsets=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        parents=np.array([-1, 0], dtype=np.int32),
        names=("root", "child"),
        fps=30.0,
    )
    built = BVH.from_animation(anim)
    path = tmp_path / "default.bvh"
    built.write(path)
    assert BVH.read(path).channels == built.channels
