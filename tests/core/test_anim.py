import numpy as np
import pytest

from poseydon.core.anim import Anim
from poseydon.core.rotations import QUAT_IDENTITY


def make_anim(n_frames=4, n_joints=3) -> Anim:
    rng = np.random.default_rng(0)
    q = rng.normal(size=(n_frames, n_joints, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return Anim(
        rotations=q,
        root_pos=rng.normal(size=(n_frames, 3)),
        offsets=rng.normal(size=(n_joints, 3)),
        parents=np.array([-1] + list(range(n_joints - 1)), dtype=np.int32),
        names=tuple(f"joint_{i}" for i in range(n_joints)),
        fps=24.0,
    )


def test_shape_properties():
    anim = make_anim(n_frames=7, n_joints=5)
    assert anim.n_frames == 7
    assert anim.n_joints == 5


def test_rejects_name_count_mismatch():
    with pytest.raises(ValueError, match="names"):
        Anim(
            rotations=np.broadcast_to(QUAT_IDENTITY, (2, 3, 4)).copy(),
            root_pos=np.zeros((2, 3)),
            offsets=np.zeros((3, 3)),
            parents=np.array([-1, 0, 1], dtype=np.int32),
            names=("only", "two"),
            fps=24.0,
        )


def test_rejects_frame_count_mismatch():
    with pytest.raises(ValueError, match="root_pos"):
        Anim(
            rotations=np.broadcast_to(QUAT_IDENTITY, (2, 3, 4)).copy(),
            root_pos=np.zeros((5, 3)),
            offsets=np.zeros((3, 3)),
            parents=np.array([-1, 0, 1], dtype=np.int32),
            names=("a", "b", "c"),
            fps=24.0,
        )


def test_npz_round_trip(tmp_path):
    anim = make_anim()
    path = tmp_path / "clip.npz"
    anim.save(path)
    back = Anim.load(path)

    np.testing.assert_allclose(back.rotations, anim.rotations, atol=0)
    np.testing.assert_allclose(back.root_pos, anim.root_pos, atol=0)
    np.testing.assert_allclose(back.offsets, anim.offsets, atol=0)
    np.testing.assert_array_equal(back.parents, anim.parents)
    assert back.names == anim.names
    assert back.fps == anim.fps


def test_global_positions_places_root_at_root_pos():
    anim = make_anim()
    np.testing.assert_allclose(anim.global_positions()[:, 0], anim.root_pos, atol=1e-12)


def test_slice_selects_frames():
    anim = make_anim(n_frames=10)
    cut = anim.slice(2, 5)
    assert cut.n_frames == 3
    assert cut.n_joints == anim.n_joints
    np.testing.assert_allclose(cut.root_pos, anim.root_pos[2:5], atol=0)
    assert cut.names == anim.names
