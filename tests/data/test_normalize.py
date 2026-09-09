"""The per-block normalization policy (parity spec §5).

Each `scale` mode must produce statistics of a documented SHAPE, and
`joint_block` must leave a 6D rotation row's recovered rotation unchanged under
Gram-Schmidt -- which is the property that makes the rotation block survivable
at all, and the reason pooling exists rather than per-channel scaling.
"""

from __future__ import annotations

import numpy as np
import pytest

from poseydon.core.spec import FeatureSpec
from poseydon.data.normalize import BlockPolicy, Normalizer

SPEC = FeatureSpec((("ric_pos", 3), ("rot6d", 6), ("foot_contact", 1)))


def _clips(n_clips: int = 3, frames: int = 20, joints: int = 5) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n_clips):
        a = rng.normal(size=(frames, joints, SPEC.dim))
        # Give channels within a block deliberately different scales, so a
        # pooled statistic is provably not the per-channel one.
        a[..., SPEC.slice("ric_pos")] *= np.array([1.0, 10.0, 100.0])
        a[..., SPEC.slice("foot_contact")] = 1.0     # zero variance, on purpose
        out.append(a)
    return out


def _gram_schmidt(row: np.ndarray) -> np.ndarray:
    """The 6D-to-rotation recovery: two vectors, orthonormalized."""
    a, b = row[:3], row[3:]
    e1 = a / np.linalg.norm(a)
    rest = b - (e1 @ b) * e1
    e2 = rest / np.linalg.norm(rest)
    return np.stack([e1, e2, np.cross(e1, e2)])


def test_channel_mode_keeps_a_statistic_per_joint_and_channel():
    fit = Normalizer.fit(_clips(), SPEC, [BlockPolicy("ric_pos", scale="channel")])
    block = fit.std[:, SPEC.slice("ric_pos")]
    # The three channels were scaled 1/10/100, so per-channel stds must differ.
    assert not np.allclose(block[:, 0], block[:, 1])


def test_joint_block_mode_gives_one_scalar_per_joint_and_block():
    fit = Normalizer.fit(_clips(), SPEC, [BlockPolicy("ric_pos", scale="joint_block")])
    block = fit.std[:, SPEC.slice("ric_pos")]
    for joint in range(block.shape[0]):
        assert np.allclose(block[joint], block[joint, 0]), (
            "joint_block must be uniform across the block's channels"
        )
    # Different joints may still differ; if they did not, this would be `block`.
    assert block.shape == (5, 3)


def test_block_mode_gives_one_scalar_for_root_and_one_for_the_rest():
    fit = Normalizer.fit(_clips(), SPEC, [BlockPolicy("ric_pos", scale="block")])
    block = fit.std[:, SPEC.slice("ric_pos")]
    assert np.allclose(block[0], block[0, 0])
    assert np.allclose(block[1:], block[1, 0]), "non-root joints must share one scalar"


def test_none_mode_leaves_std_at_one():
    fit = Normalizer.fit(
        _clips(), SPEC, [BlockPolicy("foot_contact", center=False, scale="none")]
    )
    np.testing.assert_allclose(fit.std[:, SPEC.slice("foot_contact")], 1.0)
    np.testing.assert_allclose(fit.mean[:, SPEC.slice("foot_contact")], 0.0)


def test_joint_block_preserves_the_recovered_rotation():
    """The property, not an example: a uniform divisor commutes with Gram-Schmidt.

    Per-channel scaling does not, which is why the rotation block is pooled.
    """
    fit = Normalizer.fit(_clips(), SPEC, [BlockPolicy("rot6d", center=False,
                                                     scale="joint_block")])
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(1, 5, SPEC.dim))
    normalized = fit.normalize(raw)

    for joint in range(5):
        before = _gram_schmidt(raw[0, joint, SPEC.slice("rot6d")])
        after = _gram_schmidt(normalized[0, joint, SPEC.slice("rot6d")])
        np.testing.assert_allclose(after, before, atol=1e-12)


def test_per_channel_scaling_does_NOT_preserve_the_recovered_rotation():
    """The contrast that makes the previous test meaningful."""
    fit = Normalizer.fit(_clips(), SPEC, [BlockPolicy("rot6d", center=False,
                                                      scale="channel")])
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(1, 5, SPEC.dim))
    normalized = fit.normalize(raw)

    before = _gram_schmidt(raw[0, 0, SPEC.slice("rot6d")])
    after = _gram_schmidt(normalized[0, 0, SPEC.slice("rot6d")])
    assert not np.allclose(after, before, atol=1e-6)


def test_no_policy_reproduces_the_previous_behaviour():
    """Every existing caller must be unaffected."""
    clips = _clips()
    np.testing.assert_allclose(
        Normalizer.fit(clips, SPEC).std, Normalizer.fit(clips, SPEC, None).std
    )


def test_an_unknown_scale_mode_is_refused():
    with pytest.raises(ValueError, match="nonsense"):
        Normalizer.fit(_clips(), SPEC, [BlockPolicy("ric_pos", scale="nonsense")])


def test_a_policy_naming_an_absent_block_is_refused():
    with pytest.raises(ValueError, match="no_such_block"):
        Normalizer.fit(_clips(), SPEC, [BlockPolicy("no_such_block")])


def test_the_policy_survives_save_and_load(tmp_path):
    policy = [BlockPolicy("ric_pos", scale="joint_block"),
              BlockPolicy("rot6d", center=False, scale="joint_block"),
              BlockPolicy("foot_contact", center=False, scale="none")]
    fit = Normalizer.fit(_clips(), SPEC, policy)
    path = tmp_path / "stats.npz"
    fit.save(path)
    back = Normalizer.load(path)
    np.testing.assert_allclose(back.mean, fit.mean)
    np.testing.assert_allclose(back.std, fit.std)
    assert back.spec.blocks == fit.spec.blocks
