import numpy as np
import pytest

from poseydon.core.spec import FeatureSpec
from poseydon.data.normalize import Normalizer, spec_hash

SPEC = FeatureSpec((("ric_pos", 3), ("rot6d", 6)))


def clips(seed=0, n=2, frames=17, joints=5):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(frames, joints, SPEC.dim)) for _ in range(n)]


def test_fit_produces_per_joint_per_channel_statistics():
    norm = Normalizer.fit(clips(), SPEC)
    assert norm.mean.shape == (5, SPEC.dim)
    assert norm.std.shape == (5, SPEC.dim)


def test_normalize_then_denormalize_round_trips():
    norm = Normalizer.fit(clips(), SPEC)
    x = clips(seed=1, n=1)[0]
    np.testing.assert_allclose(norm.denormalize(norm.normalize(x)), x, atol=1e-9)


def test_normalized_data_is_centred_and_scaled():
    data = clips(seed=2, n=3, frames=400)
    norm = Normalizer.fit(data, SPEC)
    out = norm.normalize(np.concatenate(data, axis=0))
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-4)


def test_constant_channel_does_not_divide_by_zero():
    # A contact flag that never fires has zero variance; the epsilon must keep
    # it finite rather than producing inf or nan.
    spec = FeatureSpec((("foot_contact", 1),))
    data = [np.zeros((10, 3, 1))]
    out = Normalizer.fit(data, spec).normalize(data[0])
    assert np.isfinite(out).all()


def test_statistics_follow_the_spec_not_a_fixed_layout():
    subset = FeatureSpec((("rot6d", 6),))
    norm = Normalizer.fit([np.zeros((4, 3, 6))], subset)
    assert norm.mean.shape[-1] == 6


def test_rejects_statistics_that_do_not_match_the_spec():
    with pytest.raises(ValueError, match="width"):
        Normalizer(mean=np.zeros((3, 4)), std=np.ones((3, 4)), spec=SPEC)


def test_save_load_round_trip(tmp_path):
    norm = Normalizer.fit(clips(), SPEC)
    path = tmp_path / "stats.npz"
    norm.save(path)
    back = Normalizer.load(path)

    np.testing.assert_array_equal(back.mean, norm.mean)
    np.testing.assert_array_equal(back.std, norm.std)
    assert back.spec == norm.spec


def test_spec_hash_is_stable_and_layout_sensitive():
    assert spec_hash(SPEC) == spec_hash(FeatureSpec((("ric_pos", 3), ("rot6d", 6))))
    assert spec_hash(SPEC) != spec_hash(FeatureSpec((("rot6d", 6), ("ric_pos", 3))))
    assert spec_hash(SPEC) != spec_hash(FeatureSpec((("ric_pos", 3),)))
