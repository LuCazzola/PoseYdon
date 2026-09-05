from dataclasses import dataclass

import numpy as np

from poseydon.augment.base import AUGMENTATIONS, Augmentation, AugmentPipeline
from poseydon.augment.joint_edit import JointEdit


class _FakeAnim:
    """Stand-in for poseydon.core.anim.Anim: only `.n_joints` is needed here."""

    def __init__(self, n_joints: int) -> None:
        self.n_joints = n_joints


@dataclass
class _DropLast(Augmentation):
    """Test double: always drops the current last joint."""

    def apply_structural(self, anim, resolved, rng):
        new_anim = _FakeAnim(anim.n_joints - 1)
        edit = JointEdit(source_of=tuple(range(anim.n_joints - 1)))
        return new_anim, resolved, edit


@dataclass
class _AddOne(Augmentation):
    """Test double: always adds 1 to every feature value."""

    def apply_features(self, features, spec, resolved, rng):
        return features + 1


def test_default_hooks_are_no_ops():
    aug = Augmentation()
    anim = _FakeAnim(5)
    rng = np.random.default_rng(0)

    _, _, edit = aug.apply_structural(anim, None, rng)
    assert edit.source_of == (0, 1, 2, 3, 4)

    features = np.zeros((2, 3))
    assert aug.apply_features(features, None, None, rng) is features


def test_pipeline_skips_when_the_probability_trial_fails():
    pipeline = AugmentPipeline([_DropLast(p=0.0)])
    anim = _FakeAnim(5)

    _, _, edit = pipeline.apply_structural(anim, None, np.random.default_rng(0))

    assert edit.source_of == (0, 1, 2, 3, 4)


def test_pipeline_applies_when_the_probability_trial_passes():
    pipeline = AugmentPipeline([_DropLast(p=1.0)])
    anim = _FakeAnim(5)

    _, _, edit = pipeline.apply_structural(anim, None, np.random.default_rng(0))

    assert edit.source_of == (0, 1, 2, 3)


def test_pipeline_composes_structural_edits_across_entries_in_order():
    pipeline = AugmentPipeline([_DropLast(p=1.0), _DropLast(p=1.0)])
    anim = _FakeAnim(5)

    _, _, edit = pipeline.apply_structural(anim, None, np.random.default_rng(0))

    assert edit.source_of == (0, 1, 2)


def test_pipeline_applies_feature_hooks_in_order():
    pipeline = AugmentPipeline([_AddOne(p=1.0), _AddOne(p=1.0)])
    features = np.zeros((2, 2))

    out = pipeline.apply_features(features, None, None, np.random.default_rng(0))

    np.testing.assert_array_equal(out, np.full((2, 2), 2.0))


def test_empty_pipeline_is_a_no_op():
    pipeline = AugmentPipeline([])
    anim = _FakeAnim(5)
    rng = np.random.default_rng(0)

    _, _, edit = pipeline.apply_structural(anim, None, rng)
    assert edit.source_of == (0, 1, 2, 3, 4)

    features = np.zeros((2, 2))
    out = pipeline.apply_features(features, None, None, rng)
    np.testing.assert_array_equal(out, features)


def test_augmentations_registry_exists_and_starts_empty_before_topology_import():
    registry = AUGMENTATIONS
    assert registry.kind == "augmentation"
