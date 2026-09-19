"""Structural augmentations, and the statistics that must follow them.

`DropEndEffector` removes leaves, so it is exactly invariant: nothing
downstream referenced a leaf, and `JointEdit.transport` is a pure gather.
`DuplicateJoint` inserts a joint that did not exist, which is the harder case
-- the new joint has no statistics of its own, and borrowing the wrong ones
puts it many sigma out of distribution on every frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from poseydon.augment.joint_edit import JointEdit
from poseydon.data.normalize import Normalizer

CORPUS = Path("data/truebones")


def _normalizer(n_joints: int, spec) -> Normalizer:
    """Distinct per-joint moments, so a mis-gather is visible in the values."""
    mean = np.arange(n_joints, dtype=np.float64)[:, None] * np.ones((1, spec.dim))
    std = np.ones((n_joints, spec.dim))
    return Normalizer(mean=mean, std=std, spec=spec)


def test_a_plain_gather_is_unchanged_by_the_blend_path(spec_stub):
    """`blend_of` defaulting to `source_of` must leave a pure reindex exact --
    every existing augmentation depends on that."""
    edit = JointEdit(source_of=(2, 0, 1))
    out = edit.transport(_normalizer(3, spec_stub))
    assert out.mean[:, 0].tolist() == [2.0, 0.0, 1.0]
    assert edit.transport_row(np.arange(3.0)[:, None]).ravel().tolist() == [2.0, 0.0, 1.0]


def test_a_blended_slot_averages_its_two_sources(spec_stub):
    """A joint inserted BETWEEN two joints is described by the average of their
    statistics, not by either one alone."""
    edit = JointEdit(source_of=(0, 1, 2), blend_of=(0, 3, 2))
    out = edit.transport(_normalizer(4, spec_stub))
    assert out.mean[:, 0].tolist() == [0.0, 2.0, 2.0]   # slot 1 = (1 + 3) / 2
    assert edit.transport_row(np.arange(4.0)[:, None]).ravel().tolist() == [0.0, 2.0, 2.0]


def test_compose_carries_a_blend_through_a_later_gather(spec_stub):
    """Drop runs after duplicate in the pipeline, so a gather composed on top of
    a blend must not silently discard the blend."""
    blended = JointEdit(source_of=(0, 1, 2), blend_of=(0, 3, 2))
    gather = JointEdit(source_of=(1, 2))          # keep slots 1 and 2
    composed = gather.compose(blended)
    assert composed.source_of == (1, 2)
    assert composed.blend_of == (3, 2)


pytestmark_corpus = pytest.mark.skipif(
    not (CORPUS / "index.jsonl").is_file(),
    reason="stage-2 corpus absent; run scripts/build_features.py",
)


@pytestmark_corpus
def test_a_duplicated_stiff_joint_stays_in_distribution():
    """The regression this file exists for.

    `duplicate_joint` puts the new joint half a bone offset from its source and
    the edit used to hand it the SOURCE's mean. For a stiff joint -- Alligator's
    neck has a ric_pos sigma of 0.007 against a 0.169 bone -- that half-offset
    is a constant -11 sigma on every frame, and `simple` squares it. Measured
    ratios reached 395x the sample's median joint energy.
    """
    from poseydon.augment.topology import _excluded_joints, duplicate_joint
    from poseydon.build.index import CorpusIndex
    from poseydon.data.dataset import MotionDataset
    from poseydon.data.window import RandomCrop
    from poseydon.features import extract_features

    ds = MotionDataset(
        index=CorpusIndex.load(CORPUS / "index.jsonl"),
        root=CORPUS,
        manifest_dir=CORPUS / "rigs",
        features=["ric_pos", "rot6d", "local_vel", "foot_contact"],
        window=RandomCrop(length=40),
        split="train",
        seed=0,
    )
    record = next(r for r in ds.records if r.clip_id == "Alligator__die")
    anim, resolved = ds._anim(record), ds._resolved(record)

    counts = np.zeros(anim.n_joints, dtype=int)
    valid = anim.parents >= 0
    np.add.at(counts, anim.parents[valid], 1)
    excluded = _excluded_joints(resolved)
    candidates = [
        j for j in range(1, anim.n_joints)
        if counts[j] == 1 and anim.parents[j] != 0 and j not in excluded
    ]
    assert candidates

    # The STIFFEST candidate -- smallest ric_pos sigma -- is the worst case.
    raw, spec = extract_features(anim, resolved, ds.features)
    base = ds._normalizer(record.skeleton, spec)
    sigma = spec.take(base.std[None], "ric_pos", axis=2)[0]
    joint = min(candidates, key=lambda j: float(sigma[j].min()))

    edited, edited_resolved, edit = duplicate_joint(anim, resolved, joint)
    raw, spec = extract_features(edited, edited_resolved, ds.features)
    z = edit.transport(ds._normalizer(record.skeleton, spec)).normalize(raw)

    energy = (z ** 2).sum(axis=2).mean(axis=0)          # (J,)
    ratio = float(energy[joint] / np.median(np.delete(energy, joint)))
    assert ratio < 5.0, (
        f"inserted joint carries {ratio:.0f}x the sample's median joint energy; "
        f"its statistics do not describe it"
    )
