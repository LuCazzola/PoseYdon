"""Transport of per-joint feature-space arrays through a structural edit.

A structural augmentation changes which joints exist and in what order. Two
things live in feature space, indexed by the ORIGINAL joint layout, and must
follow the same edit: per-skeleton normalization statistics
(:class:`~poseydon.data.normalize.Normalizer`) and the T-pose rest frame. A
``JointEdit`` records, for each joint in the NEW layout, which joint in the
layout it was derived from, so both can be re-derived by a gather rather than
recomputed.

A joint that is INSERTED, rather than kept or moved, has no single source:
it sits between two joints and is described by neither alone. ``blend_of``
names its second source, and transport averages the pair -- except where the
inserted joint's value is known exactly: its own rotation is identity on every
frame, and ``identity_slots`` names the (block, joint) slots that carry it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from poseydon.core.rotations import matrix_to_rot6d
from poseydon.core.spec import FeatureSpec
from poseydon.data.normalize import Normalizer
from poseydon.features.kinematics import _REFERENCE_6D_LAYOUT


@dataclass(frozen=True)
class JointEdit:
    """``source_of[new_index]`` is the pre-edit joint index ``new_index`` came from."""

    source_of: tuple[int, ...]
    #: Second pre-edit source per new joint, for a joint that was INSERTED
    #: between two others. ``None`` -- and any entry equal to its ``source_of``
    #: counterpart -- means a plain gather, so an edit that inserts nothing
    #: behaves exactly as it did before this existed.
    blend_of: tuple[int, ...] | None = None
    #: ``(block, new_index)`` slots holding an identity rotation on every frame:
    #: an inserted joint's own rotation, in whichever slot its layout stores it
    #: (`rot6d_joint` at the joint, `rot6d_bone` at its child). Averaging the
    #: neighbours' moments there is wrong whenever their rotations sit far from
    #: identity, as on a bone-relative corpus (Roach's `Bip01_Tail`: 137 sigma).
    identity_slots: tuple[tuple[str, int], ...] = ()

    def _sources(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self.source_of, self.blend_of or self.source_of

    @classmethod
    def identity(cls, n_joints: int) -> JointEdit:
        return cls(source_of=tuple(range(n_joints)))

    def compose(self, prior: JointEdit) -> JointEdit:
        """``self`` is the edit applied AFTER ``prior``; the result maps straight
        through to whatever ``prior`` itself was relative to.

        Both source lists are mapped through ``prior``, so a gather composed on
        top of an insertion keeps the insertion's pair. Two insertions touching
        one slot would need four sources and are approximated by the outer pair;
        nothing in the tree inserts twice -- ``DuplicateJoint`` inserts one
        joint per sample.
        """
        mine, mine_blend = self._sources()
        theirs, theirs_blend = prior._sources()
        # The prior's slots follow its joints into this layout; a joint this
        # edit dropped takes its slot with it.
        new_of = {old: new for new, old in enumerate(mine) if mine_blend[new] == old}
        carried = tuple((b, new_of[j]) for b, j in prior.identity_slots if j in new_of)
        return JointEdit(
            source_of=tuple(theirs[i] for i in mine),
            blend_of=tuple(theirs_blend[i] for i in mine_blend),
            identity_slots=carried + self.identity_slots,
        )

    def transport(self, normalizer: Normalizer) -> Normalizer:
        """The statistics re-expressed in the new layout.

        An inserted joint takes the MEAN of its two neighbours', which is exact
        for a midpoint: under forward kinematics its position is the midpoint of
        theirs, and averaging commutes with that. The averaged sigma is an
        approximation -- the exact one needs the pair's cross-covariance -- but
        sigma was never what hurt. Borrowing the source's moments alone left the
        inserted joint centred half a bone away from its own mean, which on a
        stiff joint (Alligator's neck: sigma 0.007 against a 0.169 bone) is a
        constant 11 sigma on every frame, and `simple` squares it.
        """
        first, second = (np.asarray(s, dtype=np.int64) for s in self._sources())
        mean = (normalizer.mean[first] + normalizer.mean[second]) / 2.0
        std = (normalizer.std[first] + normalizer.std[second]) / 2.0
        # A slot known to hold identity is a constant: centred on it and left
        # unscaled, the way `Normalizer` treats every constant channel.
        for block, joint in self._identity_slots_in(normalizer.spec):
            mean[joint, block] = _IDENTITY_6D
            std[joint, block] = 1.0
        return Normalizer(mean=mean, std=std, spec=normalizer.spec)

    def transport_row(self, row: np.ndarray, spec: FeatureSpec) -> np.ndarray:
        """One NORMALIZED ``(J, D)`` row -- the rest frame -- through the same edit.

        Averaged for the same reason as `transport`: the rest pose is how the
        model is told what the rig IS, and describing an inserted midpoint by
        its source's rest row contradicts the features handed over beside it.
        An identity slot is exactly 0 under the statistics `transport` gives it.
        """
        first, second = (np.asarray(s, dtype=np.int64) for s in self._sources())
        out = (row[first] + row[second]) / 2.0
        for block, joint in self._identity_slots_in(spec):
            out[joint, block] = 0.0
        return out

    def _identity_slots_in(self, spec: FeatureSpec) -> list[tuple[slice, int]]:
        """The identity slots whose block this spec carries, as channel slices."""
        return [(spec.slice(name), joint) for name, joint in self.identity_slots if name in spec]


_IDENTITY_6D = matrix_to_rot6d(np.eye(3), layout=_REFERENCE_6D_LAYOUT)
