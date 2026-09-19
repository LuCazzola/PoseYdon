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
names its second source, and transport averages the pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from poseydon.data.normalize import Normalizer


@dataclass(frozen=True)
class JointEdit:
    """``source_of[new_index]`` is the pre-edit joint index ``new_index`` came from."""

    source_of: tuple[int, ...]
    #: Second pre-edit source per new joint, for a joint that was INSERTED
    #: between two others. ``None`` -- and any entry equal to its ``source_of``
    #: counterpart -- means a plain gather, so an edit that inserts nothing
    #: behaves exactly as it did before this existed.
    blend_of: tuple[int, ...] | None = None

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
        return JointEdit(
            source_of=tuple(theirs[i] for i in mine),
            blend_of=tuple(theirs_blend[i] for i in mine_blend),
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
        return Normalizer(
            mean=(normalizer.mean[first] + normalizer.mean[second]) / 2.0,
            std=(normalizer.std[first] + normalizer.std[second]) / 2.0,
            spec=normalizer.spec,
        )

    def transport_row(self, row: np.ndarray) -> np.ndarray:
        """One ``(J, D)`` row -- the rest frame -- through the same edit.

        Averaged for the same reason as `transport`: the rest pose is how the
        model is told what the rig IS, and describing an inserted midpoint by
        its source's rest row contradicts the features handed over beside it.
        """
        first, second = (np.asarray(s, dtype=np.int64) for s in self._sources())
        return (row[first] + row[second]) / 2.0
