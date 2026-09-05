"""Transport of per-joint feature-space arrays through a structural edit.

A structural augmentation changes which joints exist and in what order. Two
things live in feature space, indexed by the ORIGINAL joint layout, and must
follow the same edit: per-skeleton normalization statistics
(:class:`~poseydon.data.normalize.Normalizer`) and the T-pose rest frame. A
``JointEdit`` records, for each joint in the NEW layout, which joint in the
layout it was derived from, so both can be re-derived by a gather rather than
recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from poseydon.data.normalize import Normalizer


@dataclass(frozen=True)
class JointEdit:
    """``source_of[new_index]`` is the pre-edit joint index ``new_index`` came from."""

    source_of: tuple[int, ...]

    @classmethod
    def identity(cls, n_joints: int) -> JointEdit:
        return cls(source_of=tuple(range(n_joints)))

    def compose(self, prior: JointEdit) -> JointEdit:
        """``self`` is the edit applied AFTER ``prior``; the result maps straight
        through to whatever ``prior`` itself was relative to."""
        return JointEdit(source_of=tuple(prior.source_of[i] for i in self.source_of))

    def transport(self, normalizer: Normalizer) -> Normalizer:
        index = np.asarray(self.source_of, dtype=np.int64)
        return Normalizer(
            mean=normalizer.mean[index],
            std=normalizer.std[index],
            spec=normalizer.spec,
        )

    def transport_row(self, row: np.ndarray) -> np.ndarray:
        index = np.asarray(self.source_of, dtype=np.int64)
        return row[index]
