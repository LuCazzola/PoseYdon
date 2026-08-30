"""The canonical animation container.

One ``Anim`` is one animation, full length. Ingest never chunks; windowing is a
load-time sampling concern (see the design spec, section 6.6).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from poseydon.core.kinematics import check_topological_order, forward_kinematics


@dataclass(frozen=True)
class Anim:
    """Local joint rotations plus a root trajectory over a fixed skeleton.

    Attributes:
        rotations: ``(F, J, 4)`` local rotations, scalar-last quaternions.
        root_pos:  ``(F, 3)`` global translation of joint 0.
        offsets:   ``(J, 3)`` rest-pose offset of each joint from its parent.
        parents:   ``(J,)`` int32 parent index, ``-1`` for the root.
        names:     ``J`` joint names, including End Sites.
        fps:       frames per second of this animation as stored.
    """

    rotations: np.ndarray
    root_pos: np.ndarray
    offsets: np.ndarray
    parents: np.ndarray
    names: tuple[str, ...]
    fps: float

    def __post_init__(self) -> None:
        if self.rotations.ndim != 3 or self.rotations.shape[-1] != 4:
            raise ValueError(f"rotations must be (F, J, 4), got {self.rotations.shape}")
        n_frames, n_joints = self.rotations.shape[:2]
        if self.root_pos.shape != (n_frames, 3):
            raise ValueError(
                f"root_pos must be ({n_frames}, 3) to match rotations, "
                f"got {self.root_pos.shape}"
            )
        if self.offsets.shape != (n_joints, 3):
            raise ValueError(f"offsets must be ({n_joints}, 3), got {self.offsets.shape}")
        if self.parents.shape != (n_joints,):
            raise ValueError(f"parents must be ({n_joints},), got {self.parents.shape}")
        if len(self.names) != n_joints:
            raise ValueError(
                f"names must have {n_joints} entries to match rotations, got {len(self.names)}"
            )
        if len(set(self.names)) != n_joints:
            raise ValueError("joint names must be unique")
        check_topological_order(self.parents)

    @property
    def n_frames(self) -> int:
        return int(self.rotations.shape[0])

    @property
    def n_joints(self) -> int:
        return int(self.rotations.shape[1])

    def global_positions(self) -> np.ndarray:
        """``(F, J, 3)`` global joint positions."""
        positions, _ = forward_kinematics(
            self.rotations, self.root_pos, self.offsets, self.parents
        )
        return positions

    def slice(self, start: int, stop: int) -> "Anim":
        """A new ``Anim`` over frames ``[start, stop)``, same skeleton."""
        return Anim(
            rotations=self.rotations[start:stop].copy(),
            root_pos=self.root_pos[start:stop].copy(),
            offsets=self.offsets,
            parents=self.parents,
            names=self.names,
            fps=self.fps,
        )

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            Path(path),
            rotations=self.rotations,
            root_pos=self.root_pos,
            offsets=self.offsets,
            parents=self.parents,
            names=np.array(self.names, dtype=object),
            fps=np.float64(self.fps),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Anim":
        with np.load(Path(path), allow_pickle=True) as data:
            return cls(
                rotations=data["rotations"],
                root_pos=data["root_pos"],
                offsets=data["offsets"],
                parents=data["parents"].astype(np.int32),
                names=tuple(str(n) for n in data["names"]),
                fps=float(data["fps"]),
            )
