"""The canonical animation containers.

One ``Animation`` is one animation, full length. Ingest never chunks; windowing
is a load-time sampling concern (see the design spec, section 6.6).

Two types live here, and the distinction is the rigid-bone assumption:

``Animation``
    What a motion file actually declares: every joint carries its own
    per-frame translation. BVH permits position channels on any joint, and
    exporters use them -- every one of the 3ds Max Biped rigs in Truebones
    gives all its joints six channels.

``RigidBodyAnimation``
    ``Animation`` plus the promise that only the root translates, so bone
    lengths are constant. Everything downstream of ingest -- features, the
    IK solver, the topology augmentations, the trained representation --
    is built on that promise, so this is the type they ask for.

Keeping the general type underneath means parsing stays lossless and the
rigid assumption becomes an explicit, checked conversion
(:meth:`Animation.as_rigid_body`) rather than something a parser silently
enforces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from poseydon.core.kinematics import check_topological_order, forward_kinematics
from poseydon.core.rotations import quat_apply, quat_inverse

# Rigid translations are built by broadcasting offsets, so they match
# exactly; this only absorbs float round trips through disk.
_RIGID_ATOL = 1e-6


@dataclass(frozen=True)
class Animation:
    """Local joint rotations plus a per-joint translation over a hierarchy.

    Attributes:
        rotations:    ``(F, J, 4)`` local rotations, scalar-last quaternions.
        translations: ``(F, J, 3)`` per-frame local translation of each joint
            from its parent. Joint 0 has no parent, so its entry is the
            root's GLOBAL position.
        offsets:      ``(J, 3)`` rest-pose offset of each joint from its parent.
        parents:      ``(J,)`` int32 parent index, ``-1`` for the root.
        names:        ``J`` joint names, including End Sites.
        fps:          frames per second of this animation as stored.
    """

    rotations: np.ndarray
    translations: np.ndarray
    offsets: np.ndarray
    parents: np.ndarray
    names: tuple[str, ...]
    fps: float

    def __post_init__(self) -> None:
        if self.rotations.ndim != 3 or self.rotations.shape[-1] != 4:
            raise ValueError(f"rotations must be (F, J, 4), got {self.rotations.shape}")
        n_frames, n_joints = self.rotations.shape[:2]
        if self.translations.shape != (n_frames, n_joints, 3):
            raise ValueError(
                f"translations must be ({n_frames}, {n_joints}, 3) to match "
                f"rotations, got {self.translations.shape}"
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

    @property
    def root_pos(self) -> np.ndarray:
        """``(F, 3)`` global position of joint 0."""
        return self.translations[:, 0]

    def global_transforms(self) -> tuple[np.ndarray, np.ndarray]:
        """``(positions (F, J, 3), rotations (F, J, 4))`` in world space."""
        return forward_kinematics(self.rotations, self.translations, self.parents)

    def global_positions(self) -> np.ndarray:
        """``(F, J, 3)`` global joint positions."""
        positions, _ = self.global_transforms()
        return positions

    def is_rigid(self) -> bool:
        """Whether every non-root joint sits at its declared rest offset."""
        return bool(
            np.allclose(
                self.translations[:, 1:],
                self.offsets[np.newaxis, 1:],
                atol=_RIGID_ATOL,
            )
        )

    def as_rigid_body(self, joint_translation: str = "error") -> RigidBodyAnimation:
        """Reinterpret under the rigid-bone assumption.

        ``joint_translation`` decides what happens when non-root joints do
        translate: ``"error"`` (the default) refuses, because silently
        freezing a rig that genuinely translates produces subtly wrong
        motion; ``"drop"`` discards it, pinning every joint to its rest
        offset. Raw Biped exports need ``"drop"`` -- they give every joint
        six channels as an axis-convention artefact, not as real motion.
        """
        if joint_translation not in ("error", "drop"):
            raise ValueError(
                f"joint_translation must be 'error' or 'drop', got {joint_translation!r}"
            )
        if joint_translation == "error" and not self.is_rigid():
            moving = int(
                np.argmax(
                    np.abs(self.translations[:, 1:] - self.offsets[np.newaxis, 1:])
                    .max(axis=(0, 2))
                )
            ) + 1
            raise ValueError(
                f"joint `{self.names[moving]}` translates, so this animation is not "
                "rigid; pass joint_translation='drop' to pin every joint to its rest "
                "offset, or keep it as an Animation"
            )
        return RigidBodyAnimation.from_root_motion(
            rotations=self.rotations,
            root_pos=self.translations[:, 0],
            offsets=self.offsets,
            parents=self.parents,
            names=self.names,
            fps=self.fps,
        )

    def slice(self, start: int, stop: int) -> Animation:
        """A new animation over frames ``[start, stop)``, same skeleton."""
        return type(self)(
            rotations=self.rotations[start:stop].copy(),
            translations=self.translations[start:stop].copy(),
            offsets=self.offsets,
            parents=self.parents,
            names=self.names,
            fps=self.fps,
        )

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            Path(path),
            rotations=self.rotations,
            translations=self.translations,
            offsets=self.offsets,
            parents=self.parents,
            names=np.array(self.names, dtype=object),
            fps=np.float64(self.fps),
        )

    @classmethod
    def load(cls, path: str | Path) -> Animation:
        """Read a saved animation. Returns whichever class this is called on."""
        with np.load(Path(path), allow_pickle=True) as data:
            return cls(
                rotations=data["rotations"],
                translations=data["translations"],
                offsets=data["offsets"],
                parents=data["parents"].astype(np.int32),
                names=tuple(str(n) for n in data["names"]),
                fps=float(data["fps"]),
            )


def rest_geometry(anim: Animation) -> RigidBodyAnimation:
    """Recover true bone geometry from a rest-pose frame's position channels.

    Raw Biped exports carry the real skeleton in their per-joint POSITION
    channels while the OFFSET block describes something else, so bone offsets
    have to be read back out of frame 0's world-space layout and expressed in
    each parent's own frame. Keeps the file's declared rotations, which is what
    :class:`~poseydon.build.prepare.RestRelative` needs as the bind.
    """
    first = anim.slice(0, 1)
    global_pos, global_rot = first.global_transforms()
    parent_of = first.parents[1:]

    offsets = np.zeros_like(first.offsets)
    offsets[1:] = quat_apply(
        quat_inverse(global_rot[0, parent_of]),
        global_pos[0, 1:] - global_pos[0, parent_of],
    )
    return RigidBodyAnimation.from_root_motion(
        rotations=first.rotations,
        root_pos=global_pos[:, 0],
        offsets=offsets,
        parents=first.parents,
        names=first.names,
        fps=first.fps,
    )


@dataclass(frozen=True)
class RigidBodyAnimation(Animation):
    """An :class:`Animation` in which only the root translates.

    Bone lengths are therefore constant, which is what forward kinematics,
    the feature layer, the IK solver and the topology augmentations all
    assume. The invariant is checked on construction rather than trusted,
    so a non-rigid animation cannot reach that code by mistake.

    ``global_positions`` needs no override: ``translations`` repeats each
    joint's own ``offsets`` entry on every frame, so the general sweep in
    :func:`poseydon.core.kinematics.forward_kinematics` computes exactly
    the rigid result. One definition of forward kinematics, not two.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.is_rigid():
            raise ValueError(
                "RigidBodyAnimation requires every non-root joint to sit at its "
                "rest offset; build it with Animation.as_rigid_body() to choose "
                "how per-joint translation is handled"
            )

    @classmethod
    def from_root_motion(
        cls,
        *,
        rotations: np.ndarray,
        root_pos: np.ndarray,
        offsets: np.ndarray,
        parents: np.ndarray,
        names: tuple[str, ...],
        fps: float,
    ) -> RigidBodyAnimation:
        """Build from a root trajectory plus fixed bone offsets.

        The natural constructor for rigid motion: it fills ``translations``
        by repeating ``offsets`` on every frame and putting ``root_pos`` in
        joint 0's slot.
        """
        rotations = np.asarray(rotations, dtype=np.float64)
        root_pos = np.asarray(root_pos, dtype=np.float64)
        offsets = np.asarray(offsets, dtype=np.float64)

        n_frames = rotations.shape[0]
        if root_pos.shape != (n_frames, 3):
            raise ValueError(
                f"root_pos must be ({n_frames}, 3) to match rotations, got {root_pos.shape}"
            )

        translations = np.broadcast_to(offsets, (n_frames, *offsets.shape)).copy()
        translations[:, 0] = root_pos
        return cls(
            rotations=rotations,
            translations=translations,
            offsets=offsets,
            parents=parents,
            names=names,
            fps=fps,
        )
