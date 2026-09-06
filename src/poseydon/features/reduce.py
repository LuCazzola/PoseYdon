"""Removing joints that carry nothing learnable, and putting them back.

A zero-length bone is invisible: its child sits exactly on its parent forever,
so the rotation orienting that bone moves nothing. Truebones rigs carry ten or
more per character -- every End Site, plus the odd dummy -- and they inflate the
feature tensor by roughly a fifth with columns that duplicate a neighbour.

Reduction is applied at FEATURE EXTRACTION time and never baked into the corpus.
Skin weights are indexed by the original joint order, so a generated clip has to
come back on the rig the user supplied; the corpus therefore keeps every joint
and this module is the lens the model looks through.

Two cases, and they are not interchangeable:

* a zero-offset LEAF can be deleted outright -- nothing hangs off it and the
  rotation reaching it is unobservable, so nothing is lost;
* a zero-offset INTERNAL joint swings its subtree, so it can only be folded into
  its parent -- and that fold turns the parent, hence every OTHER child of the
  parent too. It is exact only when the joint is its parent's sole child.
  Otherwise the joint is kept, because displacing its siblings to save one token
  is not a trade worth making.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from poseydon.augment.joint_edit import JointEdit
from poseydon.core.animation import RigidBodyAnimation
from poseydon.core.rotations import QUAT_IDENTITY, quat_mul

DROP = "drop"
COLLAPSE = "collapse"


@dataclass(frozen=True)
class RemovalOp:
    """One removal, described by NAME so it survives reindexing."""

    name: str
    parent: str
    index: int
    kind: str
    children: tuple[str, ...] = ()


@dataclass(frozen=True)
class JointReduction:
    """How to get from a full rig to the reduced one, and back."""

    ops: tuple[RemovalOp, ...]
    source_of: tuple[int, ...]
    source_names: tuple[str, ...]

    @property
    def is_identity(self) -> bool:
        return not self.ops

    def joint_edit(self) -> JointEdit:
        """The same map as a :class:`JointEdit`, for transporting statistics."""
        return JointEdit(source_of=self.source_of)


def _children_of(parents: np.ndarray, joint: int) -> list[int]:
    return [j for j in range(len(parents)) if parents[j] == joint]


def _next_removal(anim: RigidBodyAnimation, tolerance: float) -> RemovalOp | None:
    """The first removable joint, or None when the rig is fully reduced."""
    lengths = np.linalg.norm(anim.offsets, axis=-1)
    for joint in range(1, anim.n_joints):
        if lengths[joint] >= tolerance:
            continue
        parent = int(anim.parents[joint])
        children = _children_of(anim.parents, joint)
        if not children:
            return RemovalOp(anim.names[joint], anim.names[parent], joint, DROP)
        if len(_children_of(anim.parents, parent)) == 1:
            return RemovalOp(
                anim.names[joint],
                anim.names[parent],
                joint,
                COLLAPSE,
                tuple(anim.names[c] for c in children),
            )
    return None


def _remove(anim: RigidBodyAnimation, op: RemovalOp) -> RigidBodyAnimation:
    """Apply one removal, preserving every surviving joint's world transform."""
    victim = anim.names.index(op.name)
    parent = int(anim.parents[victim])

    rotations = anim.rotations.copy()
    if op.kind == COLLAPSE:
        # global_rot'[parent] becomes global_rot[victim], which is what the
        # reparented children need. Safe only because `parent` has no other
        # child to be turned by it -- checked when the op was chosen.
        rotations[:, parent] = quat_mul(rotations[:, parent], rotations[:, victim])

    keep = [j for j in range(anim.n_joints) if j != victim]
    old_to_new = {old: new for new, old in enumerate(keep)}

    parents = np.empty(len(keep), dtype=np.int32)
    for new, old in enumerate(keep):
        ancestor = int(anim.parents[old])
        if ancestor < 0:
            parents[new] = -1
        elif ancestor == victim:
            # The victim's children inherit its parent.
            parents[new] = old_to_new[parent]
        else:
            parents[new] = old_to_new[ancestor]

    return RigidBodyAnimation.from_root_motion(
        rotations=rotations[:, keep],
        root_pos=anim.root_pos,
        offsets=anim.offsets[keep],
        parents=parents,
        names=tuple(anim.names[j] for j in keep),
        fps=anim.fps,
    )


def build_reduction(
    anim: RigidBodyAnimation, tolerance: float = 1e-8
) -> JointReduction:
    """Work out which joints to remove, repeating until none is left.

    Repetition matters: dropping a zero-offset leaf can leave its parent a
    zero-offset leaf in turn.
    """
    ops: list[RemovalOp] = []
    current = anim
    # Re-derived every iteration, deliberately: dropping a zero-offset leaf can
    # leave its parent a zero-offset leaf in turn, and that parent is then
    # droppable by case 1 even though it began as an internal joint with
    # siblings. Scorpion/Bip01_Neck1 is exactly this shape. Testing the
    # CURRENT hierarchy ensures we capture joints as they actually become
    # droppable, not as they began.
    while (op := _next_removal(current, tolerance)) is not None:
        ops.append(op)
        current = _remove(current, op)

    index_of = {name: i for i, name in enumerate(anim.names)}
    return JointReduction(
        ops=tuple(ops),
        source_of=tuple(index_of[name] for name in current.names),
        source_names=tuple(anim.names),
    )


def apply_reduction(
    anim: RigidBodyAnimation, reduction: JointReduction
) -> RigidBodyAnimation:
    """Remove the joints the reduction names, in the order it recorded."""
    if tuple(anim.names) != reduction.source_names:
        raise ValueError(
            "this animation's joints differ from the rig the reduction was built "
            f"for ({len(anim.names)} vs {len(reduction.source_names)} joints)"
        )
    current = anim
    for op in reduction.ops:
        current = _remove(current, op)
    return current


def invert_reduction(
    anim: RigidBodyAnimation, reduction: JointReduction
) -> RigidBodyAnimation:
    """Put every removed joint back, in reverse order.

    A collapsed joint returns with identity rotation and the whole composed
    product left on its parent. The two are coincident, so every joint lands
    exactly where it was; which of the pair stores the rotation is unobservable
    in world space, and for generated motion there was never an original split.
    """
    current = anim
    for op in reversed(reduction.ops):
        current = _reinsert(current, op)
    return current


def _reinsert(anim: RigidBodyAnimation, op: RemovalOp) -> RigidBodyAnimation:
    parent = anim.names.index(op.parent)
    n_new = anim.n_joints + 1

    order = list(range(anim.n_joints))
    order.insert(op.index, -1)  # -1 marks the joint being restored

    names = tuple(op.name if old < 0 else anim.names[old] for old in order)
    new_of = {old: new for new, old in enumerate(order) if old >= 0}
    restored = op.index

    rotations = np.empty((anim.n_frames, n_new, 4))
    offsets = np.zeros((n_new, 3))
    parents = np.empty(n_new, dtype=np.int32)

    moved = set(op.children)
    for new, old in enumerate(order):
        if old < 0:
            rotations[:, new] = QUAT_IDENTITY
            offsets[new] = 0.0
            parents[new] = new_of[parent]
            continue
        rotations[:, new] = anim.rotations[:, old]
        offsets[new] = anim.offsets[old]
        ancestor = int(anim.parents[old])
        if anim.names[old] in moved:
            parents[new] = restored
        else:
            parents[new] = -1 if ancestor < 0 else new_of[ancestor]

    return RigidBodyAnimation.from_root_motion(
        rotations=rotations,
        root_pos=anim.root_pos,
        offsets=offsets,
        parents=parents,
        names=names,
        fps=anim.fps,
    )


@dataclass(frozen=True)
class DropDegenerateJoints:
    """Config-facing wrapper, named by ``_target_`` in a dataset config."""

    tolerance: float = 1e-8

    def build(self, anim: RigidBodyAnimation) -> JointReduction:
        return build_reduction(anim, self.tolerance)
