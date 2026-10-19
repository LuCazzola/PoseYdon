"""Removing joints that carry nothing learnable, and putting them back.

A zero-length bone is invisible: its child sits exactly on its parent forever,
so the rotation orienting that bone moves nothing. Truebones rigs carry ten or
more per character -- every End Site, plus the odd dummy -- and they inflate the
feature tensor by roughly a fifth with columns that duplicate a neighbour.

Reduction is applied at FEATURE EXTRACTION time and never baked into the corpus.
A generated clip has to come back on the rig the user supplied, so the corpus
keeps every joint that preparation's rig cleanup (`prune_unskinned`) did not
remove, and this module is the lens the model looks through.

Two cases, and they are not interchangeable:

* a zero-offset LEAF can be deleted outright when nothing hangs off it and no
  rotation is lost with it. What that means depends on the feature layout,
  and the two are not interchangeable:

  - `rot6d_joint` stores each joint's own rotation, so a leaf may go only if
    it has none: an End Site (a leaf of the FULL rig, which a BVH gives no
    channels). A real joint that becomes a leaf once its End Site is dropped
    keeps its own rotation, and that rotation drives skin (Bear's
    `NPC_*Arm2_ForearmTwist2`, Scorpion's `Bip01_Neck1`), so it is kept;
  - `rot6d_bone` stores each joint's rotation in its children's slots, so a
    leaf that is its parent's only child carries the parent's rotation and is
    kept. A real joint therefore never becomes a leaf, and never drops;
* a zero-offset INTERNAL joint swings its subtree, so it can only be folded into
  its parent -- and that fold turns the parent, hence every OTHER child of the
  parent too. It is exact only when the joint is its parent's sole child.
  Otherwise the joint is kept, because displacing its siblings to save one token
  is not a trade worth making. The fold is world-exact for every joint but the
  parent, which comes back carrying the product while the joint gets identity:
  the PARENT's world rotation is lost. So it is done only into the root, whose
  rotation drives no skin (Truebones' `Hips`: zero weighted vertices, with
  `Bip01_Pelvis`, the joint folded, carrying the pelvis skin and coming back
  exact). Anywhere else the joint is kept. The root fold is wanted, not just
  tolerated: some rigs (the 46 Biped ones on Truebones) carry an extra root,
  `Hips` above a coincident `Bip01_Pelvis`, and folding it gives every
  skeleton the same one-joint root (the owner's rationale).

A third case is measured on the data rather than read off the geometry: a
STILL subtree, a real joint whose rotation never changes over any frame of
any prepared clip of its rig, with every joint below it (3ds Max `*Nub` tip
bones, never-animated lips, eyes and muscles, Rat's `Trajectory` ground
marker). It is dropped leaf by leaf, whatever its offset, and comes back at its
constant rotation and its offset, so on the corpus it is exact; a generated
clip cannot move it, which no clip does either. `still_joints` measures it at
stage 2, under the manifest's `still_joint_degrees`, and `skeleton.npz`
records it, since a reduction rebuilt from geometry alone cannot see it.
Under `rot6d_bone` a still leaf that is a moving parent's only child is kept,
as above, since its slot holds that parent's rotation.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import ClassVar

import numpy as np

from poseydon.augment.joint_edit import JointEdit
from poseydon.core.animation import RigidBodyAnimation
from poseydon.core.registry import Registry
from poseydon.core.rotations import QUAT_IDENTITY, quat_mul
from poseydon.core.spec import ROTATION_BLOCKS

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
    #: Put back with these: the removed joint's offset, and its constant
    #: rotation when it is a still joint (identity otherwise).
    offset: tuple[float, ...] = (0.0, 0.0, 0.0)
    rotation: tuple[float, ...] = tuple(QUAT_IDENTITY)

    @property
    def operation_id(self) -> str:
        return f"reduction/{self.kind}/{self.name}"

    @property
    def recipient(self) -> str | None:
        return self.parent if self.kind == COLLAPSE else None

    @property
    def reason(self) -> str:
        return "zero-offset or still leaf" if self.kind == DROP else "zero-offset sole-child chain"


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


class ReduceRule(ABC):
    """One way a joint can leave the model view. Rules are tried in the order a
    dataset lists them (``reduce:``), joint by joint; the first that removes a
    joint wins, and the hierarchy is re-read after every removal. A concrete
    subclass declaring ``name`` registers itself in `REDUCE` under it."""

    name: ClassVar[str]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "name" in cls.__dict__:
            REDUCE.register(cls.name)(cls)

    @abstractmethod
    def removal(self, anim: RigidBodyAnimation, joint: int, layout: str,
                end_sites: frozenset[str], still: Mapping[str, np.ndarray]) -> RemovalOp | None:
        """The removal of ``joint`` this rule makes, or None."""


#: Every reduce rule, by name (`ReduceRule`).
REDUCE: Registry[ReduceRule] = Registry("reduce rule")


@dataclass(frozen=True)
class DropStill(ReduceRule):
    """A still leaf (`still_joints`, measured at stage 2 under ``degrees``), put
    back at its constant rotation. Under `rot6d_bone` a still leaf that is a
    moving parent's only child is kept: its slot holds that parent's rotation."""

    degrees: float = 0.5
    name: ClassVar[str] = "drop_still"

    def removal(self, anim, joint, layout, end_sites, still):
        parent = int(anim.parents[joint])
        name = anim.names[joint]
        if _children_of(anim.parents, joint) or name not in still:
            return None
        if layout == "rot6d_bone" and anim.names[parent] not in still and len(_children_of(anim.parents, parent)) == 1:
            return None
        return RemovalOp(name, anim.names[parent], joint, DROP,
                         offset=tuple(anim.offsets[joint]), rotation=tuple(still[name]))


@dataclass(frozen=True)
class DropEndSites(ReduceRule):
    """A zero-offset leaf that carries no rotation of its own: under
    `rot6d_joint` an End Site of the full rig; under `rot6d_bone` any such leaf
    except a parent's only child, whose slot holds the parent's rotation."""

    tolerance: float = 1e-8
    name: ClassVar[str] = "drop_end_sites"

    def removal(self, anim, joint, layout, end_sites, still):
        if _children_of(anim.parents, joint) or np.linalg.norm(anim.offsets[joint]) >= self.tolerance:
            return None
        parent = int(anim.parents[joint])
        # `rot6d_bone` stores a joint's rotation in its children's slots,
        # so a leaf that is its parent's only child holds the only copy of
        # the parent's rotation (an ear's, a tail tip's): keep it.
        if layout == "rot6d_bone" and len(_children_of(anim.parents, parent)) == 1:
            return None
        # `rot6d_joint` keeps a joint's own rotation in its own slot: only
        # an End Site has none to lose.
        if layout == "rot6d_joint" and anim.names[joint] not in end_sites:
            return None
        return RemovalOp(anim.names[joint], anim.names[parent], joint, DROP, offset=tuple(anim.offsets[joint]))


@dataclass(frozen=True)
class FoldIntoRoot(ReduceRule):
    """A zero-offset internal joint that is the root's only child, folded into
    the root (the fold loses the parent's own world rotation, so the root only)."""

    tolerance: float = 1e-8
    name: ClassVar[str] = "fold_into_root"

    def removal(self, anim, joint, layout, end_sites, still):
        children = _children_of(anim.parents, joint)
        parent = int(anim.parents[joint])
        if not children or np.linalg.norm(anim.offsets[joint]) >= self.tolerance:
            return None
        if anim.parents[parent] >= 0 or len(_children_of(anim.parents, parent)) != 1:
            return None
        return RemovalOp(anim.names[joint], anim.names[parent], joint, COLLAPSE,
                         tuple(anim.names[c] for c in children), offset=tuple(anim.offsets[joint]))


#: The rules and order every Truebones config lists; what a caller with no
#: recorded rules (a probe, a synthetic rig) reduces by.
DEFAULT_RULES: tuple[ReduceRule, ...] = (DropStill(), DropEndSites(), FoldIntoRoot())


def rules_from(entries) -> tuple[ReduceRule, ...]:
    """The rules a config's ``reduce:`` list (or `skeleton.npz`'s record) names."""
    return tuple(REDUCE.build(entry) for entry in entries)


def rule_entries(rules: Sequence[ReduceRule]) -> list[dict]:
    """``rules`` as config entries, for recording beside the reduction."""
    return [{"type": rule.name, **asdict(rule)} for rule in rules]


def _next_removal(
    anim: RigidBodyAnimation,
    rules: Sequence[ReduceRule],
    layout: str,
    end_sites: frozenset[str],
    still: Mapping[str, np.ndarray],
) -> RemovalOp | None:
    """The first removable joint, or None when the rig is fully reduced.

    ``end_sites`` are the leaves of the rig the reduction started from; see the
    module docstring for why `rot6d_joint` drops only those. ``still`` maps the
    still joints to their constant rotation (`still_joints`).
    """
    for joint in range(1, anim.n_joints):
        for rule in rules:
            op = rule.removal(anim, joint, layout, end_sites, still)
            if op is not None:
                return op
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
    anim: RigidBodyAnimation,
    rules: Sequence[ReduceRule] = DEFAULT_RULES,
    layout: str = "rot6d_joint",
    still: Mapping[str, np.ndarray] | None = None,
) -> JointReduction:
    """Work out which joints to remove, repeating until none is left.

    Repetition matters: dropping a zero-offset leaf can leave its parent a
    zero-offset leaf in turn.

    ``rules`` are tried in order (`ReduceRule`). ``layout`` is the rotation block the features use (`rot6d_joint` or
    `rot6d_bone`, see `reduction_layout`): what a dropped leaf takes with it
    depends on where rotations are stored. ``still`` is `still_joints`' result,
    as `skeleton.npz` records it (`load_still`); without it no still joint is
    dropped.
    """
    still = dict(still or {})
    ops: list[RemovalOp] = []
    current = anim
    # Re-derived every iteration, deliberately: dropping a zero-offset leaf can
    # leave its parent a zero-offset leaf in turn, and that parent is then
    # droppable by case 1 even though it began as an internal joint with
    # siblings. Scorpion/Bip01_Neck1 is exactly this shape. Testing the
    # CURRENT hierarchy ensures we capture joints as they actually become
    # droppable, not as they began.
    # The FULL rig's leaves: its End Sites, which carry no rotation of their own.
    end_sites = frozenset(
        name for j, name in enumerate(anim.names) if not _children_of(anim.parents, j)
    )
    while (op := _next_removal(current, rules, layout, end_sites, still)) is not None:
        ops.append(op)
        current = _remove(current, op)

    index_of = {name: i for i, name in enumerate(anim.names)}
    return JointReduction(
        ops=tuple(ops),
        source_of=tuple(index_of[name] for name in current.names),
        source_names=tuple(anim.names),
    )


def reduction_layout(features: Iterable[str]) -> str:
    """The rotation layout a feature list implies for the reduction.

    A list with no rotation block (positions only) reduces as `rot6d_joint`:
    no rotation is stored, so none can be lost.
    """
    present = [name for name in features if name in ROTATION_BLOCKS]
    return present[0] if present else "rot6d_joint"


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
            rotations[:, new] = op.rotation
            offsets[new] = op.offset
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


def still_joints(
    anims: Sequence[RigidBodyAnimation],
    degrees: float | None,
    protected: Iterable[str] = (),
) -> dict[str, np.ndarray]:
    """The joints that never rotate, each with its constant rotation.

    A joint qualifies when its local rotation stays within ``degrees`` of the
    first clip's first frame over every frame of every clip in ``anims`` (one
    rig's prepared clips), and every joint below it does too. Only subtrees
    under a real joint count: an End Site always qualifies, and dropping every
    End Site with an offset would remove each limb's tip, not a joint nobody
    animates. The root never counts, nor does a subtree holding a
    ``protected`` joint: the manifest's feet and facing pairs, which features
    read by name (a toe's `*Nub` tip is where foot contact is measured on
    many Truebones rigs). ``degrees=None`` returns nothing.
    """
    protected = set(protected)
    if degrees is None or not anims:
        return {}
    first = anims[0]
    reference = first.rotations[0]
    turned = np.zeros(first.n_joints)
    for anim in anims:
        dot = np.abs((anim.rotations * reference).sum(-1)).clip(0.0, 1.0)
        turned = np.maximum(turned, 2.0 * np.arccos(dot).max(axis=0))
    steady = turned <= np.radians(degrees)
    children = [_children_of(first.parents, joint) for joint in range(first.n_joints)]

    def subtree(joint: int) -> list[int]:
        return [joint] + [j for child in children[joint] for j in subtree(child)]

    keep: set[int] = set()
    for joint in range(1, first.n_joints):
        below = subtree(joint)
        if children[joint] and all(steady[j] and first.names[j] not in protected for j in below):
            keep.update(below)
    return {first.names[j]: reference[j] for j in sorted(keep)}


def load_still(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """`still_joints`' result as `skeleton.npz` stores it; empty for a file
    written before the still-joint drop existed."""
    if "reduction_still_names" not in data:
        return {}
    names = [str(n) for n in data["reduction_still_names"]]
    return dict(zip(names, np.asarray(data["reduction_still_rotations"]).reshape(len(names), 4)))


def load_rules(data: Mapping[str, np.ndarray]) -> tuple[ReduceRule, ...]:
    """The reduce rules `skeleton.npz` records (the dataset's ``reduce:`` when
    stage 2 ran), so every later reduction of the rig uses the same ones."""
    if "reduction_rules" not in data:
        raise ValueError("this skeleton.npz records no reduce rules (made before 2026-10-19): rebuild stage 2")
    return rules_from(json.loads(str(data["reduction_rules"])))
