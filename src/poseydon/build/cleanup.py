"""Rig cleanup stages: preparation that removes joints from a rig, recorded.

Kept apart from `poseydon.build.prepare`'s geometry stages because the decision
is about the rig, not one clip: it needs every clip of the rig and its skin
weights (`RigEvidence`), and it changes the joint set the whole corpus is built
on. It is not the model's lens (`reduce:` in the config, read at feature time):
what is removed here is gone from the prepared files and is only restored, at
its bind pose, by `invert`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from poseydon.build.prepare import RIG, PrepareStage, RigEvidence
from poseydon.core.animation import Animation
from poseydon.core.rotations import quat_apply, quat_inverse, quat_mul

# UniMate's naming convention for Blender rig mechanism bones
# (`name_is_control_bone` in data_process/utils/blender_export.py).
_CONTROL_PREFIXES = ("MCH-", "ORG-", "VIS-")
_CONTROL_SUFFIXES = ("_ik", ".ik", "_pole", ".pole", "_target", "_tip", "_ctrl", ".ctrl", "_tweak", "_helper")


def _is_control_name(name: str) -> bool:
    return name.startswith(_CONTROL_PREFIXES) or name.lower().endswith(_CONTROL_SUFFIXES)


def _children(parents: np.ndarray) -> list[list[int]]:
    out: list[list[int]] = [[] for _ in parents]
    for j, p in enumerate(parents):
        if p >= 0:
            out[p].append(j)
    return out


def _remove(anim: Animation, j: int, bind_rotation: np.ndarray) -> Animation:
    """``anim`` without joint ``j``. A leaf is dropped; a joint with one child
    (the root included) gives that child its transform, so the child and its
    subtree keep their world pose. Offsets compose through ``bind_rotation``,
    ``j``'s rest local rotation, and rotations and translations frame by frame."""
    parents, rot = anim.parents.copy(), anim.rotations.copy()
    tr, off = anim.translations.copy(), anim.offsets.copy()
    kids = np.flatnonzero(parents == j)
    if len(kids):
        c = int(kids[0])
        off[c] = off[j] + quat_apply(bind_rotation, off[c])
        tr[:, c] = tr[:, j] + quat_apply(rot[:, j], tr[:, c])
        rot[:, c] = quat_mul(rot[:, j], rot[:, c])
        parents[c] = parents[j]
    keep = [i for i in range(anim.n_joints) if i != j]
    new = {old: i for i, old in enumerate(keep)}
    return Animation(
        rot[:, keep], tr[:, keep], off[keep],
        np.array([new[p] if p >= 0 else -1 for p in parents[keep]], dtype=np.int32),
        tuple(anim.names[i] for i in keep), anim.fps,
    )


def _amplitude(rot: np.ndarray, j: int) -> float:
    """Largest angle (radians) joint ``j`` turns from its first frame."""
    w = quat_mul(quat_inverse(rot[0:1, j]), rot[:, j])[..., 3]
    return float(np.max(2.0 * np.arccos(np.abs(w).clip(0.0, 1.0))))


@dataclass(frozen=True)
class PruneUnskinned(PrepareStage):
    """Remove the joints that are neither skinned nor moving, across all of a rig's clips.

    UniMate's `prune_skeleton_shared` (data_process/utils/blender_export.py),
    ported as it decides: three passes repeated until nothing changes, stopping a
    leaf pass that would leave fewer than ``min_joints``.

    1. Leaves, all at once: a leaf with no skin weight (``skin_eps``) goes if its
       name is a control name, its rest offset is under ``offset_eps`` of the mean
       bone length, it never leaves the parent's origin, it moves more than
       ``pos_var_eps`` from its rest offset (an IK stretch), or it never turns
       ``rot_eps_deg`` -- unless its parent does (``consider_parent_rotate``: a
       still end effector on a moving bone is kept).
    2. An unskinned root with one child folds into the child.
    3. Pass-through joints (one child, same tests minus the leaf-only ones) merge
       into the child, deepest first, one per round.

    The tests read every clip, which is why this is a rig-scope stage with
    ``uses_evidence``. Values are UniMate's; rationale not recorded for them.

    `invert` brings each removed joint back at its bind pose: its rest local
    translation as offset, its rest local rotation, motion not stored. A merged
    joint's child takes ``bind_rotation⁻¹ · composite``, so it and its subtree
    land exactly where they were; a folded root is reinserted above its child the
    same way, taking the child's trajectory. A source with no skin (a BVH) cannot
    be fitted.
    """

    offset_eps: float = 1e-3
    pos_var_eps: float = 1e-3
    rot_eps_deg: float = 2.0
    min_joints: int = 4
    skin_eps: float = 1e-5
    consider_parent_rotate: bool = True

    name: ClassVar[str] = "prune_unskinned"
    after: ClassVar[tuple[str, ...]] = ("up_axis",)
    scope: ClassVar[str] = RIG
    uses_evidence: ClassVar[bool] = True

    def fit_evidence(self, anim: Animation, resolved: Any, evidence: RigEvidence) -> dict[str, np.ndarray]:
        if evidence.skin is None:
            raise ValueError("`prune_unskinned` needs skin weights, and this source has none (a BVH)")
        if not evidence.clips:
            raise ValueError("`prune_unskinned` decides from the rig's clips, and none were given")
        skin = evidence.skin
        if not np.allclose(anim.offsets, anim.translations[0]):
            # `invert` recovers a merged child's offset from `bind_offset`, which is
            # the rest translation; that only equals the offset on such a rig.
            raise ValueError("`prune_unskinned` needs a rest pose whose translations equal its offsets")
        rest, clips = anim, list(evidence.clips)
        source_names, source_parents = np.array(anim.names), np.asarray(anim.parents, dtype=np.int64)
        log: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        rot_eps = np.deg2rad(self.rot_eps_deg)

        def remove(name: str, kind: str) -> None:
            nonlocal rest, clips
            j = rest.names.index(name)
            q = rest.rotations[0, j].copy()
            log.append((name, kind, rest.translations[0, j].copy(), q))
            rest = _remove(rest, j, q)
            clips = [_remove(c, j, q) for c in clips]

        def mean_bone_len() -> float:
            lengths = np.linalg.norm(rest.offsets[1:], axis=-1)
            lengths = lengths[lengths > 1e-6]
            return float(lengths.mean()) if len(lengths) else 1.0

        def prunable(j: int, mbl: float, is_leaf: bool) -> bool:
            name = rest.names[j]
            if skin.get(name, 0.0) >= self.skin_eps:
                return False
            if _is_control_name(name):
                return True
            norm = max(mbl, 1e-9)
            rest_pos = rest.translations[0, j]
            if np.linalg.norm(rest_pos) / norm < self.offset_eps:
                return True
            if all(np.max(np.linalg.norm(c.translations[:, j], axis=-1)) / norm < self.offset_eps for c in clips):
                return True
            if is_leaf and any(
                np.max(np.linalg.norm(c.translations[:, j] - rest_pos, axis=-1)) / norm > self.pos_var_eps
                for c in clips
            ):
                return True
            if rot_eps > 0.0 and all(_amplitude(c.rotations, j) < rot_eps for c in clips):
                if is_leaf and self.consider_parent_rotate:
                    parent = int(rest.parents[j])
                    if parent >= 0 and any(_amplitude(c.rotations, parent) >= rot_eps for c in clips):
                        return False
                return True
            return False

        def leaves() -> int:
            count, mbl = 0, mean_bone_len()
            while True:
                kids = _children(rest.parents)
                found = [j for j in range(rest.n_joints) if not kids[j] and prunable(j, mbl, True)]
                if not found or rest.n_joints - len(found) < self.min_joints:
                    return count
                for name in [rest.names[j] for j in found]:
                    remove(name, "leaf")
                count += len(found)

        def root() -> int:
            count = 0
            while True:
                kids = _children(rest.parents)[0]
                if len(kids) != 1 or skin.get(rest.names[0], 0.0) >= self.skin_eps:
                    return count
                remove(rest.names[0], "root")
                count += 1

        def passthrough() -> int:
            count, mbl = 0, mean_bone_len()
            while True:
                kids = _children(rest.parents)
                found = [j for j in range(1, rest.n_joints) if len(kids[j]) == 1 and prunable(j, mbl, False)]
                if not found:
                    return count
                j, chain = found[0], set(found)
                while kids[j][0] in chain:   # deepest first
                    j = kids[j][0]
                remove(rest.names[j], "passthrough")
                count += 1

        while rest.n_joints > self.min_joints:
            if leaves() + root() + passthrough() == 0:
                break

        return {
            "source_names": source_names,
            "source_parents": source_parents,
            "removed": np.array([n for n, *_ in log]) if log else np.empty(0, dtype="<U1"),
            "kind": np.array([k for _, k, *_ in log]) if log else np.empty(0, dtype="<U1"),
            "bind_offset": np.array([t for _, _, t, _ in log]).reshape(-1, 3),
            "bind_rotation": np.array([q for *_, q in log]).reshape(-1, 4),
        }

    def apply(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        if tuple(anim.names) != tuple(str(n) for n in params["source_names"]):
            raise ValueError(
                "this animation's joints differ from the rig `prune_unskinned` was "
                f"fitted to ({anim.n_joints} vs {len(params['source_names'])} joints)"
            )
        for name, q in zip(params["removed"], params["bind_rotation"]):
            anim = _remove(anim, anim.names.index(str(name)), q)
        return anim

    def invert(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        source = [str(n) for n in params["source_names"]]
        parent = {n: (source[p] if p >= 0 else None) for n, p in zip(source, params["source_parents"])}
        removed = [str(n) for n in params["removed"]]
        # Replay the removals on names alone: who each joint's parent and child
        # were when it went.
        steps = []
        for name in removed:
            kids = [n for n in parent if parent[n] == name]
            steps.append((dict(parent), kids[0] if kids else None))
            for n in kids:
                parent[n] = parent[name]
            del parent[name]
        for i in reversed(range(len(removed))):
            name, kind, (before, kid) = removed[i], str(params["kind"][i]), steps[i]
            t_b, q_b = params["bind_offset"][i], params["bind_rotation"][i]
            names = tuple(n for n in source if n in before)
            post = {n: k for k, n in enumerate(anim.names)}
            n_frames = anim.n_frames
            rot, tr = np.empty((n_frames, len(names), 4)), np.empty((n_frames, len(names), 3))
            off = np.empty((len(names), 3))
            for k, n in enumerate(names):
                if n == name:
                    rot[:, k], tr[:, k], off[k] = q_b, t_b, t_b
                    continue
                s = post[n]
                rot[:, k], tr[:, k], off[k] = anim.rotations[:, s], anim.translations[:, s], anim.offsets[s]
                if n == kid:
                    inv = quat_inverse(q_b)
                    rot[:, k] = quat_mul(inv, anim.rotations[:, s])
                    off[k] = quat_apply(inv, anim.offsets[s] - t_b)
                    if kind == "root":
                        # the child is the root and holds the trajectory; the
                        # folded root above it follows, the child stays at its offset
                        tr[:, names.index(name)] = anim.translations[:, s] - (anim.offsets[s] - t_b)
                        tr[:, k] = off[k]
                    else:
                        tr[:, k] = quat_apply(inv, anim.translations[:, s] - t_b)
            anim = Animation(
                rot, tr, off,
                np.array([names.index(before[n]) if before[n] is not None else -1 for n in names], dtype=np.int32),
                names, anim.fps,
            )
        return anim


@dataclass(frozen=True)
class BfsOrder(PrepareStage):
    """Renumber a rig's joints in breadth-first order from the root.

    UniMate's `bfs_reorder_joints` (data_process/utils/skeleton.py): children are
    visited by subtree size, largest first, then by rest offset length, shortest
    first, so two rigs of the same shape list their limbs alike. Parents still
    precede children. `invert` restores the source order exactly; the permutation
    is the stored parameter, with the BFS joint names so a BVH's depth-first
    re-listing is undone by name.
    """

    name: ClassVar[str] = "bfs_order"
    after: ClassVar[tuple[str, ...]] = ("prune_unskinned",)
    scope: ClassVar[str] = RIG

    def fit(self, anim: Animation, resolved: Any) -> dict[str, np.ndarray]:
        kids = _children(anim.parents)
        size = np.ones(anim.n_joints, dtype=np.int64)
        for j in range(anim.n_joints - 1, 0, -1):   # parents precede children
            size[anim.parents[j]] += size[j]
        length = np.linalg.norm(anim.offsets, axis=-1)
        order, queue = [], [0]
        for j in queue:
            order.append(j)
            queue.extend(sorted(kids[j], key=lambda c: (-size[c], length[c])))
        return {"order": np.array(order, dtype=np.int64), "names": np.array([anim.names[j] for j in order])}

    @staticmethod
    def _reorder(anim: Animation, order: np.ndarray) -> Animation:
        """``anim`` with joint ``order[k]`` at position ``k``; parents follow."""
        new = np.argsort(order)
        old = anim.parents[order]
        return type(anim)(
            anim.rotations[:, order], anim.translations[:, order], anim.offsets[order],
            np.where(old >= 0, new[np.maximum(old, 0)], -1).astype(np.int32),
            tuple(anim.names[j] for j in order), anim.fps,
        )

    def apply(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        return self._reorder(anim, params["order"])

    def invert(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        # A BVH lists joints depth-first whatever order they were written in, so a
        # prepared file read back is first put into the stored BFS order by name.
        bfs = [str(n) for n in params["names"]]
        if list(anim.names) != bfs:
            anim = self._reorder(anim, np.array([anim.names.index(n) for n in bfs]))
        return self._reorder(anim, np.argsort(params["order"]))
