"""Loss laws: what each lossy stage of the pipeline is defined to do to a clip's joints.

Written here from each stage's stated definition, on world matrices, and never
imported from `poseydon.build` or `poseydon.features`: running the pipeline's own code
would make the reference circular. `expected_pose` is the source clip with every lossy
stage of a rig applied; the pipeline's file must equal it (docs/reference/tooling.md).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class SourcePose:
    """Joints in the user's rig: ``world`` is ``(F, J, 4, 4)``, ``rest`` ``(J, 4, 4)`` the dataset's rest pose
    (bone offsets, bind rotations). ``asset`` is the pose of the FBX the file is written onto (its node scale,
    and the transform of the ``held`` joints); it defaults to ``rest``, which it equals where the dataset's
    rest pose is that asset."""

    names: tuple[str, ...]
    parents: np.ndarray
    world: np.ndarray
    rest: np.ndarray
    held: tuple[str, ...] = ()
    asset: np.ndarray | None = None

    @property
    def placed(self) -> np.ndarray:
        return self.rest if self.asset is None else self.asset


@dataclass
class MissingLaw:
    stage: str


#: Stages that move, turn, scale or re-label the whole clip and are inverted to float precision.
LOSSLESS = frozenset({
    "up_axis", "rest_relative", "bfs_order", "face_rest_axis", "face_axis", "centre_xz", "centre_xz_clip",
    "scale", "ground", "ground_clip", "bone_relative",
})


def _scale(m: np.ndarray) -> np.ndarray:
    return np.linalg.norm(m[..., :3, :3], axis=-2)


def _rot(m: np.ndarray) -> np.ndarray:
    return m[..., :3, :3] / _scale(m)[..., None, :]


def _turn(pose: SourcePose, j: int, rotation: np.ndarray) -> None:
    pose.world[:, j, :3, :3] = rotation * _scale(pose.world[:, j])[:, None, :]


def _rest_bind(pose: SourcePose, j: int, p: int) -> np.ndarray:
    """``j``'s rest rotation relative to its parent's."""
    return _rot(pose.rest[p]).T @ _rot(pose.rest[j]) if p >= 0 else _rot(pose.rest[j])


def promote_root(pose: SourcePose, params: dict) -> SourcePose:
    """The locator chain above the promoted root and its dead branches are not in the model: the file
    holds them at the asset's own transform. ``params``: ``chain``, ``dead`` (joint names)."""
    return replace(pose, held=(*pose.held, *params["chain"], *params["dead"]))


def enforce_rigid(pose: SourcePose, params: dict) -> SourcePose:
    """Constant bone lengths, only the root moves: a driven joint keeps its rotation, takes the
    asset's world scale, and sits at ``p_parent + R_parent o`` with ``o`` the rest offset in the
    parent's rest frame; a joint with no driven parent keeps its position. A held joint is its
    parent's world times its rest local transform. Runs last, on the rotations the other laws left."""
    world, scale = pose.world.copy(), _scale(pose.placed)
    for j, name in enumerate(pose.names):
        p = int(pose.parents[j])
        if name in pose.held:
            at = pose.placed
            local = at[j] if p < 0 else np.linalg.inv(at[p]) @ at[j]
            world[:, j] = local if p < 0 else world[:, p] @ local
            continue
        rotation = _rot(world[:, j])
        if p >= 0 and pose.names[p] not in pose.held:
            offset = _rot(pose.rest[p]).T @ (pose.rest[j, :3, 3] - pose.rest[p, :3, 3])
            world[:, j, :3, 3] = world[:, p, :3, 3] + _rot(world[:, p]) @ offset
        world[:, j, :3, :3] = rotation * scale[j]
    return replace(pose, world=world)


def prune_unskinned(pose: SourcePose, params: dict) -> SourcePose:
    """A joint removed as unskinned and not moving comes back at its bind: its rest rotation relative to
    its parent. A removed root (``kind`` ``root``) is held at the asset's transform instead.
    ``params``: ``removed``, ``kind``."""
    kind = dict(zip(params["removed"], params["kind"]))
    out = replace(pose, world=pose.world.copy(), held=(*pose.held, *(n for n, k in kind.items() if k == "root")))
    for j, name in enumerate(pose.names):
        if kind.get(name, "root") != "root":
            p = int(pose.parents[j])
            _turn(out, j, (_rot(out.world[:, p]) if p >= 0 else np.eye(3)) @ _rest_bind(pose, j, p))
    return out


def drop_still(pose: SourcePose, params: dict) -> SourcePose:
    """A joint that never turned is held at its recorded constant rotation ``q`` (relative to rest:
    identity is the rest pose): world ``G_j = G_p B_p^T q B_j``, ``B`` the rest world rotations.
    ``params``: ``names``, ``rotations`` (xyzw)."""
    q = dict(zip(params["names"], Rotation.from_quat(np.asarray(params["rotations"]).reshape(-1, 4)).as_matrix()))
    out = replace(pose, world=pose.world.copy())
    for j, name in enumerate(pose.names):
        if name in q:
            p = int(pose.parents[j])
            above = _rot(out.world[:, p]) @ _rot(pose.rest[p]).T if p >= 0 else np.eye(3)
            _turn(out, j, above @ q[name] @ _rot(pose.rest[j]))
    return out


def drop_end_sites(pose: SourcePose, params: dict) -> SourcePose:
    """A zero-offset leaf with no rotation of its own leaves the model and changes nothing."""
    return pose


def fold_into_root(pose: SourcePose, params: dict) -> SourcePose:
    """A zero-offset sole child of the root folds into it: the root takes the child's world rotation,
    the child's children are unchanged. ``params``: ``names`` (the folded joints), ``parents``."""
    out = replace(pose, world=pose.world.copy())
    for name, parent in zip(params["names"], params["parents"]):
        _turn(out, pose.names.index(parent), _rot(pose.world[:, pose.names.index(name)]))
    return out


#: A law per lossy stage, keyed by the stage's (or reduction rule's) name.
LAWS: dict[str, Callable[[SourcePose, dict], SourcePose]] = {
    "promote_root": promote_root, "enforce_rigid": enforce_rigid, "prune_unskinned": prune_unskinned,
    "drop_still": drop_still, "drop_end_sites": drop_end_sites, "fold_into_root": fold_into_root,
}


def expected_pose(pose: SourcePose, stages: Sequence[str], params: dict) -> SourcePose | MissingLaw:
    """``pose`` with the law of each of ``stages`` applied, or the first stage with neither a law nor a
    lossless declaration. ``enforce_rigid`` goes last: it rebuilds positions from the final rotations."""
    for stage in stages:
        if stage not in LAWS and stage not in LOSSLESS:
            return MissingLaw(stage)
    ordered = [s for s in stages if s != "enforce_rigid"] + [s for s in stages if s == "enforce_rigid"]
    for stage in ordered:
        if stage in LAWS:
            pose = LAWS[stage](pose, params.get(stage, {}))
    return pose
