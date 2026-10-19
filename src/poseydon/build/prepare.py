"""Invertible preparation stages.

A prepared corpus is only useful to an application if the transform that
produced it can be undone: a user hands over a rig, the model generates on the
canonical one, and the result has to come back on theirs. So every stage here
declares an ``invert`` beside its ``apply``, and the parameters it fitted are
persisted rather than recomputed -- the facing rotation in particular is derived
from a clip's own frame 0 and is unrecoverable once the file already faces +Z.

Stages are fitted at one of two scopes. A ``"rig"`` stage fits once, against the
rest pose, and every clip of that character reuses the result; fitting per clip
would ground a flying creature onto the floor and destroy the height difference
between a crouch and a stand. A ``"clip"`` stage fits per animation.
"""

from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from poseydon.build.corpus import corpus_root, rig_cache
from poseydon.core.animation import Animation, RigidBodyAnimation, rest_geometry
from poseydon.core.registry import Registry
from poseydon.core.rotations import (
    QUAT_IDENTITY,
    matrix_to_quat,
    quat_apply,
    quat_between,
    quat_inverse,
    quat_mul,
)
from poseydon.core.skeleton import HML_MEAN_BONE_LENGTH, resolve
from poseydon.ingest.align import axis_vector, facing_quats, rotate_rig, rotate_to_face_z

if TYPE_CHECKING:
    from poseydon.build.report import AuditSink

RIG = "rig"
CLIP = "clip"


#: Every preparation stage, by the name its parameters are stored under. A stage
#: joins by subclassing `PrepareStage` with a ``name`` (see `PrepareStage`), so a
#: user's own stage needs only its module in the config's ``custom_imports``.
PREPARE: Registry[PrepareStage] = Registry("prepare stage")


def chain_from(entries) -> PrepareChain:
    """The chain a config's ``prepare:`` list describes, in its order."""
    return PrepareChain(tuple(PREPARE.build(entry) for entry in entries), tuple(_entry(e) for e in entries))


def _entry(entry) -> dict:
    from omegaconf import OmegaConf

    if OmegaConf.is_config(entry):
        return OmegaConf.to_container(entry, resolve=True)
    return {"type": entry} if isinstance(entry, str) else dict(entry)


@dataclass(frozen=True)
class RigEvidence:
    """What a rig-wide decision looks at beyond the rest pose: the rig's clips and
    its skin weights. `PrepareChain.fit_rig` hands each stage that
    ``uses_evidence`` the clips as the preceding stages left them.

    ``skin`` is the largest skin weight per joint name on the rig's main mesh;
    ``None`` when the source carries none (a BVH).
    """

    clips: tuple[Animation, ...] = ()
    skin: Mapping[str, float] | None = None


class PrepareStage(ABC):
    """One invertible step of the source-to-prepared transform.

    A concrete subclass declaring ``name`` registers itself in `PREPARE` under
    that name, which is also the key its fitted parameters are stored under and
    the ``type:`` a config names it by.
    """

    #: Key under which this stage's fitted parameters are stored.
    name: ClassVar[str]
    #: ``RIG`` to fit once against the rest pose, ``CLIP`` to fit per animation.
    scope: ClassVar[str] = RIG
    #: Stages that must run before this one when the chain has them (the order
    #: check in `PrepareChain`); ``"*"`` means every other stage.
    after: ClassVar[tuple[str, ...]] = ()
    #: ``True`` for a stage that decides from the rig's clips and skin
    #: (`RigEvidence`) and so is fitted by `fit_evidence`, not `fit`.
    uses_evidence: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "name" in cls.__dict__:
            PREPARE.register(cls.name)(cls)

    def fit(self, anim: Animation, resolved: Any) -> dict[str, np.ndarray]:
        """Parameters derived from ``anim`` as it stands at this point in the chain."""
        return {}

    def fit_evidence(self, anim: Animation, resolved: Any, evidence: RigEvidence) -> dict[str, np.ndarray]:
        """`fit`, for a stage with ``uses_evidence``: ``anim`` is the rest pose."""
        return self.fit(anim, resolved)

    @abstractmethod
    def apply(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation: ...

    @abstractmethod
    def invert(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation: ...


@dataclass(frozen=True)
class PrepareChain:
    """An ordered composition of stages, invertible as a whole.

    The order is contractual, so it is checked here rather than trusted: each
    stage declares in ``after`` the stages that must precede it, and a chain
    breaking one is refused before anything runs. ``entries`` are the config
    entries the chain was built from (`chain_from`); `RigTransform` stores them
    in ``preproc.npz``, so a corpus is inverted with the chain it was built
    with, not whatever the config says today.
    """

    stages: tuple[PrepareStage, ...]
    entries: tuple[dict, ...] = ()

    def __post_init__(self) -> None:
        order = [stage.name for stage in self.stages]
        if len(set(order)) != len(order):
            raise ValueError(f"a prepare chain lists a stage twice: {order}")
        for i, stage in enumerate(self.stages):
            wanted = [n for n in order if n != stage.name] if "*" in stage.after else stage.after
            late = [n for n in wanted if n in order[i + 1:]]
            if late:
                raise ValueError(f"prepare stage `{stage.name}` must come after {late}; the chain is {order}")

    def fit_rig(
        self, rest: Animation, resolved: Any, audit: AuditSink | None = None,
        evidence: RigEvidence = RigEvidence(),
    ) -> dict[str, dict]:
        """Fit every stage against the rest pose.

        Each stage fits against the animation as the PRECEDING stages left it,
        which is why this applies as it goes: the scale factor is measured on an
        already-rest-relative skeleton, and the ground height on an already
        scaled one. ``evidence.clips`` go through the same stages (a clip-scoped
        one refitted per clip) up to the last stage that ``uses_evidence``, so
        that stage sees them as it will see a clip.

        CLIP-scoped stages are fitted here too, and their results are kept
        rather than discarded. A clip refits them anyway -- `apply` overwrites
        every clip-scoped key -- so keeping them costs nothing there, and it is
        the only record of what the stage would say about the rig ITSELF. That
        matters for `face_axis`: a GENERATED clip has no historical yaw to put
        back, and the rig's own rest-pose facing is the one answer that leaves
        the restored motion in the same frame as the source-frame constants
        `PromoteRoot.invert` and `RestRelative.invert` stamp back. Substituting
        identity there does not merely leave the rig unturned; it lands the
        body in canonical +Z while those constants stay in the source frame,
        which displaces the reinserted locator chain by up to 1.7 rig extents.
        """
        params: dict[str, dict] = {}
        current = rest
        clips = evidence.clips
        last = max((i for i, s in enumerate(self.stages) if s.uses_evidence), default=-1)
        for i, stage in enumerate(self.stages):
            bound = _rebind(resolved, rest, current)
            if stage.uses_evidence:
                params[stage.name] = stage.fit_evidence(current, bound, RigEvidence(clips, evidence.skin))
            else:
                params[stage.name] = stage.fit(current, bound)
            if i < last:
                clips = tuple(self._through(stage, clip, resolved, rest, params[stage.name]) for clip in clips)
            before = current
            current = stage.apply(current, params[stage.name])
            if audit is not None:
                audit.record_transform(
                    stage=stage.name, direction="fit_rig", scope=stage.scope,
                    parameters=params[stage.name], before=before, after=current,
                )
        return params

    @staticmethod
    def _through(stage: PrepareStage, clip: Animation, resolved: Any, rest: Animation, rig_params) -> Animation:
        """``clip`` after ``stage``: with the rig's parameters, or its own if clip-scoped."""
        if stage.scope == CLIP:
            rig_params = stage.fit(clip, _rebind(resolved, rest, clip))
        return stage.apply(clip, rig_params)

    def apply(
        self, anim: Animation, resolved: Any, rig_params: dict[str, dict],
        audit: AuditSink | None = None,
    ) -> tuple[Animation, dict[str, dict]]:
        """Prepare one animation, returning it with the full parameter set."""
        params: dict[str, dict] = dict(rig_params)
        current = anim
        for stage in self.stages:
            if stage.scope == CLIP:
                params[stage.name] = stage.fit(current, _rebind(resolved, anim, current))
            elif stage.name not in params:
                raise KeyError(
                    f"no fitted parameters for rig-scoped stage `{stage.name}`; "
                    "call fit_rig against this rig's rest pose first"
                )
            before = current
            current = stage.apply(current, params[stage.name])
            if audit is not None:
                audit.record_transform(
                    stage=stage.name, direction="apply", scope=stage.scope,
                    parameters=params[stage.name], before=before, after=current,
                )
        return current, params

    def invert(
        self, anim: Animation, params: dict[str, dict],
        audit: AuditSink | None = None,
    ) -> Animation:
        """Undo the whole chain, stages in reverse order."""
        current = anim
        for stage in reversed(self.stages):
            before = current
            current = stage.invert(current, params[stage.name])
            if audit is not None:
                audit.record_transform(
                    stage=stage.name, direction="invert", scope=stage.scope,
                    parameters=params[stage.name], before=before, after=current,
                )
        return current


def _rebind(resolved: Any, source: Animation, current: Animation) -> Any:
    """``resolved``, re-bound to ``current``'s joints if a stage changed them.

    A manifest is resolved against the SOURCE joints, and `PromoteRoot` removes
    the locator chain, shifting every surviving index. Indices resolved before
    it then name whichever joints moved into their slots -- feet for thighs on
    Horse -- so a later stage fitting by index reads the wrong joints silently.
    """
    if resolved is None or tuple(current.names) == tuple(source.names):
        return resolved
    return resolve(resolved.manifest, current.names)


def _global_binds(local_bind: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Accumulate local rest rotations down the hierarchy. Parents precede children."""
    binds = np.empty_like(local_bind)
    binds[0] = local_bind[0]
    for joint in range(1, len(parents)):
        binds[joint] = quat_mul(binds[parents[joint]], local_bind[joint])
    return binds


class RestRelative(PrepareStage):
    """Re-express rotations so zero on every joint reproduces the rest pose.

    Raw Biped rigs bake a per-joint bind rotation into every clip: a
    bone-aligned frame (OFFSETs mostly ``(L, 0, 0)``) whose twist is the
    exporter's own, constant, not motion. Removing it is what makes the rest
    pose read as identity, which every stage before `BoneRelative` assumes.
    `BoneRelative` puts a bone-aligned frame back, with a twist chosen by one
    rule for every rig, when the corpus asks for it.

    Offsets become plain WORLD-SPACE differences of the rest pose's own global
    positions, not offsets rotated into a joint's local frame: identity rotation
    applied to a world-space offset reproduces that offset unchanged, which is
    exactly what makes forward kinematics self-consistent at the rest frame.
    Pairing identity rotations with parent-local offsets instead is a
    coordinate-frame mismatch invisible at the rest frame and wrong everywhere
    else.
    """

    name: ClassVar[str] = "rest_relative"
    #: Every stage after `up_axis` takes +Y as up.
    after: ClassVar[tuple[str, ...]] = ("up_axis",)
    scope: ClassVar[str] = RIG

    def fit(self, anim: Animation, resolved: Any) -> dict[str, np.ndarray]:
        rest = rest_geometry(anim)
        rest_global = rest.global_positions()
        parent_of = rest.parents[1:]

        rest_offsets = anim.offsets.copy()
        rest_offsets[1:] = rest_global[0, 1:] - rest_global[0, parent_of]
        return {
            "local_bind": rest.rotations[0].copy(),
            "rest_offsets": rest_offsets,
            "source_offsets": anim.offsets.copy(),
            "names": np.array(anim.names, dtype=object),
        }

    @staticmethod
    def _check(anim: Animation, params: dict[str, np.ndarray]) -> None:
        fitted = tuple(str(n) for n in params["names"])
        if tuple(anim.names) != fitted:
            raise ValueError(
                "this animation's joints differ from the rig `rest_relative` was "
                f"fitted to ({len(anim.names)} vs {len(fitted)} joints); a clip "
                "rigged differently from its own rest pose cannot be prepared "
                "against it"
            )

    def apply(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        self._check(anim, params)
        local_bind = params["local_bind"]
        binds = _global_binds(local_bind, anim.parents)

        rotations = anim.rotations.copy()
        rotations[:, 0] = quat_mul(anim.rotations[:, 0], quat_inverse(local_bind[0]))

        translations = anim.translations.copy()
        for joint in range(1, anim.n_joints):
            parent_bind = binds[anim.parents[joint]]
            # Sandwiched between its own local bind (undone) and its parent's
            # global bind (removed, then reapplied).
            rotations[:, joint] = quat_mul(
                quat_mul(
                    quat_mul(parent_bind, anim.rotations[:, joint]),
                    quat_inverse(local_bind[joint]),
                ),
                quat_inverse(parent_bind),
            )
            # A translation is expressed in the PARENT's frame, and that frame
            # has just turned, so it turns with it.
            translations[:, joint] = quat_apply(parent_bind, anim.translations[:, joint])

        return Animation(
            rotations=rotations,
            translations=translations,
            offsets=params["rest_offsets"].copy(),
            parents=anim.parents,
            names=anim.names,
            fps=anim.fps,
        )

    def invert(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        self._check(anim, params)
        local_bind = params["local_bind"]
        binds = _global_binds(local_bind, anim.parents)

        rotations = anim.rotations.copy()
        rotations[:, 0] = quat_mul(anim.rotations[:, 0], local_bind[0])

        translations = anim.translations.copy()
        for joint in range(1, anim.n_joints):
            parent_bind = binds[anim.parents[joint]]
            rotations[:, joint] = quat_mul(
                quat_mul(
                    quat_mul(quat_inverse(parent_bind), anim.rotations[:, joint]),
                    parent_bind,
                ),
                local_bind[joint],
            )
            translations[:, joint] = quat_apply(
                quat_inverse(parent_bind), anim.translations[:, joint]
            )

        return Animation(
            rotations=rotations,
            translations=translations,
            offsets=params["source_offsets"].copy(),
            parents=anim.parents,
            names=anim.names,
            fps=anim.fps,
        )


def _subtree(parents: np.ndarray, root: int) -> set[int]:
    """Every joint index descending from ``root``, ``root`` included."""
    indices = {root}
    frontier = [root]
    while frontier:
        parent = frontier.pop()
        children = [k for k, p in enumerate(parents) if p == parent]
        indices.update(children)
        frontier.extend(children)
    return indices


@dataclass(frozen=True)
class PromoteRoot(PrepareStage):
    """Move the root from a ground locator onto the first real body joint.

    14 of 73 Truebones rigs root the skeleton at a locator sitting at y=0 while
    the body hangs 0.7-6.5 bone lengths above it. `ric_pos` is root-relative, so
    those rigs hand the model every joint with a constant vertical bias, and
    their root-trajectory block is a ground projection where the other 59 give a
    body trajectory -- two conventions for one feature across 19% of the corpus.

    ONLY the root can absorb an offset, which is what makes this principled
    rather than an exception: `EnforceRigid` drops non-root translation, so no
    other joint has a channel to put one in. That is exactly why the reduction
    rule's collapse case requires a zero offset.

    The joints above `target` form a single-child chain and hierarchy order is
    depth-first, so they occupy indices 0..b-1 contiguously and every other
    joint descends from `target`. Promotion is a slice.

    `invert` re-inserts the chain with IDENTITY rotations and the composed
    product left on the promoted joint. Every joint returns with its name,
    parent and offset, and `target` and its descendants return to their exact
    world positions; the locator joints above it do not, because the chain's
    rotations all turn the same subtree and only their product was ever
    observable. Spec §9 states this as the contract: structure and reference
    frame, not motion values.

    One rig (Tukan) has a dead sibling subtree hanging off an ancestor --
    zero offset throughout, no motion, an FBX geometry-holder branch rather
    than skeleton. `fit` drops it from `apply`'s output (recorded as
    `dead_names`/`dead_offsets`/`dead_parents`/`dead_indices`) rather than
    raising the single-child-chain error, but `invert` splices it back at its
    exact original position with identity rotation and its own rest offset --
    the same treatment as the chain above the target -- so the structure
    guarantee still covers every joint, dead or not.
    """

    #: The joint to promote is the manifest's ``promote_root:``; a rig without
    #: one is untouched (every method is the identity), as 59 of 73 are.
    #: Measured on the rest pose as height fraction (rootY - minY) / (maxY - minY),
    #: the corpus splits absolutely: the 14 at f <= 0.005, the other 59 at
    #: f >= 0.189. Thirteen targets are the first branching joint along the
    #: root's single-child chain; Tukan's is authored, because its root branches
    #: straight into the skeleton (N_ALL -> locator) and a dead MESH subtree. The
    #: gate is the locator classification, never the presence of an offset:
    #: Lynx and BrownBear root correctly on the pelvis with a chain to
    #: Bip01_Spine 0.79 and 0.89 bone lengths long, and an offset-keyed rule
    #: would promote their root onto the spine.

    name: ClassVar[str] = "promote_root"
    #: After `rest_relative`, whose check needs the clip to carry exactly the rest pose's joints, which a promoted clip no longer does.
    after: ClassVar[tuple[str, ...]] = ("up_axis", "rest_relative")
    scope: ClassVar[str] = RIG

    def fit(self, anim: Animation, resolved: Any) -> dict[str, np.ndarray]:
        target = resolved.manifest.promote_root
        if target is None:
            return {
                "n_removed": np.int64(0),
                "names": np.empty(0, dtype="<U1"),
                "offsets": np.zeros((0, 3)),
                "dead_indices": np.empty(0, dtype=np.int64),
                "dead_names": np.empty(0, dtype="<U1"),
                "dead_offsets": np.zeros((0, 3)),
                "dead_parents": np.empty(0, dtype=np.int64),
            }

        names = list(anim.names)
        if target not in names:
            raise ValueError(
                f"cannot promote `{target}`: this rig has no such joint "
                f"(it has {len(names)}, starting {names[:3]})"
            )
        index = names.index(target)

        # Every joint above the target must have exactly one child continuing
        # the chain toward it. An ancestor with a second child is normally the
        # sibling-subtree hazard this check exists to catch -- but Tukan's
        # `Hips` carries a `MESH` branch of geometry-holder joints (zero offset
        # throughout, no motion, dead weight from the FBX export) alongside the
        # real skeleton. Such a branch occupies no space and nothing surviving
        # depends on it, so it is dropped outright rather than treated as an
        # ambiguous chain; anything else with more than one child still raises.
        dead: list[int] = []
        for joint in range(index):
            children = [k for k, p in enumerate(anim.parents) if p == joint]
            if len(children) == 1:
                continue
            live = [c for c in children if index in _subtree(anim.parents, c)]
            extra = [c for c in children if c not in live]
            if len(live) != 1 or any(
                not np.array_equal(anim.offsets[j], np.zeros(3))
                for c in extra
                for j in _subtree(anim.parents, c)
            ):
                raise ValueError(
                    f"cannot promote `{target}`: joint `{names[joint]}` above "
                    f"it has {len(children)} children, so the joints above the "
                    "target are not a single-child chain"
                )
            for c in extra:
                dead.extend(_subtree(anim.parents, c))

        # `invert` needs to splice the dead subtree back at its exact original
        # position (Spec §9's structure guarantee is non-negotiable), so its
        # names, offsets and -- critically -- its ORIGINAL parent indices are
        # recorded wholesale rather than just the bare indices `apply` needs.
        # Small in practice: 5 joints for Tukan, the only rig that has any.
        dead_sorted = sorted(dead)
        dead_names = np.array([names[i] for i in dead_sorted]) if dead_sorted else np.empty(
            0, dtype="<U1"
        )
        dead_offsets = anim.offsets[dead_sorted].copy() if dead_sorted else np.zeros((0, 3))
        dead_parents = (
            anim.parents[dead_sorted].astype(np.int64).copy()
            if dead_sorted
            else np.empty(0, dtype=np.int64)
        )

        chain = np.array(names[: index + 1])
        return {
            "n_removed": np.int64(index),
            "names": chain,
            "offsets": anim.offsets[: index + 1].copy(),
            "dead_indices": np.array(dead_sorted, dtype=np.int64),
            "dead_names": dead_names,
            "dead_offsets": dead_offsets,
            "dead_parents": dead_parents,
        }

    def apply(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        index = int(params["n_removed"])
        if index == 0:
            return anim

        dead = {int(i) for i in params.get("dead_indices", np.empty(0, dtype=np.int64))}
        keep = [j for j in range(anim.n_joints) if j >= index and j not in dead]
        old_to_new = {old: new for new, old in enumerate(keep)}

        positions, rotations_world = anim.global_transforms()

        translations = anim.translations[:, keep].copy()
        # The new root's translation slot holds a GLOBAL position (Animation's
        # contract for joint 0), which is exactly where the target already is.
        translations[:, 0] = positions[:, index]

        rotations = anim.rotations[:, keep].copy()
        # Its rotation absorbs the whole chain: with no ancestors left, its
        # local rotation must equal what its world rotation was.
        rotations[:, 0] = rotations_world[:, index]

        parents = np.array(
            [-1 if k == 0 else old_to_new[int(anim.parents[j])] for k, j in enumerate(keep)],
            dtype=np.int32,
        )

        return Animation(
            rotations=rotations,
            translations=translations,
            offsets=anim.offsets[keep].copy(),
            parents=parents,
            names=tuple(anim.names[j] for j in keep),
            fps=anim.fps,
        )

    def invert(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        index = int(params["n_removed"])
        if index == 0:
            return anim

        chain_names = [str(n) for n in params["names"]]
        chain_offsets = params["offsets"]
        dead_indices = [int(i) for i in params.get("dead_indices", np.empty(0, dtype=np.int64))]
        dead_names = [str(n) for n in params.get("dead_names", np.empty(0, dtype="<U1"))]
        dead_offsets = params.get("dead_offsets", np.zeros((0, 3)))
        dead_parents = [int(i) for i in params.get("dead_parents", np.empty(0, dtype=np.int64))]

        n_frames = anim.n_frames
        n_dead = len(dead_indices)
        n_joints = index + anim.n_joints + n_dead

        dead_set = set(dead_indices)
        # The target and its surviving descendants, in original absolute-index
        # order: everything from `index` upward that is not one of the dead
        # subtree's own joints. `apply` built `anim`'s rows in exactly this
        # order, so position `local` in `anim` is `kept_abs[local]` here.
        kept_abs = [p for p in range(index, n_joints) if p not in dead_set]

        rotations = np.empty((n_frames, n_joints, 4))
        translations = np.empty((n_frames, n_joints, 3))
        offsets = np.empty((n_joints, 3))
        parents = np.empty(n_joints, dtype=np.int32)
        names: list[str] = [""] * n_joints

        # The ancestor chain above the target: identity rotation throughout
        # (only their product was ever observable), its own rest offset for
        # translation, and a strict single-parent chain.
        rotations[:, :index] = QUAT_IDENTITY
        translations[:, 1:index] = chain_offsets[1:index]
        # The old root's global position is the target's position minus the
        # offsets accumulated down the (identity-rotation) chain.
        translations[:, 0] = anim.translations[:, 0] - chain_offsets[1 : index + 1].sum(axis=0)
        offsets[:index] = chain_offsets[:index]
        parents[0] = -1
        parents[1:index] = np.arange(index - 1, dtype=np.int32)
        for p in range(index):
            names[p] = chain_names[p]

        # The dead subtree: it never moved (zero offset, identity rotation in
        # every recorded clip), so reinsertion at its exact original position
        # and parent is exact, not approximate.
        for local, p in enumerate(dead_indices):
            rotations[:, p] = QUAT_IDENTITY
            translations[:, p] = dead_offsets[local]
            offsets[p] = dead_offsets[local]
            parents[p] = dead_parents[local]
            names[p] = dead_names[local]

        # The target and its surviving descendants: `anim`'s own rows,
        # remapped from post-`apply` indices back to their absolute original
        # position (and, for non-root rows, their absolute original parent).
        for local, p in enumerate(kept_abs):
            rotations[:, p] = anim.rotations[:, local]
            offsets[p] = anim.offsets[local]
            names[p] = anim.names[local]
            if local == 0:
                translations[:, p] = chain_offsets[index]
                parents[p] = index - 1
            else:
                translations[:, p] = anim.translations[:, local]
                parents[p] = kept_abs[int(anim.parents[local])]

        return Animation(
            rotations=rotations,
            translations=translations,
            offsets=offsets,
            parents=parents,
            names=tuple(names),
            fps=anim.fps,
        )


@dataclass(frozen=True)
class UpAxis(PrepareStage):
    """Turn the whole rig so the manifest's ``up_axis`` becomes +Y.

    A BVH does not record its up axis, so it is declared (`up_axis:` in the
    manifest, default ``+Y``, which makes this the identity). Forward needs no
    declaration: `FaceRestAxis` derives it from the facing joints. Inverting
    puts the rig back in its own convention.
    """

    name: ClassVar[str] = "up_axis"
    scope: ClassVar[str] = RIG

    def fit(self, anim: Animation, resolved: Any) -> dict[str, np.ndarray]:
        up = axis_vector(resolved.manifest.up_axis)
        return {"rotation": quat_between(up, np.array([0.0, 1.0, 0.0]))}

    def apply(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        return rotate_rig(anim, params["rotation"])

    def invert(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        return rotate_rig(anim, quat_inverse(params["rotation"]))


@dataclass(frozen=True)
class FaceRestAxis(PrepareStage):
    """Turn the whole rig so its REST skeleton faces ``axis``, +Y up.

    Rig-scoped, fitted on the rest geometry as a DCC tool draws it (offsets,
    no rotations), and applied with `rotate_rig`: offsets and every local
    rotation turn together, by the same amount for every clip of the rig. So
    every rig's rest pose faces +Z, its joints' local frames are the same in
    every clip, and they line up across rigs. `FaceAxis` then turns each clip
    the rest of the way, by its root alone.
    """

    axis: str = "+Z"

    name: ClassVar[str] = "face_rest_axis"
    #: After `enforce_rigid`: facing fitted on a Biped clip's per-joint translations came out up to ~90 deg off once they were dropped (Ant, Bat, Bird).
    after: ClassVar[tuple[str, ...]] = ("up_axis", "enforce_rigid")
    scope: ClassVar[str] = RIG

    def fit(self, anim: Animation, resolved: Any) -> dict[str, np.ndarray]:
        skeleton = RigidBodyAnimation.from_root_motion(
            rotations=np.tile(QUAT_IDENTITY, (1, anim.n_joints, 1)),
            root_pos=np.zeros((1, 3)), offsets=anim.offsets,
            parents=anim.parents, names=anim.names, fps=anim.fps,
        )
        rotation = facing_quats(
            skeleton.global_positions(),
            resolved.facing_indices,
            resolved.manifest.extra_yaw_deg,
            axis_vector(self.axis),
        )[0]
        return {"rotation": rotation}

    def apply(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        return rotate_rig(anim, params["rotation"])

    def invert(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        return rotate_rig(anim, quat_inverse(params["rotation"]))


@dataclass(frozen=True)
class FaceAxis(PrepareStage):
    """Turn each clip so its frame-0 facing direction points along ``axis``.

    Clip-scoped, and that is the whole reason this module exists: the rotation
    comes from THIS clip's frame 0, so once the prepared file faces +Z the
    original orientation cannot be recovered from it. It has to be recorded.

    It turns the ROOT only (`rotate_to_face_z`), after `FaceRestAxis` has
    turned the whole rig to face ``axis``: offsets and every other joint's
    local rotation stay shared by all clips of the rig, so the same pose has
    the same rotation features in every clip, and frame 0 faces the way the
    rest skeleton does. Turning the whole rig per clip instead gives the same
    world motion but conjugates every local rotation by that clip's own yaw,
    which put each clip in its own frame (up to 179 degrees apart on
    Truebones).
    """

    axis: str = "+Z"

    name: ClassVar[str] = "face_axis"
    #: After `enforce_rigid`, for the same reason as `face_rest_axis`.
    after: ClassVar[tuple[str, ...]] = ("up_axis", "enforce_rigid", "face_rest_axis")
    scope: ClassVar[str] = CLIP

    def fit(self, anim: Animation, resolved: Any) -> dict[str, np.ndarray]:
        rotation = facing_quats(
            anim.global_positions()[:1],
            resolved.facing_indices,
            resolved.manifest.extra_yaw_deg,
            axis_vector(self.axis),
        )[0]
        return {"rotation": rotation}

    def apply(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        return rotate_to_face_z(anim, params["rotation"])

    def invert(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        return rotate_to_face_z(anim, quat_inverse(params["rotation"]))


def _with_root(anim: RigidBodyAnimation, root_pos: np.ndarray) -> RigidBodyAnimation:
    return RigidBodyAnimation.from_root_motion(
        rotations=anim.rotations,
        root_pos=root_pos,
        offsets=anim.offsets,
        parents=anim.parents,
        names=anim.names,
        fps=anim.fps,
    )


@dataclass(frozen=True)
class EnforceRigid(PrepareStage):
    """Impose the rigid-bone assumption, recording the source channel layout.

    Features, the IK solver and the topology augmentations all assume constant
    bone lengths, so this is where a corpus that animates per-joint translation
    stops doing so. The motion discarded is real -- roughly 10% of skeleton size
    on raw Truebones -- which is why the stage is explicit rather than something
    the parser does silently.

    ``invert`` is the identity on the animation, because a rigid animation
    already carries its offsets in every translation slot. What has to be put
    back is the FILE's channel declaration, and that is the writer's job:
    ``BVH.from_animation(anim, channels=params["source_channels"])``.

    ``fit`` infers a channel layout from the observed motion, but that
    inference is fitted against the rest pose, which is often a single static
    frame -- so it under-declares (root only) for exactly the files where the
    real answer matters. A caller with the source file to hand should pass its
    real ``.channels`` through ``source_channels`` rather than rely on the
    inference; this is a second, independent copy of the layout rule ``BVH``
    itself applies on read, and letting a fitted guess silently replace a
    known-correct answer is the wrong default for a production pipeline.
    """

    joint_translation: str = "drop"
    #: The source file's real channel layout, when known. Recorded verbatim by
    #: `fit` in preference to inferring one from observed motion -- see the
    #: class docstring. `None` falls back to the inference, for callers with no
    #: file to hand (tests, synthetic rigs).
    source_channels: tuple[tuple[str, ...], ...] | None = None

    name: ClassVar[str] = "enforce_rigid"
    #: After `promote_root`, so the translation channel kept is the PROMOTED root's, not the locator's.
    after: ClassVar[tuple[str, ...]] = ("up_axis", "promote_root")
    scope: ClassVar[str] = RIG

    def fit(self, anim: Animation, resolved: Any) -> dict[str, np.ndarray]:
        if self.source_channels is not None:
            stored = np.empty(len(self.source_channels), dtype=object)
            stored[:] = list(self.source_channels)
            return {"source_channels": stored}

        moves = ~np.all(
            np.isclose(anim.translations, anim.offsets[np.newaxis], atol=1e-9), axis=(0, 2)
        )
        # A childless joint is written as an End Site, which declares an OFFSET
        # and no CHANNELS -- so it gets the empty tuple, matching what BVH.read
        # produces for one. Anything else would not be the source's own layout.
        has_child = np.zeros(anim.n_joints, dtype=bool)
        real = anim.parents >= 0
        has_child[anim.parents[real]] = True

        channels = tuple(
            ()
            if not has_child[joint]
            else (
                "Xposition", "Yposition", "Zposition", "Zrotation", "Xrotation", "Yrotation"
            )
            if joint == 0 or moves[joint]
            else ("Zrotation", "Xrotation", "Yrotation")
            for joint in range(anim.n_joints)
        )
        # Build the object array explicitly: np.array() on ragged tuples has
        # shape-inference heuristics that differ with the data.
        stored = np.empty(anim.n_joints, dtype=object)
        stored[:] = channels
        return {"source_channels": stored}

    def apply(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        return anim.as_rigid_body(joint_translation=self.joint_translation)

    def invert(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        return anim


@dataclass(frozen=True)
class CentreXZ(PrepareStage):
    """Translate so the root sits at the XZ origin on the rest pose's first frame."""

    name: ClassVar[str] = "centre_xz"
    #: The reference's order: rotate, centre XZ, scale, ground; reordering changes the result.
    after: ClassVar[tuple[str, ...]] = ("face_rest_axis", "face_axis")
    scope: ClassVar[str] = RIG

    def fit(self, anim: Animation, resolved: Any) -> dict[str, np.ndarray]:
        return {"root_xz": anim.translations[0, 0] * np.array([1.0, 0.0, 1.0])}

    def apply(self, anim, params):
        return _with_root(anim, anim.root_pos - params["root_xz"])

    def invert(self, anim, params):
        return _with_root(anim, anim.root_pos + params["root_xz"])


@dataclass(frozen=True)
class ScaleToMeanBoneLength(PrepareStage):
    """Uniformly scale so the rig's MEAN bone length equals ``target``.

    ``target`` defaults to the mean of the 21 SMPL bone lengths, which is what
    the reference scales every character to so a scorpion and an elephant occupy
    comparable numeric ranges.
    """

    target: float = HML_MEAN_BONE_LENGTH
    #: Bones at or below this length are zero-length End Sites, not bones. They
    #: must not enter the mean: they contribute nothing to the numerator and one
    #: to the denominator, so including them inflates the factor by
    #: (J_all - 1) / (J_real - 1) -- a rig-dependent number, which is exactly the
    #: cross-rig inconsistency this stage exists to remove. Matches the tolerance
    #: `features.reduce` uses to identify the same joints.
    tolerance: float = 1e-8

    name: ClassVar[str] = "scale"
    #: After `promote_root`: a locator-to-body connector is not an anatomical bone and must not enter the mean.
    after: ClassVar[tuple[str, ...]] = ("promote_root", "centre_xz")
    scope: ClassVar[str] = RIG

    def fit(self, anim: Animation, resolved: Any) -> dict[str, np.ndarray]:
        lengths = np.linalg.norm(anim.offsets[1:], axis=-1)
        real = lengths[lengths > self.tolerance]
        if real.size == 0:
            raise ValueError("skeleton has zero mean bone length; cannot scale")
        mean_length = float(real.mean())
        return {"factor": np.float64(self.target / mean_length)}

    def apply(self, anim, params):
        factor = float(params["factor"])
        return RigidBodyAnimation.from_root_motion(
            rotations=anim.rotations, root_pos=anim.root_pos * factor,
            offsets=anim.offsets * factor, parents=anim.parents,
            names=anim.names, fps=anim.fps,
        )

    def invert(self, anim, params):
        factor = float(params["factor"])
        return RigidBodyAnimation.from_root_motion(
            rotations=anim.rotations, root_pos=anim.root_pos / factor,
            offsets=anim.offsets / factor, parents=anim.parents,
            names=anim.names, fps=anim.fps,
        )


#: Each rig's rest-pose mesh, cached under `rig_cache()` by `tools/build_rest_mesh.py`
#: (the FBX SDK runs only in the `fbxsdk` container; the build never needs it).
REST_MESH = "rest_mesh.npz"


def similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """``s, R, t`` minimising ``|s R x + t - y|`` over paired ``(N, 3)`` points (Umeyama)."""
    mu_x, mu_y = source.mean(0), target.mean(0)
    x, y = source - mu_x, target - mu_y
    u, d, vt = np.linalg.svd(y.T @ x / len(x))
    fix = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ fix @ vt
    scale = float(np.trace(np.diag(d) @ fix) / (x**2).sum(1).mean())
    return scale, rotation, mu_y - scale * rotation @ mu_x


@dataclass(frozen=True)
class PutOnGround(PrepareStage):
    """Translate in Y so the rest pose sits on ``y = 0``, the way ``mode`` says.

    ``mode`` is the dataset's choice; a manifest's ``ground:`` overrides it for its rig.
    ``joints``: the rest clip's lowest joint over all its frames. ``mesh``: the
    lowest vertex of the vendor mesh skinned on the rest pose's first frame
    (`REST_MESH`), carried onto the joints as they stand here by the similarity
    that best maps the cached rest joints onto them. A rig set to ``mesh`` whose
    cache is missing, or was made for another rest pose, falls back to
    ``joints`` and says so with a `UserWarning`.

    Rig-scoped deliberately: fitting per clip would ground a flying creature and
    flatten the height difference between a crouch and a stand.
    """

    mode: str = "joints"

    name: ClassVar[str] = "ground"
    #: Grounds the scaled rig, +Y up.
    after: ClassVar[tuple[str, ...]] = ("up_axis", "scale")
    scope: ClassVar[str] = RIG

    def fit(self, anim: Animation, resolved: Any) -> dict[str, np.ndarray]:
        manifest = resolved.manifest
        mode = manifest.ground or self.mode
        if mode not in ("joints", "mesh"):
            raise ValueError(f"{manifest.name}: ground mode must be `joints` or `mesh`, got `{mode}`")
        if mode == "mesh":
            height, reason = _mesh_floor(anim, manifest)
            if height is not None:
                return {"height": np.float64(height)}
            warnings.warn(
                f"{manifest.name}: ground mode `mesh`, but {reason}; grounded on the lowest joint instead",
                stacklevel=2,
            )
        return {"height": np.float64(anim.global_positions()[..., 1].min())}

    def apply(self, anim, params):
        shift = np.array([0.0, float(params["height"]), 0.0])
        return _with_root(anim, anim.root_pos - shift)

    def invert(self, anim, params):
        shift = np.array([0.0, float(params["height"]), 0.0])
        return _with_root(anim, anim.root_pos + shift)


def _mesh_floor(anim: Animation, manifest: Any) -> tuple[float | None, str]:
    """The cached rest mesh's lowest height under ``anim``'s first frame, or why not."""
    path = rig_cache(corpus_root(manifest), Path(manifest.source).parent.name) / REST_MESH
    if not path.is_file():
        return None, f"{path} is missing (run tools/build_rest_mesh.py)"
    with np.load(path) as cache:
        if str(cache["rest_pose"]) != manifest.rest_pose:
            return None, f"{path} was made for {cache['rest_pose']}, not {manifest.rest_pose}"
        names, joints, vertices = [str(n) for n in cache["names"]], cache["joints"], cache["vertices"]
    common = [n for n in names if n in anim.names]
    if len(common) < 3:
        return None, f"{path} shares {len(common)} joints with the rest pose"
    here = anim.global_positions()[0, [anim.names.index(n) for n in common]]
    scale, rotation, translation = similarity(joints[[names.index(n) for n in common]], here)
    return float((scale * vertices @ rotation.T + translation)[:, 1].min()), ""


def bone_frames(offsets: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """``(J, 4)`` world-space rest orientation of a bone-aligned frame per joint.

    Local +X points along the joint's main child bone: the non-zero child
    offset with the most descendants, the longer one on a tie. Local +Y is
    the parent frame's +Y with its +X component removed (the parent's +Z when
    that is degenerate), so twist carries down a chain instead of being picked
    per joint. A joint with no non-zero child takes its parent's frame; the
    root's parent frame is the world. ``offsets`` are world-space rest bone
    vectors, as `RestRelative` leaves them.
    """
    n = len(parents)
    descendants = np.zeros(n, dtype=int)
    for joint in range(n - 1, 0, -1):
        descendants[parents[joint]] += descendants[joint] + 1
    frames = np.empty((n, 3, 3))
    for joint in range(n):
        parent = np.eye(3) if parents[joint] < 0 else frames[parents[joint]]
        children = [
            c for c in range(joint + 1, n)
            if parents[c] == joint and np.linalg.norm(offsets[c]) > 1e-8
        ]
        if not children:
            frames[joint] = parent
            continue
        main = max(children, key=lambda c: (descendants[c], np.linalg.norm(offsets[c])))
        x = offsets[main] / np.linalg.norm(offsets[main])
        y = parent[:, 1] - parent[:, 1].dot(x) * x
        if np.linalg.norm(y) < 1e-3:
            y = parent[:, 2] - parent[:, 2].dot(x) * x
        y /= np.linalg.norm(y)
        frames[joint] = np.stack([x, y, np.cross(x, y)], axis=-1)
    return matrix_to_quat(frames)


def _reframe(anim: Animation, frames: np.ndarray) -> Animation:
    """Express every joint in its own constant frame ``frames``: world motion unchanged.

    ``q'_j = A_p^-1 q_j A_j``, ``q'_0 = q_0 A_0``, ``o'_j = A_p^-1 o_j``; the
    root's world translation stays. Passing the inverse frames undoes it.
    """
    inverse = quat_inverse(frames)
    parents = anim.parents[1:]
    rotations = anim.rotations.copy()
    rotations[:, 0] = quat_mul(anim.rotations[:, 0], frames[0])
    rotations[:, 1:] = quat_mul(quat_mul(inverse[parents], anim.rotations[:, 1:]), frames[1:])
    translations = anim.translations.copy()
    translations[:, 1:] = quat_apply(inverse[parents], anim.translations[:, 1:])
    offsets = anim.offsets.copy()
    offsets[1:] = quat_apply(inverse[parents], anim.offsets[1:])
    return type(anim)(
        rotations=rotations, translations=translations, offsets=offsets,
        parents=anim.parents, names=anim.names, fps=anim.fps,
    )


@dataclass(frozen=True)
class BoneRelative(PrepareStage):
    """Re-express rotations so identity on every joint is a straight chain.

    The LAFAN1 / 3ds Max Biped convention, kept beside `RestRelative` for
    ablation: each joint's local +X runs along its bone (`bone_frames`), so a
    zero rotation means "continue the parent's bone" on every rig, instead of
    "the rig's own rest clip", and X is twist, Y/Z swing, on every joint.
    World motion is unchanged; only the local rotations and offsets change.

    Last in the chain because every position-fitting stage before it builds
    the rest skeleton from offsets under identity rotations. A dataset turns it
    on by listing it; ``enabled=False`` records identity frames, which makes
    every method the identity.
    """

    enabled: bool = True

    name: ClassVar[str] = "bone_relative"
    #: Last: every stage before it reads the rest skeleton as offsets under identity rotations.
    after: ClassVar[tuple[str, ...]] = ("*",)
    scope: ClassVar[str] = RIG

    def fit(self, anim: Animation, resolved: Any) -> dict[str, np.ndarray]:
        if not self.enabled:
            return {"frames": np.tile(QUAT_IDENTITY, (anim.n_joints, 1))}
        return {"frames": bone_frames(anim.offsets, anim.parents)}

    def apply(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        return _reframe(anim, params["frames"])

    def invert(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        return _reframe(anim, quat_inverse(params["frames"]))


@dataclass(frozen=True)
class RigTransform:
    """Everything the chain fitted for one rig, plus its per-clip parameters.

    Persisted because the transform is otherwise one-way. The facing rotation in
    particular is derived from a clip's own frame 0, so a prepared file already
    facing +Z no longer knows which way it started.

    Stored as a flat npz with ``/``-joined keys -- ``rig/scale/factor``,
    ``clip/walk/face_axis/rotation``. Most fields are plain numeric or
    fixed-width string arrays and read back with a bare ``np.load``, but a few
    (``enforce_rigid``'s per-joint channel tuples, which are ragged) are
    genuinely object arrays, so `load` opens the file with
    ``allow_pickle=True``. The file is still ``np.load``-inspectable; it is
    not pickle-free.
    """

    rig_params: dict[str, dict]
    clip_params: dict[str, dict[str, dict]]
    #: The ``prepare:`` entries the corpus was built with (`PrepareChain.entries`).
    chain: tuple[dict, ...] = ()

    def prepare_chain(self) -> PrepareChain:
        """The chain this corpus was prepared with, to invert it by. A stage of
        the user's own must be imported first (``custom_imports``)."""
        if not self.chain:
            raise ValueError("this preproc.npz records no chain (made before 2026-10-19): rebuild stage 1")
        return chain_from(self.chain)

    def params_for(self, clip: str) -> dict[str, dict]:
        """The full parameter set for one clip: rig-scoped plus its own."""
        if clip not in self.clip_params:
            known = ", ".join(sorted(self.clip_params)[:5]) or "(none)"
            raise KeyError(
                f"no recorded parameters for clip `{clip}`; known clips: {known}"
            )
        return {**self.rig_params, **self.clip_params[clip]}

    def save(self, path) -> None:
        arrays: dict[str, np.ndarray] = {}
        for stage, params in self.rig_params.items():
            for key, value in params.items():
                if key == "source_channels":
                    arrays[f"rig/{stage}/{key}"] = np.array(json.dumps([[str(channel) for channel in channels] for channels in value]))
                else:
                    arrays[f"rig/{stage}/{key}"] = np.asarray(value, dtype="<U" if key == "names" else None)
        for clip, stages in self.clip_params.items():
            # Clip names become path segments in the npz keys, and `load`
            # splits on "/" to recover the nesting. A name containing one
            # would misparse into the wrong shape -- silently, in the file
            # that is the only record of how to undo this corpus.
            if "/" in clip:
                raise ValueError(
                    f"clip name `{clip}` contains `/`, which is the separator "
                    "used inside preproc.npz keys; clip names must not contain it"
                )
            for stage, params in stages.items():
                for key, value in params.items():
                    arrays[f"clip/{clip}/{stage}/{key}"] = np.asarray(value, dtype="<U" if key == "names" else None)
        arrays["__clips__"] = np.array(json.dumps(sorted(self.clip_params)))
        arrays["__chain__"] = np.array(json.dumps(list(self.chain)))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(path), **arrays)

    @classmethod
    def load(cls, path) -> RigTransform:
        rig: dict[str, dict] = {}
        clips: dict[str, dict[str, dict]] = {}
        with np.load(Path(path), allow_pickle=True) as data:
            for name in json.loads(str(data["__clips__"])):
                clips[name] = {}
            chain = tuple(json.loads(str(data["__chain__"]))) if "__chain__" in data.files else ()
            for key in data.files:
                if key in ("__clips__", "__chain__"):
                    continue
                head, *rest = key.split("/")
                if head == "rig":
                    stage, field = rest
                    value = data[key]
                    if field == "source_channels":
                        if getattr(value, "dtype", None) is not None and value.dtype == object:
                            value = tuple(tuple(str(channel) for channel in channels) for channels in value.tolist())
                        else:
                            value = tuple(tuple(channel for channel in channels) for channels in json.loads(str(value)))
                    rig.setdefault(stage, {})[field] = value
                else:
                    clip, stage, field = rest
                    clips.setdefault(clip, {}).setdefault(stage, {})[field] = data[key]
        return cls(rig_params=rig, clip_params=clips, chain=chain)
