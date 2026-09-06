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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from poseydon.core.animation import Animation, RigidBodyAnimation, rest_geometry
from poseydon.core.rotations import quat_apply, quat_inverse, quat_mul
from poseydon.core.skeleton import HML_MEAN_BONE_LENGTH
from poseydon.ingest.align import axis_vector, facing_quats, rotate_rig

RIG = "rig"
CLIP = "clip"


class PrepareStage(ABC):
    """One invertible step of the source-to-prepared transform."""

    #: Key under which this stage's fitted parameters are stored.
    name: ClassVar[str]
    #: ``RIG`` to fit once against the rest pose, ``CLIP`` to fit per animation.
    scope: ClassVar[str] = RIG

    def fit(self, anim: Animation, resolved: Any) -> dict[str, np.ndarray]:
        """Parameters derived from ``anim`` as it stands at this point in the chain."""
        return {}

    @abstractmethod
    def apply(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation: ...

    @abstractmethod
    def invert(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation: ...


@dataclass(frozen=True)
class PrepareChain:
    """An ordered composition of stages, invertible as a whole."""

    stages: tuple[PrepareStage, ...]

    def fit_rig(self, rest: Animation, resolved: Any) -> dict[str, dict]:
        """Fit every stage against the rest pose; keep the rig-scoped results.

        Each stage fits against the animation as the PRECEDING stages left it,
        which is why this applies as it goes: the scale factor is measured on an
        already-rest-relative skeleton, and the ground height on an already
        scaled one.
        """
        params: dict[str, dict] = {}
        current = rest
        for stage in self.stages:
            params[stage.name] = stage.fit(current, resolved)
            current = stage.apply(current, params[stage.name])
        return {stage.name: params[stage.name] for stage in self.stages if stage.scope == RIG}

    def apply(
        self, anim: Animation, resolved: Any, rig_params: dict[str, dict]
    ) -> tuple[Animation, dict[str, dict]]:
        """Prepare one animation, returning it with the full parameter set."""
        params: dict[str, dict] = dict(rig_params)
        current = anim
        for stage in self.stages:
            if stage.scope == CLIP:
                params[stage.name] = stage.fit(current, resolved)
            elif stage.name not in params:
                raise KeyError(
                    f"no fitted parameters for rig-scoped stage `{stage.name}`; "
                    "call fit_rig against this rig's rest pose first"
                )
            current = stage.apply(current, params[stage.name])
        return current, params

    def invert(self, anim: Animation, params: dict[str, dict]) -> Animation:
        """Undo the whole chain, stages in reverse order."""
        current = anim
        for stage in reversed(self.stages):
            current = stage.invert(current, params[stage.name])
        return current


def _global_binds(local_bind: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Accumulate local rest rotations down the hierarchy. Parents precede children."""
    binds = np.empty_like(local_bind)
    binds[0] = local_bind[0]
    for joint in range(1, len(parents)):
        binds[joint] = quat_mul(binds[parents[joint]], local_bind[joint])
    return binds


class RestRelative(PrepareStage):
    """Re-express rotations so zero on every joint reproduces the rest pose.

    Raw Biped rigs bake an arbitrary per-joint bind rotation into every clip --
    an exporter axis-convention artefact, not motion. Removing it is what makes a
    T-pose read as identity, which every downstream consumer assumes.

    Offsets become plain WORLD-SPACE differences of the rest pose's own global
    positions, not offsets rotated into a joint's local frame: identity rotation
    applied to a world-space offset reproduces that offset unchanged, which is
    exactly what makes forward kinematics self-consistent at the rest frame.
    Pairing identity rotations with parent-local offsets instead is a
    coordinate-frame mismatch invisible at the rest frame and wrong everywhere
    else.
    """

    name = "rest_relative"
    scope = RIG

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


@dataclass(frozen=True)
class FaceAxis(PrepareStage):
    """Turn each clip so its frame-0 facing direction points along ``axis``.

    Clip-scoped, and that is the whole reason this module exists: the rotation
    comes from THIS clip's frame 0, so once the prepared file faces +Z the
    original orientation cannot be recovered from it. It has to be recorded.
    """

    axis: str = "+Z"

    name: ClassVar[str] = "face_axis"
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
        return rotate_rig(anim, params["rotation"])

    def invert(self, anim: Animation, params: dict[str, np.ndarray]) -> Animation:
        return rotate_rig(anim, quat_inverse(params["rotation"]))


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
    """

    joint_translation: str = "drop"

    name: ClassVar[str] = "enforce_rigid"
    scope: ClassVar[str] = RIG

    def fit(self, anim: Animation, resolved: Any) -> dict[str, np.ndarray]:
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

    name: ClassVar[str] = "scale"
    scope: ClassVar[str] = RIG

    def fit(self, anim: Animation, resolved: Any) -> dict[str, np.ndarray]:
        mean_length = float(np.linalg.norm(anim.offsets[1:], axis=-1).mean())
        if mean_length < 1e-12:
            raise ValueError("skeleton has zero mean bone length; cannot scale")
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


@dataclass(frozen=True)
class PutOnGround(PrepareStage):
    """Translate in Y so the rest pose's lowest joint sits at ``y = 0``.

    Rig-scoped deliberately: fitting per clip would ground a flying creature and
    flatten the height difference between a crouch and a stand.
    """

    name: ClassVar[str] = "ground"
    scope: ClassVar[str] = RIG

    def fit(self, anim: Animation, resolved: Any) -> dict[str, np.ndarray]:
        return {"height": np.float64(anim.global_positions()[..., 1].min())}

    def apply(self, anim, params):
        shift = np.array([0.0, float(params["height"]), 0.0])
        return _with_root(anim, anim.root_pos - shift)

    def invert(self, anim, params):
        shift = np.array([0.0, float(params["height"]), 0.0])
        return _with_root(anim, anim.root_pos + shift)


@dataclass(frozen=True)
class RigTransform:
    """Everything the chain fitted for one rig, plus its per-clip parameters.

    Persisted because the transform is otherwise one-way. The facing rotation in
    particular is derived from a clip's own frame 0, so a prepared file already
    facing +Z no longer knows which way it started.

    Stored as a flat npz with ``/``-joined keys -- ``rig/scale/factor``,
    ``clip/walk/face_axis/rotation`` -- so the file stays inspectable with
    ``np.load`` and needs no pickle.
    """

    rig_params: dict[str, dict]
    clip_params: dict[str, dict[str, dict]]

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
                arrays[f"rig/{stage}/{key}"] = np.asarray(value)
        for clip, stages in self.clip_params.items():
            for stage, params in stages.items():
                for key, value in params.items():
                    arrays[f"clip/{clip}/{stage}/{key}"] = np.asarray(value)
        arrays["__clips__"] = np.array(json.dumps(sorted(self.clip_params)))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(path), **arrays)

    @classmethod
    def load(cls, path) -> RigTransform:
        rig: dict[str, dict] = {}
        clips: dict[str, dict[str, dict]] = {}
        with np.load(Path(path), allow_pickle=True) as data:
            for name in json.loads(str(data["__clips__"])):
                clips[name] = {}
            for key in data.files:
                if key == "__clips__":
                    continue
                head, *rest = key.split("/")
                if head == "rig":
                    stage, field = rest
                    rig.setdefault(stage, {})[field] = data[key]
                else:
                    clip, stage, field = rest
                    clips.setdefault(clip, {}).setdefault(stage, {})[field] = data[key]
        return cls(rig_params=rig, clip_params=clips)
