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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from poseydon.core.animation import Animation, rest_geometry
from poseydon.core.rotations import quat_apply, quat_inverse, quat_mul

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
