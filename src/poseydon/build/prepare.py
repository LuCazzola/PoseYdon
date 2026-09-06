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

from poseydon.core.animation import Animation

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
