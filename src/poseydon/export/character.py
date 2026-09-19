from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from poseydon.build.authority import ArtifactUnavailableError
from poseydon.build.prepare import PrepareChain
from poseydon.core.animation import Animation, RigidBodyAnimation
from poseydon.core.rotations import QUAT_IDENTITY
from poseydon.features.reduce import build_reduction, invert_reduction
from poseydon.io.bvh import BVH
from scripts.anim_2_fbx import _chain_for

@dataclass
class CharacterExporter:
    corpus: Path
    raw: Path
    rig: str
    chain: PrepareChain
    params: dict
    reduction: object
    source_fbx: Path | None = None
    model_fbx: Path | None = None
    presentation_fbx: Path | None = None

    @classmethod
    def load(cls, corpus: str | Path, raw: str | Path, rig: str):
        corpus, raw = Path(corpus), Path(raw)
        skeleton = corpus / "rigs" / rig / "skeleton.npz"
        if not skeleton.is_file():
            raise FileNotFoundError(skeleton)
        with np.load(skeleton, allow_pickle=True) as data:
            names = tuple(str(x) for x in data["names"])
            rest = RigidBodyAnimation.from_root_motion(
                rotations=np.tile(QUAT_IDENTITY, (1, len(names), 1)),
                root_pos=np.zeros((1, 3)), offsets=data["offsets"],
                parents=data["parents"].astype(np.int32), names=names, fps=1.0)
            reduction = build_reduction(rest)
        prep = corpus / "rigs" / rig / "prepare.npz"
        params = {}
        if prep.is_file():
            with np.load(prep, allow_pickle=True) as data:
                for key in data.files:
                    if key.startswith("rig/"):
                        _, stage, field = key.split("/", 2)
                        params.setdefault(stage, {})[field] = data[key]
        source_fbx = next(iter(sorted((raw / rig).glob("*.fbx"))), None) if (raw / rig).is_dir() else None
        model = corpus / "rigs" / rig / "model_character.fbx"
        presentation = corpus / "rigs" / rig / "presentation.fbx"
        return cls(corpus, raw, rig, _chain_for(None, rig), params, reduction,
                   source_fbx if source_fbx and source_fbx.is_file() else None,
                   model if model.is_file() else None,
                   presentation if presentation.is_file() else None)

    def _restore(self, anim: Animation, stop_before: str | None = None):
        expected = tuple(self.reduction.source_names[i] for i in self.reduction.source_of)
        if tuple(anim.names) != expected:
            raise ValueError(f"{self.rig}: generated animation does not match reduced model topology")
        current = invert_reduction(anim, self.reduction)
        for stage in reversed(self.chain.stages):
            if stage.name == stop_before:
                break
            current = stage.invert(current, self.params.get(stage.name, {}))
        return current

    def prepared_bvh(self, anim): return BVH.from_animation(anim)
    def original_bvh(self, anim): return BVH.from_animation(self._restore(anim))

    def _fbx(self, anim, path, representation):
        if path is None:
            raise ArtifactUnavailableError("corpus", self.rig, representation, "fbx", "source_surface_missing")
        from poseydon.io.fbx import FBX
        scene = FBX.read(path)
        scene.apply_animation(self._restore(anim, "rest_relative") if representation == "presentation" else self._restore(anim))
        return scene

    def prepared_fbx(self, anim): return self._fbx(anim, self.model_fbx, "prepared")
    def presentation_fbx(self, anim): return self._fbx(anim, self.presentation_fbx, "presentation")
    def original_fbx(self, anim): return self._fbx(anim, self.source_fbx, "original")
