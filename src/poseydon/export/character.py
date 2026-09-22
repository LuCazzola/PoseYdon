from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from poseydon.build.authority import ArtifactUnavailableError
from poseydon.core.animation import Animation, RigidBodyAnimation
from poseydon.core.rotations import QUAT_IDENTITY
from poseydon.features.reduce import apply_reduction, build_reduction, invert_reduction
from poseydon.io.bvh import BVH

from poseydon.build.prepare import PrepareChain, prepare_chain

@dataclass
class CharacterExporter:
    corpus: Path
    raw: Path
    rig: str
    chain: PrepareChain
    params: dict
    reduction: object
    clip_params: dict = field(default_factory=dict)
    source_fbx: Path | None = None
    model_fbx: Path | None = None
    presentation_asset: Path | None = None

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
        prep = corpus / "rigs" / rig / "preproc.npz"
        params = {}
        clip_params = {}
        if prep.is_file():
            from poseydon.build.prepare import RigTransform
            transform = RigTransform.load(prep)
            params, clip_params = transform.rig_params, transform.clip_params
        # Inversion reads the persisted parameters, so the chain needs neither
        # this rig's channel layout nor its promotion target -- only the ORDER,
        # which `prepare_chain` is the single definition of.
        chain = prepare_chain()
        raw_rig = raw / "source" / rig if (raw / "source" / rig).is_dir() else raw / rig
        source_fbx = None
        candidates = sorted(raw_rig.glob("*.fbx")) if raw_rig.is_dir() else []
        if candidates:
            # Pick the raw asset whose bind geometry corresponds to the source
            # skeleton, rather than assuming the first vendor action is the
            # character template (several Truebones files contain locator
            # roots or action-specific bind variants).
            try:
                from poseydon.io.fbx import FBX, _rest_positions, _similarity, _to_y_up
                model_rest = rest
                restore_params = {name: dict(value) for name, value in params.items()}
                restore_params.setdefault(
                    # Only reached for a rig prepared before the rig-level
                    # facing was recorded; `_similarity` solves for a rotation
                    # anyway, so the candidate ranking survives it.
                    "face_axis", {"rotation": np.array([1.0, 0.0, 0.0, 0.0])}
                )
                for stage in reversed(chain.stages):
                    model_rest = stage.invert(model_rest, restore_params.get(stage.name, {}))
                for candidate in candidates:
                    try:
                        scene = FBX.read(candidate)
                        if not all(name in scene.joint_names for name in model_rest.names):
                            continue
                        source_positions = np.stack([
                            _to_y_up(scene.joint_position(name, rest=True))
                            for name in model_rest.names
                        ])
                        target_positions = _rest_positions(model_rest)
                        scale, rotation, offset = _similarity(source_positions, target_positions)
                        fitted = scale * (source_positions @ rotation.T) + offset
                        residual = float(np.max(np.linalg.norm(fitted - target_positions, axis=1)))
                        if source_fbx is None or residual < source_fbx[0]:
                            source_fbx = (residual, candidate)
                    except Exception:
                        continue
                source_fbx = source_fbx[1] if source_fbx is not None else candidates[0]
            except ImportError:
                source_fbx = candidates[0]
        model = corpus / "rigs" / rig / "char-model.fbx"
        presentation = corpus / "rigs" / rig / "char-full.fbx"
        return cls(corpus, raw, rig, chain, params, reduction, clip_params,
                   source_fbx if source_fbx and source_fbx.is_file() else None,
                   model if model.is_file() else None,
                   presentation if presentation.is_file() else None)

    def _restore(self, anim: Animation, stop_before: str | None = None, facing_from: str | None = None):
        expected = tuple(self.reduction.source_names[i] for i in self.reduction.source_of)
        if tuple(anim.names) != expected:
            raise ValueError(f"{self.rig}: generated animation does not match reduced model topology")
        current = invert_reduction(anim, self.reduction)
        params = {name: dict(value) for name, value in self.params.items()}
        if facing_from is not None:
            if facing_from not in self.clip_params:
                raise KeyError(f"unknown facing clip `{facing_from}` for rig `{self.rig}`")
            params.update(self.clip_params[facing_from])
        elif "face_axis" not in params:
            raise KeyError(
                f"`{self.rig}`'s preproc.npz records no rig-level facing, so a "
                "generated clip has no frame to be restored into; pass "
                "`facing_from` to borrow a recorded clip's yaw, or rebuild the "
                "rig with scripts/process_dataset_truebones.py"
            )
        running = []
        for stage in reversed(self.chain.stages):
            if stage.name == stop_before:
                break
            running.append(stage)
        # An inverse is only as good as its recorded parameters, and a missing
        # one used to fall back to `{}` -- which for `enforce_rigid` is a silent
        # no-op and for the rest an opaque KeyError from inside the stage. Name
        # what is absent instead: the usual cause is a corpus prepared by an
        # older chain than the one being inverted with.
        missing = [stage.name for stage in running if stage.name not in params]
        if missing:
            raise KeyError(
                f"`{self.rig}` has no recorded parameters for preparation "
                f"stage(s) {', '.join(missing)}; this corpus was prepared by a "
                "different chain than the one being inverted with, so rebuild "
                "it with scripts/process_dataset_truebones.py"
            )
        for stage in running:
            current = stage.invert(current, params[stage.name])
        return current

    def prepared_bvh(self, anim): return BVH.from_animation(anim)

    def original_bvh(self, anim, facing_from: str | None = None):
        """The clip on the source rig, in the source file's own layout.

        The CHANNELS block comes back too. `EnforceRigid.invert` is the
        identity on the animation precisely because the declaration is the
        writer's job, and the source layout was recorded for this: a joint the
        source gave position channels gets them back, carrying its rest offset,
        so the file is structurally the rig the user supplied and the bones
        still do not stretch.
        """
        return BVH.from_animation(
            self._restore(anim, facing_from=facing_from),
            channels=self._source_channels(),
        )

    def _source_channels(self):
        recorded = self.params.get("enforce_rigid", {}).get("source_channels")
        return None if recorded is None else tuple(tuple(spec) for spec in recorded)

    def _fbx(self, anim, path, representation, facing_from: str | None = None):
        if path is None:
            raise ArtifactUnavailableError("corpus", self.rig, representation, "fbx", "source_surface_missing")
        from poseydon.io.fbx import FBX
        scene = FBX.read(path)
        if representation == "prepared":
            # The model FBX is already in canonical prepared space; applying
            # the reduced model animation directly is the point of this mode.
            restored = anim
        else:
            # Both `presentation` and `original` drive an existing armature, and
            # `apply_animation` requires a REST-RELATIVE animation: identity has
            # to mean the bind pose, which is the shared meaning of zero that
            # lets a BVH-derived clip and a vendor FBX be married at all. So the
            # inverse stops before `rest_relative` in each case, restoring the
            # source hierarchy and frame while leaving that meaning intact. The
            # two differ only in WHICH asset they drive.
            restored = self._restore(anim, "rest_relative", facing_from=facing_from)
        scene.apply_animation(restored, allow_undriven=representation in {"presentation", "original"})
        return scene

    def prepared_fbx(self, anim, facing_from: str | None = None): return self._fbx(anim, self.model_fbx, "prepared", facing_from)
    def presentation_fbx(self, anim, facing_from: str | None = None): return self._fbx(anim, self.presentation_asset, "presentation", facing_from)
    def original_fbx(self, anim, facing_from: str | None = None): return self._fbx(anim, self.source_fbx, "original", facing_from)
