"""Everything recorded about one rig, and moving its clips between spaces.

A clip lives in one of three spaces: the model view (reduced, canonical),
prepared (every joint, canonical) and source (the user's rig, as supplied).
`RigRecord.expand` and `RigRecord.restore` move a clip between them; writing
it is then the file format's job, `BVH.from_animation` or
`FBX.from_animation`. The two things a file needs that a clip does not carry
come from the record too: a source BVH's CHANNELS layout
(`RigRecord.source_channels`) and the FBX asset a clip is written onto, the
rig's rest asset (`RigRecord.asset`). A restored clip is in that asset's own
space, so it drives the asset's nodes by name; a prepared one is placed in
canonical space by `RigRecord.placement`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from poseydon.build.authority import ArtifactUnavailableError
from poseydon.build.corpus import rig_cache
from poseydon.build.prepare import PrepareChain, RigTransform, similarity
from poseydon.build.sources import REST_FBX
from poseydon.core.animation import Animation, RigidBodyAnimation
from poseydon.core.rotations import QUAT_IDENTITY
from poseydon.features.reduce import build_reduction, invert_reduction, load_rules, load_still

@dataclass
class RigRecord:
    """One rig as the corpus recorded it: its reduction, its preparation
    parameters and, for FBX, its rest asset.

    It holds no file: ``expand`` and ``restore`` return plain animations, and
    `BVH.from_animation` / `FBX.from_animation` write them.
    """

    corpus: Path
    rig: str
    chain: PrepareChain
    params: dict
    reduction: object
    clip_params: dict = field(default_factory=dict)
    #: The reduce rules `skeleton.npz` records, the dataset's when stage 2 ran.
    reduction_rules: tuple = ()

    @classmethod
    def load(cls, corpus: str | Path, rig: str, layout: str = "rot6d_joint") -> RigRecord:
        """``layout`` is the rotation block the model's features use: it
        decides which joints the model view drops (`build_reduction`)."""
        corpus = Path(corpus)
        skeleton = corpus / "rigs" / rig / "skeleton.npz"
        if not skeleton.is_file():
            raise FileNotFoundError(skeleton)
        with np.load(skeleton, allow_pickle=True) as data:
            names = tuple(str(x) for x in data["names"])
            rest = RigidBodyAnimation.from_root_motion(
                rotations=np.tile(QUAT_IDENTITY, (1, len(names), 1)),
                root_pos=np.zeros((1, 3)), offsets=data["offsets"],
                parents=data["parents"].astype(np.int32), names=names, fps=1.0)
            rules = load_rules(data)
            reduction = build_reduction(rest, rules, layout=layout, still=load_still(data))
        prep = corpus / "rigs" / rig / "preproc.npz"
        params, clip_params, chain = {}, {}, PrepareChain(())
        if prep.is_file():
            transform = RigTransform.load(prep)
            params, clip_params, chain = transform.rig_params, transform.clip_params, transform.prepare_chain()
        # Inversion reads the persisted parameters, so the chain needs neither
        # this rig's channel layout nor its promotion target -- only the stages
        # and their order, which preproc.npz records as built.
        return cls(corpus, rig, chain, params, reduction, clip_params, reduction_rules=rules)

    @property
    def _model_names(self) -> tuple[str, ...]:
        return tuple(self.reduction.source_names[i] for i in self.reduction.source_of)

    def expand(self, anim: Animation) -> Animation:
        """The model view -> the prepared clip, every joint, still canonical."""
        if tuple(anim.names) != self._model_names:
            raise ValueError(f"{self.rig}: animation does not match the reduced model topology")
        return invert_reduction(anim, self.reduction)

    def restore(self, anim: Animation, facing_from: str | None = None) -> Animation:
        """The model view or a prepared clip -> the source rig, in source space.

        ``facing_from=<action>`` restores that clip's recorded yaw; without it,
        the rig-level facing (the rest pose's).
        """
        if tuple(anim.names) == self._model_names:
            current = self.expand(anim)
        elif tuple(anim.names) == tuple(self.reduction.source_names):
            current = anim
        else:
            raise ValueError(f"{self.rig}: animation is neither the model view nor the prepared rig")
        params = {name: dict(value) for name, value in self.params.items()}
        if facing_from is not None:
            if facing_from not in self.clip_params:
                raise KeyError(f"unknown facing clip `{facing_from}` for rig `{self.rig}`")
            params.update(self.clip_params[facing_from])
        running = list(reversed(self.chain.stages))
        if facing_from is None and "face_axis" not in params:
            raise KeyError(
                f"`{self.rig}`'s preproc.npz records no rig-level facing, so a "
                "generated clip has no frame to be restored into; pass "
                "`facing_from` to borrow a recorded clip's yaw, or rebuild the "
                "rig with scripts/process_dataset_truebones.py"
            )
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

    @property
    def source_channels(self) -> tuple[tuple[str, ...], ...] | None:
        """The source BVH's CHANNELS layout, for writing a restored clip.

        `EnforceRigid.invert` is the identity on the animation precisely
        because the declaration is the writer's job: a joint the source gave
        position channels gets them back, carrying its rest offset, so the file
        is structurally the rig the user supplied and the bones still do not
        stretch.

        ``None`` (the writers' default layout) where the source has no layout
        (`fbx_cache`): `EnforceRigid` then infers one, flagged `inferred`, on the
        pruned, BFS-ordered joints, which is not the order of a restored clip
        (every joint `prune_unskinned` removed back, source order). A recorded
        layout for another skeleton (joint count differs) is dropped too.
        """
        enforce = self.params.get("enforce_rigid", {})
        recorded = enforce.get("source_channels")
        restored = self.params.get("prune_unskinned", {}).get("source_names")
        if (recorded is None or "inferred" in enforce
                or (restored is not None and len(recorded) != len(restored))):
            return None
        return tuple(tuple(spec) for spec in recorded)

    @property
    def fixed(self) -> tuple[str, ...]:
        """Joints a restored clip holds by structure, not by value: the chain
        above a promoted root and its dead branches (`PromoteRoot`), and the
        roots rig cleanup removed above the real one (`prune_unskinned`). A BVH
        has to give them the motion `restore` puts on them, which follows the
        body; an FBX leaves them at the asset's own rest transform
        (`FBX.from_animation`'s ``fixed``), where a static locator really is,
        and drives the body's root absolutely."""
        promote = self.params.get("promote_root", {})
        prune = self.params.get("prune_unskinned", {})
        roots = [n for n, kind in zip(prune.get("removed", ()), prune.get("kind", ())) if kind == "root"]
        names = list(promote.get("names", ())[:-1]) + list(promote.get("dead_names", ())) + roots
        return tuple(str(n) for n in names)

    @property
    def asset(self) -> Path:
        """The FBX a clip is written onto: the rig's rest asset, its vendor FBX
        re-bound to the rest pose (`tools/build_rest_fbx.py`), mesh and skin
        included."""
        path = rig_cache(self.corpus, self.rig) / REST_FBX
        if not path.is_file():
            raise ArtifactUnavailableError(
                "corpus", self.rig, "source", "fbx", f"{path} is missing (run tools/build_rest_fbx.py)"
            )
        return path

    def placement(self, anim: Animation, facing_from: str | None = None) -> tuple[Animation, np.ndarray]:
        """The model view or a prepared clip as the asset takes it in canonical
        space: the clip restored to the source rig, and the 4x4 similarity that
        puts the source back where the clip stands (`FBX.from_animation`'s
        ``space``), mesh included.

        Every stage moves, turns or scales the whole clip or re-frames joints
        without moving them, so the restored clip and the prepared one are one
        similarity apart; it is solved on every joint and frame, not composed
        from the stages.
        """
        prepared = self.expand(anim) if tuple(anim.names) == self._model_names else anim
        restored = self.restore(prepared, facing_from=facing_from)
        names = [n for n in prepared.names if n in restored.names]
        source = restored.global_positions()[:, [restored.names.index(n) for n in names]]
        target = prepared.global_positions()[:, [prepared.names.index(n) for n in names]]
        scale, rotation, translation = similarity(source.reshape(-1, 3), target.reshape(-1, 3))
        space = np.eye(4)
        space[:3, :3], space[:3, 3] = scale * rotation, translation
        return restored, space
