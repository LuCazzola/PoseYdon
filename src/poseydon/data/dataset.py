"""Reading a corpus into feature windows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from poseydon.augment.base import Augmentation, AugmentPipeline
from poseydon.build.index import ClipRecord, CorpusIndex
from poseydon.conditioners.base import CONDITIONERS, Conditioner
from poseydon.core.animation import RigidBodyAnimation
from poseydon.core.skeleton import ResolvedSkeleton, SkeletonManifest, resolve
from poseydon.core.spec import FeatureSpec
from poseydon.data.normalize import Normalizer
from poseydon.data.window import RandomCrop, Window
from poseydon.features import DEFAULT_FEATURES, extract_features
from poseydon.features.reduce import JointReduction, apply_reduction, build_reduction


@dataclass(frozen=True)
class ClipView:
    """What a conditioner sees for one item."""

    record: ClipRecord
    anim: RigidBodyAnimation
    resolved: ResolvedSkeleton
    normalizer: Normalizer
    rest_frame: np.ndarray  # (J, D) normalized rest pose describing this rig
    clip: str  # record.clip_id, for a conditioner that needs to name the clip
    rig: str   # record.skeleton, likewise


@dataclass(frozen=True)
class Item:
    """One windowed sample, before collation."""

    features: np.ndarray  # (T, J, D) normalized
    spec: FeatureSpec
    start: int
    source_length: int
    cond: dict[str, Any]


class MotionDataset:
    """Corpus index plus manifests to feature windows.

    Features are extracted at read time from the canonical animation, so the
    representation is a config choice with no re-ingest. Normalization statistics
    are LOADED per skeleton from `rigs/<Rig>/stats.npz` -- the moments a
    checkpoint was trained against, not moments refitted at startup -- and the
    clip is put in the shape those moments were measured on: rigid body, then
    reduced.
    """

    def __init__(
        self,
        index: CorpusIndex,
        root: str | Path,
        manifest_dir: str | Path,
        features: Sequence[str] = DEFAULT_FEATURES,
        window: Window | None = None,
        conditioners: Sequence[str] = (),
        augmentations: Sequence[Augmentation] = (),
        split: str | None = None,
        seed: int = 0,
        reduce_tolerance: float = 1e-8,
    ) -> None:
        self.root = Path(root)
        self.manifest_dir = Path(manifest_dir)
        self.features = tuple(features)
        self.window = window or RandomCrop()
        self.conditioners: list[Conditioner] = [
            CONDITIONERS.get(name)() for name in conditioners
        ]
        self.augment_pipeline = AugmentPipeline(augmentations)
        # Matches `BuildConfig.reduce_tolerance`; taken as an argument so it
        # cannot drift from the value stage 2 reduced under.
        self.reduce_tolerance = reduce_tolerance
        self.records = index.query(split=split) if split else list(index.records)
        if not self.records:
            raise ValueError(f"no clips in the index for split={split!r}")

        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._manifests: dict[str, SkeletonManifest] = {}
        self._reductions: dict[str, JointReduction] = {}
        self._normalizers: dict[str, Normalizer] = {}
        self._anims: dict[str, RigidBodyAnimation] = {}
        self._rest_frames: dict[str, np.ndarray] = {}

        # `__init__` used to extract features for every clip just to count
        # windows -- 1145 full extractions before the first sample. The index
        # already knows each clip's frame count; the only unknown is how many
        # frames the SCHEMA costs (`local_vel` consumes one), and that is a
        # constant across clips. Measure it once, on the first record.
        #
        # That one probe leaves one clip in `self._anims`, and it stays there:
        # it is the clip `spec` reaches for next anyway.
        first = self.records[0]
        self._frame_cost = first.n_frames - self._extract(first)[0].shape[0]

        # A clip yields more than one window under a deterministic policy, so the
        # dataset is indexed by (clip, window) rather than by clip.
        self._plan: list[tuple[int, int]] = []
        for clip_index, record in enumerate(self.records):
            frames = record.n_frames - self._frame_cost
            for window_index in range(self.window.count(frames)):
                self._plan.append((clip_index, window_index))

    def __len__(self) -> int:
        return len(self._plan)

    def set_worker_seed(self, worker_id: int) -> None:
        """Give this worker its own stream.

        `__init__` builds one Generator, and `fork` copies it into every
        dataloader worker -- so with `num_workers > 0` all workers drew the
        SAME crops and the same augmentations, silently reducing the effective
        variety of a batch by a factor of `num_workers`. Called from
        `worker_init_fn`.
        """
        self._rng = np.random.default_rng([self.seed, worker_id])

    @property
    def spec(self) -> FeatureSpec:
        _, spec = self._extract(self.records[0])
        return spec

    def _manifest(self, skeleton: str) -> SkeletonManifest:
        if skeleton not in self._manifests:
            # Entity-first layout: `rigs/<Rig>/manifest.yaml`, not the flat
            # `rigs/<Rig>.yaml` the migration deleted.
            self._manifests[skeleton] = SkeletonManifest.load(
                self.manifest_dir / skeleton / "manifest.yaml"
            )
        return self._manifests[skeleton]

    def _rest_clip(self, skeleton: str) -> RigidBodyAnimation:
        """The rig's DECLARED rest clip, as a rigid body.

        The action is read from `skeleton.npz`, not re-derived from the
        manifest: `build_skeleton` recorded the very action it reduced under,
        so the two cannot drift, and the read path stays clear of
        `build.corpus` (a build-time internal -- tests/build/test_layering.py).
        """
        with np.load(
            self.manifest_dir / skeleton / "skeleton.npz", allow_pickle=True
        ) as data:
            action = str(data["rest_action"])
        return RigidBodyAnimation.load(
            self.root / "clips" / skeleton / f"{action}.npz"
        ).as_rigid_body(joint_translation="drop")

    def _reduction(self, skeleton: str) -> JointReduction:
        """The rig's reduction, built once and checked against `skeleton.npz`.

        `skeleton.npz` stores `reduction_source_of`, which is a `JointEdit`: it
        can transport statistics but cannot APPLY a reduction, because a
        COLLAPSE removal composes the victim's rotation into its parent and a
        reindex map carries no rotations. So the ops are rebuilt here and the
        stored map is used as a consistency check on them.

        Per RIG, not per clip. Removal is selected on `norm(offsets)`,
        `parents` and `names`; clips of one rig differ only by a rotation of
        the offsets, which leaves a norm unchanged. Same invariant
        `build_skeleton` and `build_stats` rely on.
        """
        if skeleton not in self._reductions:
            with np.load(
                self.manifest_dir / skeleton / "skeleton.npz", allow_pickle=True
            ) as data:
                stored = tuple(int(i) for i in data["reduction_source_of"])
            rest = self._rest_clip(skeleton)
            reduction = build_reduction(rest, tolerance=self.reduce_tolerance)
            if reduction.source_of != stored:
                raise ValueError(
                    f"{skeleton}: the reduction built here does not match the one "
                    f"`skeleton.npz` records, so `stats.npz` would pair each joint's "
                    f"values with another joint's statistics. Rebuild stage 2."
                )
            self._reductions[skeleton] = reduction
        return self._reductions[skeleton]

    def _anim(self, record: ClipRecord) -> RigidBodyAnimation:
        """The clip as the statistics saw it: rigid body, then reduced.

        Both steps are load-bearing and ordered. `build_stats` fits on
        `as_rigid_body(joint_translation="drop")` and then reduces
        (`build/pipeline.py`); reading in any other shape normalizes against
        moments that were never measured on these values.

        `as_rigid_body` is a no-op on today's corpus -- stage 1's `EnforceRigid`
        already made every prepared clip rigid -- and stays here anyway: it is
        the form `build_stats` fits under, so a non-rigid clip appearing later
        must not silently diverge from it.
        """
        if record.clip_id not in self._anims:
            anim = RigidBodyAnimation.load(self.root / record.path)
            anim = anim.as_rigid_body(joint_translation="drop")
            self._anims[record.clip_id] = apply_reduction(
                anim, self._reduction(record.skeleton)
            )
        return self._anims[record.clip_id]

    def _resolved(self, record: ClipRecord) -> ResolvedSkeleton:
        return resolve(self._manifest(record.skeleton), self._anim(record).names)

    def _extract(self, record: ClipRecord) -> tuple[np.ndarray, FeatureSpec]:
        return extract_features(
            self._anim(record), self._resolved(record), self.features
        )

    def _normalizer(self, skeleton: str, spec: FeatureSpec) -> Normalizer:
        """Loaded, never fitted.

        Fitting walked every clip of the rig at startup. Worse than slow: a
        refit under a different `features:` than the checkpoint was trained
        with produces different moments, silently.
        """
        if skeleton not in self._normalizers:
            normalizer = Normalizer.load(self.manifest_dir / skeleton / "stats.npz")
            if normalizer.spec != spec:
                raise ValueError(
                    f"{skeleton}: stats.npz was fitted for {normalizer.spec} but this "
                    f"run declares {spec}. Re-run `scripts/build_features.py --stats-only`."
                )
            self._normalizers[skeleton] = normalizer
        return self._normalizers[skeleton]

    def _extract_rest(self, skeleton: str) -> np.ndarray:
        """Frame 0 of the rig's DECLARED rest clip, normalized.

        `manifest.rest_pose` is authored for all 73 rigs (plan A1) exactly so
        this is never guessed, and `build_skeleton` copied that choice into
        `skeleton.npz`. The old fallback took the first clip the index happened
        to list, which for Crab -- whose rest pose is `__Walk.bvh` and whose
        first indexed clip is `attack1` -- described the rig with an attack
        pose. `stats.npz` is NOT the source: it records no rest frame, whatever
        an earlier draft of the spec promised.
        """
        manifest = self._manifest(skeleton)
        reduced = apply_reduction(self._rest_clip(skeleton), self._reduction(skeleton))
        raw, spec = extract_features(
            reduced, resolve(manifest, reduced.names), self.features
        )
        return self._normalizer(skeleton, spec).normalize(raw[:1])[0]

    def _rest_frame_of(self, skeleton: str) -> np.ndarray:
        if skeleton not in self._rest_frames:
            self._rest_frames[skeleton] = self._extract_rest(skeleton)
        return self._rest_frames[skeleton]

    def __getitem__(self, index: int) -> Item:
        clip_index, window_index = self._plan[index]
        record = self.records[clip_index]

        anim = self._anim(record)
        resolved = self._resolved(record)
        anim, resolved, edit = self.augment_pipeline.apply_structural(anim, resolved, self._rng)

        raw, spec = extract_features(anim, resolved, self.features)
        base_normalizer = self._normalizer(record.skeleton, spec)
        normalizer = edit.transport(base_normalizer)
        features = normalizer.normalize(raw)
        features = self.augment_pipeline.apply_features(features, spec, resolved, self._rng)

        start, length = self.window.bounds(features.shape[0], window_index, self._rng)
        window = features[start : start + length]

        rest_frame = edit.transport_row(self._rest_frame_of(record.skeleton))

        view = ClipView(
            record=record,
            anim=anim,
            resolved=resolved,
            normalizer=normalizer,
            rest_frame=rest_frame,
            clip=record.clip_id,
            rig=record.skeleton,
        )
        return Item(
            features=window,
            spec=spec,
            start=start,
            source_length=features.shape[0],
            cond={c.name: c.extract(view) for c in self.conditioners},
        )
