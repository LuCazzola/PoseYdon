"""Reading a corpus into feature windows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from poseydon.augment.base import Augmentation, AugmentPipeline
from poseydon.conditioners.base import CONDITIONERS, Conditioner
from poseydon.core.animation import RigidBodyAnimation
from poseydon.core.skeleton import ResolvedSkeleton, SkeletonManifest, resolve
from poseydon.core.spec import FeatureSpec
from poseydon.data.normalize import Normalizer
from poseydon.data.window import RandomCrop, Window
from poseydon.features import DEFAULT_FEATURES, extract_features
from poseydon.ingest.index import ClipRecord, CorpusIndex


@dataclass(frozen=True)
class ClipView:
    """What a conditioner sees for one item."""

    record: ClipRecord
    anim: RigidBodyAnimation
    resolved: ResolvedSkeleton
    normalizer: Normalizer
    rest_frame: np.ndarray  # (J, D) normalized rest pose describing this rig


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
    are fitted per skeleton FOR THE ACTIVE SPEC and cached.
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
    ) -> None:
        self.root = Path(root)
        self.manifest_dir = Path(manifest_dir)
        self.features = tuple(features)
        self.window = window or RandomCrop()
        self.conditioners: list[Conditioner] = [
            CONDITIONERS.get(name)() for name in conditioners
        ]
        self.augment_pipeline = AugmentPipeline(augmentations)
        self.records = index.query(split=split) if split else list(index.records)
        if not self.records:
            raise ValueError(f"no clips in the index for split={split!r}")

        self._rng = np.random.default_rng(seed)
        self._manifests: dict[str, SkeletonManifest] = {}
        self._normalizers: dict[str, Normalizer] = {}
        self._anims: dict[str, RigidBodyAnimation] = {}
        self._rest_frames: dict[str, np.ndarray] = {}

        # A clip yields more than one window under a deterministic policy, so the
        # dataset is indexed by (clip, window) rather than by clip.
        self._plan: list[tuple[int, int]] = []
        for clip_index, record in enumerate(self.records):
            frames = self._feature_frames(record)
            for window_index in range(self.window.count(frames)):
                self._plan.append((clip_index, window_index))

    def __len__(self) -> int:
        return len(self._plan)

    @property
    def spec(self) -> FeatureSpec:
        _, spec = self._extract(self.records[0])
        return spec

    def _manifest(self, skeleton: str) -> SkeletonManifest:
        if skeleton not in self._manifests:
            self._manifests[skeleton] = SkeletonManifest.load(
                self.manifest_dir / f"{skeleton}.yaml"
            )
        return self._manifests[skeleton]

    def _anim(self, record: ClipRecord) -> RigidBodyAnimation:
        if record.clip_id not in self._anims:
            self._anims[record.clip_id] = RigidBodyAnimation.load(self.root / record.path)
        return self._anims[record.clip_id]

    def _resolved(self, record: ClipRecord) -> ResolvedSkeleton:
        return resolve(self._manifest(record.skeleton), self._anim(record).names)

    def _extract(self, record: ClipRecord) -> tuple[np.ndarray, FeatureSpec]:
        return extract_features(
            self._anim(record), self._resolved(record), self.features
        )

    def _feature_frames(self, record: ClipRecord) -> int:
        return self._extract(record)[0].shape[0]

    def _normalizer(self, skeleton: str, spec: FeatureSpec) -> Normalizer:
        if skeleton not in self._normalizers:
            arrays = [
                self._extract(record)[0]
                for record in self.records
                if record.skeleton == skeleton
            ]
            self._normalizers[skeleton] = Normalizer.fit(arrays, spec)
        return self._normalizers[skeleton]

    def _rest_frame(
        self, record: ClipRecord, spec: FeatureSpec, normalizer: Normalizer
    ) -> np.ndarray:
        """One frame describing the skeleton, cached per skeleton.

        From the manifest's T-pose when it names one, otherwise the first frame
        of the skeleton's first clip -- the same fallback the reference takes.
        """
        skeleton = record.skeleton
        if skeleton in self._rest_frames:
            return self._rest_frames[skeleton]

        manifest = self._manifest(skeleton)
        if manifest.tpose is not None and manifest.tpose.is_file():
            from poseydon.io.bvh import BVH

            anim = BVH.read(manifest.tpose).to_animation().as_rigid_body(joint_translation="drop")
            raw, _ = extract_features(anim, resolve(manifest, anim.names), self.features)
        else:
            first = next(r for r in self.records if r.skeleton == skeleton)
            raw, _ = self._extract(first)

        frame = normalizer.normalize(raw[:1])[0]
        self._rest_frames[skeleton] = frame
        return frame

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

        base_rest_frame = self._rest_frame(record, spec, base_normalizer)
        rest_frame = edit.transport_row(base_rest_frame)

        view = ClipView(
            record=record,
            anim=anim,
            resolved=resolved,
            normalizer=normalizer,
            rest_frame=rest_frame,
        )
        return Item(
            features=window,
            spec=spec,
            start=start,
            source_length=features.shape[0],
            cond={c.name: c.extract(view) for c in self.conditioners},
        )
