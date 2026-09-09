"""The four build passes: clip, skeleton, stats, index.

Every file here has exactly one writer. Stage 1 owns prepare.npz, mesh.npz and
the prepared BVH; this stage owns skeleton.npz, stats.npz, the clip .npz and the
index. Only stats.npz depends on the feature schema, so changing `features:`
re-runs one pass over already-prepared motion rather than rebuilding a corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from poseydon.build.corpus import PerRigDirectory, rest_action, write_labels
from poseydon.build.index import ClipRecord, CorpusIndex, clip_id
from poseydon.build.names import Humanize
from poseydon.build.prepare import RigTransform
from poseydon.core.skeleton import SkeletonManifest, resolve
from poseydon.data.normalize import BlockPolicy, Normalizer
from poseydon.features import extract_features
from poseydon.features.reduce import apply_reduction, build_reduction
from poseydon.io.bvh import BVH


@dataclass(frozen=True)
class BuildConfig:
    root: Path                      # the prepared corpus this reads
    schema: tuple[str, ...]
    corpus: PerRigDirectory
    names: Humanize
    normalize: tuple[BlockPolicy, ...]
    #: Where artefacts are written. Defaults to `root`; tests point it at a
    #: tmp_path so a test never mutates the corpus another test reads.
    out: Path | None = None
    reduce_tolerance: float = 1e-8
    relabel: bool = False

    def __post_init__(self) -> None:
        if self.out is None:
            object.__setattr__(self, "out", self.root)
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "out", Path(self.out))

    def manifest(self, rig: str) -> SkeletonManifest:
        return SkeletonManifest.load(self.root / "rigs" / rig / "manifest.yaml")

    def transform(self, rig: str) -> RigTransform:
        return RigTransform.load(self.root / "rigs" / rig / "prepare.npz")


@dataclass
class BuildResult:
    clips: int = 0
    rigs: int = 0
    warnings: list[str] = field(default_factory=list)


def build_clips(config: BuildConfig, rig: str) -> int:
    """Prepared BVH -> arrays, plus THIS clip's facing quaternion.

    The facing is clip-scoped and unrecoverable from a file that already faces
    +Z, so folding the wrong clip's rotation in would be invisible until someone
    tried to un-prepare a generated clip.
    """
    recorded = config.transform(rig)
    out_dir = config.out / "clips" / rig
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for path in config.corpus.clips(config.root, rig):
        action = path.stem
        anim = BVH.read(path).to_animation()
        np.savez_compressed(
            out_dir / f"{action}.npz",
            rotations=anim.rotations,
            translations=anim.translations,
            offsets=anim.offsets,
            parents=anim.parents,
            names=np.array(anim.names),
            fps=np.float64(anim.fps),
            facing=recorded.clip_params[action]["face_axis"]["rotation"],
        )
        write_labels(
            out_dir / f"{action}.yaml",
            {"split": "train", "action": action},
            relabel=config.relabel,
        )
        written += 1
    return written


def build_skeleton(config: BuildConfig, rig: str) -> None:
    """Rig-level geometry, names and the reduction map.

    Offsets come from the REST clip specifically: FaceAxis is clip-scoped and
    rotate_rig turns OFFSETS with the motion, so prepared clips of one rig do
    not share an OFFSET block. Which clip that is comes from the manifest via
    rest_action -- never from a filename match, since Crab's is `walk`.
    """
    manifest = config.manifest(rig)
    rest = BVH.read(
        config.root / "clips" / rig / f"{rest_action(manifest)}.bvh"
    ).to_animation()

    reduction = build_reduction(rest, tolerance=config.reduce_tolerance)
    rig_params = config.transform(rig).rig_params

    payload: dict[str, np.ndarray] = {
        "offsets": rest.offsets,
        "parents": rest.parents,
        "names": np.array(rest.names),
        "humanized": np.array(config.names(rest.names)),
        "reduction_source_of": np.array(reduction.source_of),
        "reduction_source_names": np.array(reduction.source_names),
        "rest_action": np.array(rest_action(manifest)),
    }
    # Fold stage 1's rig constants in, so the training-facing file is
    # self-contained and stage 2 can be re-run without Blender.
    for stage, params in rig_params.items():
        for key, value in params.items():
            payload[f"prepare/{stage}/{key}"] = np.asarray(value)

    out = config.out / "rigs" / rig
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "skeleton.npz", **payload)


def build_stats(config: BuildConfig, rig: str) -> None:
    """The one schema-dependent artefact."""
    manifest = config.manifest(rig)
    arrays, spec = [], None
    for path in config.corpus.clips(config.root, rig):
        anim = BVH.read(path).to_animation().as_rigid_body(joint_translation="drop")
        reduction = build_reduction(anim, tolerance=config.reduce_tolerance)
        reduced = apply_reduction(anim, reduction)
        features, spec = extract_features(
            reduced, resolve(manifest, reduced.names), config.schema
        )
        arrays.append(features)

    if spec is None:
        raise ValueError(f"{rig}: no prepared clips, so no statistics to fit")

    out = config.out / "rigs" / rig
    out.mkdir(parents=True, exist_ok=True)
    Normalizer.fit(arrays, spec, list(config.normalize)).save(out / "stats.npz")


def build_index(config: BuildConfig) -> CorpusIndex:
    """One flat row per clip, joining rig-level manifest fields.

    `tags` lives once, in the manifest, and is joined here -- rather than copied
    into every ClipRecord, which wrote `[quadruped, mammal]` twenty-two times
    for BrownBear.
    """
    index = CorpusIndex()
    for rig in config.corpus.rigs(config.root):
        manifest = config.manifest(rig)
        for path in sorted((config.out / "clips" / rig).glob("*.npz")):
            with np.load(path) as data:
                n_frames = int(data["rotations"].shape[0])
                fps = float(data["fps"])
            index.add(
                ClipRecord(
                    clip_id=clip_id(rig, path.stem),
                    skeleton=rig,
                    action=path.stem,
                    split="train",
                    n_frames=n_frames,
                    fps=fps,
                    path=str(Path("clips") / rig / path.name),
                    tags=tuple(manifest.tags),
                )
            )
    index.save(config.out / "index.jsonl")
    return index


def build_all(
    config: BuildConfig,
    rigs: list[str] | None = None,
    stats_only: bool = False,
) -> BuildResult:
    """Run the passes over every rig, collecting per-rig failures.

    Each rig is wrapped individually and deliberately: plan A1's Tukan crash
    killed the whole 73-rig loop mid-run because one stage raised outside a
    per-rig guard, leaving a partial corpus behind.
    """
    result = BuildResult()
    for rig in rigs or config.corpus.rigs(config.root):
        try:
            if not stats_only:
                result.clips += build_clips(config, rig)
                build_skeleton(config, rig)
            build_stats(config, rig)
            result.rigs += 1
        except Exception as error:  # noqa: BLE001 - collect, don't abort the corpus
            result.warnings.append(f"{rig}: {type(error).__name__}: {error}")

    if not stats_only:
        build_index(config)
    return result
