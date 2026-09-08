"""BVH corpus to aligned RigidBodyAnimation files plus a corpus index."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from poseydon.core.animation import RigidBodyAnimation
from poseydon.core.skeleton import ResolvedSkeleton, SkeletonManifest, resolve
from poseydon.ingest.align import AlignmentParams, align, compute_alignment_params
from poseydon.ingest.index import (
    SEPARATOR,
    ClipRecord,
    CorpusIndex,
    action_slug,
    clip_id,
    strip_skeleton_prefix,
)
from poseydon.io.bvh import BVH

ALIGNED_DIRNAME = "aligned"


def available_rigs(rig_root: str | Path) -> list[str]:
    """Rig names under ``rig_root``, longest first.

    A rig is a DIRECTORY containing a ``manifest.yaml``, which is why the old
    ``_``-prefix convention for shared fragments is no longer needed: a bare
    ``_base.yaml`` is a file, not a directory. A directory alone does not
    qualify either -- ``rigs/<Rig>/`` also holds derived artefacts (a
    prepared clip, a mesh, statistics), so one can exist before anyone has
    authored a manifest for it. Longest first matters -- given both `Goat`
    and `GoatKid`, a clip named `GoatKid_walk` must match `GoatKid`.
    """
    root = Path(rig_root)
    if not root.is_dir():
        return []
    names = [
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "manifest.yaml").is_file()
    ]
    return sorted(names, key=lambda name: (-len(name), name))


def infer_skeleton(path: Path, manifest_dir: str | Path) -> str:
    """Work out which skeleton a source BVH belongs to.

    Source corpora do not follow PoseYdon's `__` clip-id convention -- that
    convention describes ids we generate, not filenames we are given. So resolve
    against the manifests that actually exist:

    1. an explicit ``<Skeleton>__<whatever>`` filename, PoseYdon's own convention;
    2. the containing directory name, which is how raw Truebones is laid out
       (``Truebone_Z-OO/<Species>/*.bvh``);
    3. otherwise the longest manifest name the filename starts with.
    """
    known = available_rigs(manifest_dir)
    declared = path.stem.split(SEPARATOR, 1)[0] if SEPARATOR in path.stem else None
    if declared in known:
        return declared
    if path.parent.name in known:
        return path.parent.name
    for name in known:
        if path.stem.startswith(name):
            return name
    raise ValueError(
        f"cannot tell which skeleton `{path.name}` belongs to. Put it in a "
        f"directory named after its skeleton, name the file with the skeleton as "
        f"a prefix, or pass an explicit skeleton. Known: {', '.join(sorted(known))}"
    )


@dataclass
class IngestResult:
    index: CorpusIndex = field(default_factory=CorpusIndex)
    written: list[Path] = field(default_factory=list)
    skipped: list[tuple[Path, str]] = field(default_factory=list)


def skeleton_alignment_params(
    manifest: SkeletonManifest,
    fallback: RigidBodyAnimation,
    resolved: ResolvedSkeleton,
) -> AlignmentParams:
    """Alignment constants for a skeleton, from the first frame of its first clip.

    ``SkeletonManifest.tpose`` used to be consulted here, but no manifest ever
    declared it, so this fallback has always been the only path -- a guard
    that reads as working is worse than no guard.
    """
    reference = fallback
    return compute_alignment_params(reference, resolved)


def ingest_clip(
    bvh_path: str | Path,
    manifest: SkeletonManifest,
    out_dir: str | Path,
    split: str = "train",
    params: AlignmentParams | None = None,
) -> ClipRecord:
    """Align one BVH and write one full-length RigidBodyAnimation. Never chunks.

    ``params`` are the skeleton-level constants; when omitted they are derived
    from this clip, which is correct only for a single-clip ingest.
    """
    bvh_path = Path(bvh_path)
    out_dir = Path(out_dir)

    # Training is where bones must be rigid -- features, the IK solver and the
    # models all assume constant bone lengths -- so the corpus keeps per-joint
    # translation and it is dropped here, explicitly, at that boundary.
    anim = BVH.read(bvh_path).to_animation().as_rigid_body(joint_translation="drop")

    # Relative, not absolute: BVH stores a frame TIME, so a round rate is a
    # rounded repeating decimal on disk. Truebones writes 0.033333, which reads
    # back as 30.00003 fps -- a real 30 fps file that an exact check rejects.
    # 1e-3 accepts that rounding while still separating 30 from 24 or 25.
    if manifest.fps is not None and not math.isclose(manifest.fps, anim.fps, rel_tol=1e-3):
        raise ValueError(
            f"{bvh_path.name}: manifest requests {manifest.fps} fps but the source "
            f"is {anim.fps:.4f} fps, and resampling is not implemented. Set "
            "`fps: null` to keep the source rate."
        )

    resolved = resolve(manifest, anim.names)
    if params is None:
        params = skeleton_alignment_params(manifest, anim, resolved)
    aligned = align(anim, resolved, params)

    action = strip_skeleton_prefix(action_slug(bvh_path.stem), manifest.name)
    identifier = clip_id(manifest.name, action)
    relative = f"{ALIGNED_DIRNAME}/{identifier}.npz"
    destination = out_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    aligned.save(destination)

    return ClipRecord(
        clip_id=identifier,
        skeleton=manifest.name,
        action=action,
        split=split,
        n_frames=aligned.n_frames,
        fps=aligned.fps,
        path=relative,
        tags=manifest.tags,
    )


def ingest_corpus(
    bvh_paths: Iterable[str | Path],
    manifest_dir: str | Path,
    out_dir: str | Path,
    split: str = "train",
    skeleton_of: Callable[[Path], str] | None = None,
) -> IngestResult:
    """Ingest many BVHs, collecting failures instead of aborting the run."""
    manifest_dir = Path(manifest_dir)
    out_dir = Path(out_dir)
    result = IngestResult()
    cache: dict[str, SkeletonManifest] = {}
    params_cache: dict[str, AlignmentParams] = {}

    for raw in bvh_paths:
        path = Path(raw)
        try:
            skeleton = skeleton_of(path) if skeleton_of else infer_skeleton(path, manifest_dir)
            if skeleton not in cache:
                cache[skeleton] = SkeletonManifest.load(manifest_dir / skeleton / "manifest.yaml")
            manifest = cache[skeleton]
            if skeleton not in params_cache:
                first = BVH.read(path).to_animation().as_rigid_body(joint_translation="drop")
                params_cache[skeleton] = skeleton_alignment_params(
                    manifest, first, resolve(manifest, first.names)
                )
            record = ingest_clip(
                path, manifest, out_dir, split=split, params=params_cache[skeleton]
            )
        except Exception as error:  # noqa: BLE001 - one bad file must not stop a corpus
            result.skipped.append((path, f"{type(error).__name__}: {error}"))
            continue
        result.index.add(record)
        result.written.append(out_dir / record.path)

    return result
