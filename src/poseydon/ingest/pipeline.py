"""BVH corpus to aligned Anim files plus a corpus index."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from poseydon.core.skeleton import SkeletonManifest, resolve
from poseydon.ingest.align import align
from poseydon.ingest.index import (
    ClipRecord,
    CorpusIndex,
    action_slug,
    clip_id,
    strip_skeleton_prefix,
)
from poseydon.io.bvh import load_bvh

ALIGNED_DIRNAME = "aligned"


def available_skeletons(manifest_dir: str | Path) -> list[str]:
    """Manifest names available in ``manifest_dir``, longest first.

    Longest-first matters: given both `Goat` and `GoatKid`, a clip named
    `GoatKid_walk` must match `GoatKid`, not `Goat`.
    """
    names = [p.stem for p in Path(manifest_dir).glob("*.yaml") if not p.stem.startswith("_")]
    return sorted(names, key=len, reverse=True)


def infer_skeleton(path: Path, manifest_dir: str | Path) -> str:
    """Work out which skeleton a source BVH belongs to.

    Source corpora do not follow PoseYdon's `__` clip-id convention -- that
    convention describes ids we generate, not filenames we are given. So resolve
    against the manifests that actually exist:

    1. the containing directory name, which is how raw Truebones is laid out
       (``Truebone_Z-OO/<Species>/*.bvh``);
    2. otherwise the longest manifest name the filename starts with.
    """
    known = available_skeletons(manifest_dir)
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


def ingest_clip(
    bvh_path: str | Path,
    manifest: SkeletonManifest,
    out_dir: str | Path,
    split: str = "train",
) -> ClipRecord:
    """Align one BVH and write one full-length Anim. Never chunks."""
    bvh_path = Path(bvh_path)
    out_dir = Path(out_dir)

    anim = load_bvh(bvh_path)

    if manifest.fps is not None and abs(manifest.fps - anim.fps) > 1e-6:
        raise ValueError(
            f"{bvh_path.name}: manifest requests {manifest.fps} fps but the source "
            f"is {anim.fps:.4f} fps, and resampling is not implemented. Set "
            "`fps: null` to keep the source rate."
        )

    resolved = resolve(manifest, anim.names)
    aligned, _params = align(anim, resolved)

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

    for raw in bvh_paths:
        path = Path(raw)
        try:
            skeleton = skeleton_of(path) if skeleton_of else infer_skeleton(path, manifest_dir)
            if skeleton not in cache:
                cache[skeleton] = SkeletonManifest.load(manifest_dir / f"{skeleton}.yaml")
            record = ingest_clip(path, cache[skeleton], out_dir, split=split)
        except Exception as error:  # noqa: BLE001 - one bad file must not stop a corpus
            result.skipped.append((path, f"{type(error).__name__}: {error}"))
            continue
        result.index.add(record)
        result.written.append(out_dir / record.path)

    return result
