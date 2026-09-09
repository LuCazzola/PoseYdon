"""Finding the prepared clips, and the label files beside them."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from poseydon.build.index import action_slug, strip_skeleton_prefix
from poseydon.core.skeleton import SkeletonManifest
from poseydon.io.bvh import BVH


@dataclass(frozen=True)
class PerRigDirectory:
    """The entity-first layout: `clips/<Rig>/<action>.bvh`."""

    pattern: str = "*.bvh"

    def rigs(self, root: str | Path) -> list[str]:
        clips = Path(root) / "clips"
        if not clips.is_dir():
            return []
        return sorted(p.name for p in clips.iterdir() if p.is_dir())

    def clips(self, root: str | Path, rig: str) -> list[Path]:
        return sorted((Path(root) / "clips" / rig).glob(self.pattern))


def read_labels(path: str | Path) -> dict:
    """A clip's authored label file, or an empty mapping when absent."""
    path = Path(path)
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: a label file must be a YAML mapping")  # noqa: TRY004
    return loaded


def write_labels(path: str | Path, derived: dict, relabel: bool = False) -> None:
    """Write a label file, without ever destroying authored content.

    A rebuild writes only when the file is absent. `--relabel` refreshes the
    keys the build derives and preserves every key it did not write, so a
    hand-written `text:` or a changed `split:` survives.
    """
    path = Path(path)
    if path.is_file() and not relabel:
        return

    merged = {**read_labels(path), **derived} if relabel else dict(derived)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(merged, sort_keys=True))


def rest_action(manifest: SkeletonManifest) -> str:
    """The prepared-clip action slug for this rig's declared rest pose.

    Consumers working on `clips/<Rig>/` need the slug, not the raw filename;
    deriving it here keeps one source of truth in the manifest.
    """
    if manifest.rest_pose is None:
        raise ValueError(f"{manifest.name}: manifest declares no `rest_pose`")
    stem = Path(manifest.rest_pose).stem
    return strip_skeleton_prefix(action_slug(stem), manifest.name)


def rest_source(manifest: SkeletonManifest, clip_paths: list[Path]) -> Path:
    """The raw file supplying this rig's rest pose, validated against the corpus.

    The choice is authored -- it is a judgement about which clip's first frame
    is a neutral pose, and no rule gets that right (matching "idle" as a
    substring picks Lion's __DeathIdle.bvh, Jaguar's __LieIdle.bvh and Trex's
    __idle_attack.bvh). But an authored value never overrides the corpus's own
    evidence: the declared file must carry the rig's MODAL joint set, because
    a rest pose disagreeing with the clips makes `RestRelative` reject every
    one of them. Trex is why this guard is not ceremony -- its natural neutral
    pick, __STILL.bvh, is that rig's outlier file (66 joints against 78).

    Raises rather than falling back. A rig whose rest pose cannot be resolved
    is a data error the build must stop on; the previous behaviour returned
    "the first file in the modal set", which is what gave Crab a rest geometry
    fitted from __Attack1.bvh.
    """
    if manifest.rest_pose is None:
        raise ValueError(
            f"{manifest.name}: manifest declares no `rest_pose`; every rig must "
            "name the clip whose first frame is its rest pose"
        )

    by_name = {path.name: path for path in clip_paths}
    chosen = by_name.get(manifest.rest_pose)
    if chosen is None:
        raise ValueError(
            f"{manifest.name}: declared rest_pose `{manifest.rest_pose}` is not "
            f"among this rig's {len(clip_paths)} raw clips"
        )

    names_by_path = {path: BVH.read_names(path) for path in clip_paths}
    modal_names, modal_count = Counter(names_by_path.values()).most_common(1)[0]
    if names_by_path[chosen] != modal_names:
        raise ValueError(
            f"{manifest.name}: declared rest_pose `{manifest.rest_pose}` has "
            f"{len(names_by_path[chosen])} joints but the rig's modal skeleton "
            f"has {len(modal_names)} ({modal_count}/{len(clip_paths)} clips); "
            "a rest pose disagreeing with the clips rejects every one of them"
        )
    return chosen
