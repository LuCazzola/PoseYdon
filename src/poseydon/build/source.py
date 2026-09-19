"""Load raw motion through the modality declared by a rig manifest."""
from __future__ import annotations


from pathlib import Path
from typing import Callable, TYPE_CHECKING

from poseydon.core.animation import Animation
from poseydon.io.bvh import BVH

if TYPE_CHECKING:
    from poseydon.core.skeleton import SkeletonManifest


def _load_bvh(path: Path) -> Animation:
    return BVH.read(path).to_animation()


def _load_fbx(path: Path) -> Animation:
    from poseydon.io.fbx import FBX
    return FBX.read(path).to_animation()


class SourceMotion:
    @staticmethod
    def load(manifest: SkeletonManifest, raw_rig_dir: Path, *,
             bvh_loader: Callable[[Path], Animation] = _load_bvh,
             fbx_loader: Callable[[Path], Animation] = _load_fbx) -> Animation:
        modality = manifest.authority.motion
        relative = getattr(manifest.artifacts, modality)
        if relative is None:
            raise ValueError(f"{manifest.name}: motion authority `{modality}` has no source artifact")
        path = Path(raw_rig_dir) / relative
        return bvh_loader(path) if modality == "bvh" else fbx_loader(path)
