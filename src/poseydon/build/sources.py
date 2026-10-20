"""Where a dataset's raw clips come from: the stage-1 ``source:`` of a data config.

A `Source` hands stage 1 (`scripts/process_dataset_truebones.py`) one rig's rest
pose and its clips as `poseydon.core.animation.Animation`, in the file's native
axes and units; the dataset's ``prepare:`` chain (``up_axis`` first) does the
rest. Two are registered:

- ``bvh_files``: every ``*.bvh`` under ``paths.raw/source/<Rig>/`` (Truebones).
- ``fbx_cache``: UniMate's Truebones FBX clips as `scripts/build_fbx_cache.py`
  cached them, with the skeleton rebuilt the way Blender's FBX importer would
  have, since UniMate exported through Blender.

Imports `poseydon.build.corpus` and `poseydon.build.index` (rest-clip choice,
action slugs), `poseydon.core` (`Animation`, `Registry`, rotations) and
`poseydon.io.bvh`.
"""

from __future__ import annotations

import csv
import math
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from poseydon.build.corpus import rest_source
from poseydon.build.index import action_slug, strip_skeleton_prefix
from poseydon.core.animation import Animation
from poseydon.core.registry import Registry
from poseydon.core.rotations import matrix_to_quat, rotation_part
from poseydon.core.skeleton import SkeletonManifest
from poseydon.io.bvh import BVH


@dataclass(frozen=True)
class SourceClip:
    """One raw clip: its action name, the motion, and the source file's channel
    layout (BVH only; ``None`` where the format has no such thing)."""

    action: str
    anim: Animation
    channels: tuple[tuple[str, ...], ...] | None


class Source(ABC):
    """Stage 1's input. ``rigs`` are the raw rig names (a rig is a species);
    everything else is per ``manifest`` (the rig). Raises `ValueError` for a
    rig or clip that cannot be read; stage 1 reports it and goes on."""

    def __init__(self) -> None:
        self.dropped: dict[str, str] = {}   # "Rig-action" -> why `clips` or the rig's grouping left it out

    @abstractmethod
    def rigs(self, data) -> list[str]: ...

    @abstractmethod
    def actions(self, data, manifest: SkeletonManifest) -> list[str]:
        """The action names `clip` accepts, sorted."""

    @abstractmethod
    def rest(self, data, manifest: SkeletonManifest) -> Animation:
        """The rig's rest pose as a one-frame animation."""

    def rest_channels(self, data, manifest: SkeletonManifest) -> tuple[tuple[str, ...], ...] | None:
        """The channel layout of the rest source (BVH only)."""
        return None

    @abstractmethod
    def clip(self, data, manifest: SkeletonManifest, action: str) -> SourceClip: ...

    def clips(self, data, manifest: SkeletonManifest) -> list[SourceClip]:
        """Every readable clip; one that raises `ValueError` is left out and recorded in `dropped`."""
        out = []
        for action in self.actions(data, manifest):
            try:
                out.append(self.clip(data, manifest, action))
            except ValueError as error:
                self.dropped[f"{manifest.name}-{action}"] = str(error)
        return out

    def skin(self, data, manifest: SkeletonManifest) -> dict[str, float] | None:
        """Largest skin weight per joint name on the rig's main mesh; ``None`` when
        the format carries no skin at all (a BVH). A rig whose file has no skin mesh
        is an empty mapping instead: every joint unskinned."""
        return None


#: Stage-1 sources (``source:`` in a dataset config).
SOURCES: Registry[Source] = Registry("source")


@SOURCES.register("bvh_files")
class BvhFiles(Source):
    """``paths.raw/source/<Rig>/<pattern>``. A clip's action is its file stem as
    `action_slug` and `strip_skeleton_prefix` give it. ``fps`` is checked against
    each clip unless the manifest declares its own."""

    def __init__(self, pattern: str = "*.bvh", fps: float | None = None) -> None:
        super().__init__()
        self.pattern = pattern
        self.fps = fps

    def _paths(self, data, manifest) -> dict[str, Path]:
        species_dir = Path(data.paths.raw) / "source" / manifest.name
        if not species_dir.is_dir():
            raise ValueError(f"no raw directory at {species_dir}, skipped")
        paths = sorted(species_dir.glob(self.pattern))
        if not paths:
            raise ValueError(f"no .bvh files found in {species_dir}, skipped")
        return {strip_skeleton_prefix(action_slug(p.stem), manifest.name): p for p in paths}

    def rigs(self, data) -> list[str]:
        source = Path(data.paths.raw) / "source"
        return sorted(p.name for p in source.iterdir() if p.is_dir()) if source.is_dir() else []

    def actions(self, data, manifest) -> list[str]:
        return list(self._paths(data, manifest))

    def _rest_path(self, data, manifest) -> Path:
        return rest_source(manifest, list(self._paths(data, manifest).values()))

    def rest(self, data, manifest) -> Animation:
        return BVH.read(self._rest_path(data, manifest)).to_animation()

    def rest_channels(self, data, manifest):
        return BVH.read(self._rest_path(data, manifest)).channels

    def clip(self, data, manifest, action) -> SourceClip:
        bvh = BVH.read(self._paths(data, manifest)[action])
        anim = bvh.to_animation()
        fps = manifest.fps or self.fps
        if fps is not None and not math.isclose(fps, anim.fps, rel_tol=1e-3):
            raise ValueError(f"the dataset requires {fps} fps but this clip is {anim.fps:.4f}")
        return SourceClip(action, anim, bvh.channels)


@dataclass(frozen=True)
class _Armature:
    """A cached clip's Blender-like armature: node indices (parents first), the
    parent of each as an index into ``nodes`` (-1 for the root), and the skeleton
    trees left out."""

    nodes: tuple[int, ...]
    names: tuple[str, ...]
    parents: np.ndarray
    dropped_trees: tuple[str, ...]


def armature_of(d) -> _Armature:
    """The armature Blender's FBX importer would build from a cached clip ``d``.

    The rule (an approximation, calibrated against UniMate's published joint
    lists in the replication report):

    1. A node is a bone when its FBX attribute is ``eSkeleton`` or a skin cluster
       links it (a weight sum above zero on any mesh).
    2. Non-bone nodes BETWEEN two bones are bones too, so a bone's parent is
       always a bone (Blender cannot parent a bone to an empty). Non-bone
       ancestors ABOVE the topmost bone are not bones: they are the armature
       object's own transform, which the root bone's world matrix already holds.
    3. Each topmost bone roots one tree (one armature). The main armature has
       the most bones (UniMate's ``choose_main_armature``), ties broken by skin
       weight sum (its ``prune_secondary_roots``); the other trees are dropped
       and named in ``dropped_trees``.

    Not modelled: Blender's bone re-orientation, FBX unit scale on the armature
    (units stay the file's own), and UniMate's later pruning (`prune_unskinned`).
    """
    types, parents, names = d["types"], d["parents"], d["names"]
    n = len(names)
    weight = d["skin_sum"].sum(0) if len(d["skin_mesh"]) else np.zeros(n)
    bone = (types == "eSkeleton") | (weight > 0)

    def chain(j):
        out = []
        while j >= 0:
            out.append(j)
            j = int(parents[j])
        return out  # j, its parent, ..., the scene root

    root_of: dict[int, int] = {}   # bone -> topmost bone of its tree
    for j in np.flatnonzero(bone):
        up = chain(int(j))
        top = max(i for i, k in enumerate(up) if bone[k])
        for k in up[: top + 1]:
            root_of[k] = up[top]
    trees: dict[int, list[int]] = {}
    for k, r in sorted(root_of.items()):
        trees.setdefault(r, []).append(k)
    ranked = sorted(trees, key=lambda r: (len(trees[r]), weight[trees[r]].sum()), reverse=True)
    main = ranked[0]
    nodes = trees[main]          # indices ascending; check parents come first
    position = {k: i for i, k in enumerate(nodes)}
    if any(int(parents[k]) in position and position[int(parents[k])] > position[k] for k in nodes):
        nodes = sorted(nodes, key=lambda k: len(chain(k)))
        position = {k: i for i, k in enumerate(nodes)}
    return _Armature(
        nodes=tuple(nodes),
        names=tuple(str(names[k]) for k in nodes),
        parents=np.array([position.get(int(parents[k]), -1) for k in nodes], dtype=np.int32),
        dropped_trees=tuple(
            f"{names[r]} ({len(trees[r])} bones, skin {weight[trees[r]].sum():.1f})" for r in ranked[1:]
        ),
    )


def _local(world: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """``(..., J, 4, 4)`` world matrices to local ones (the root's stays world)."""
    local = world.copy()
    has = parents >= 0
    local[..., has, :, :] = np.linalg.inv(world[..., parents[has], :, :]) @ world[..., has, :, :]
    return local


def _animation(local: np.ndarray, offsets: np.ndarray | None, parents, names, fps) -> Animation:
    """An `Animation` from ``(F, J, 4, 4)`` local matrices (scale removed from the rotation)."""
    q = matrix_to_quat(rotation_part(local[..., :3, :3]))
    # a quaternion and its negation are one rotation: keep each joint's path continuous
    flip = np.cumprod(np.where((q[1:] * q[:-1]).sum(-1) < 0, -1.0, 1.0), axis=0)
    q[1:] *= flip[..., None]
    t = local[..., :3, 3]
    return Animation(q, t, t[0].copy() if offsets is None else offsets, parents, names, fps)


@SOURCES.register("fbx_cache")
class FbxCache(Source):
    """UniMate's Truebones clips from the FBX cache (`scripts/build_fbx_cache.py`).

    ``index`` is UniMate's ``clips.csv`` (rows with ``status == linked``; a rig
    is a ``group``, a clip's action the part of ``normalized_name`` after the
    first ``-``), ``cache`` the directory holding ``<Species>/<Action>.npz``.
    ``skip`` is a ``filtered_clips.txt``: ``Species-Action`` first on a line,
    ``#`` starts a comment. A clip shorter than ``min_frames`` or at another
    ``fps`` than the given one is refused.

    Rig variants: a species' clips are grouped by their armature's joint-name
    set (`armature_of`); the group with the most clips (the first alphabetically
    on a tie) defines the rig. A clip of another group is mapped by name when it
    holds every joint of the rig (its other joints are folded away, each joint's
    local transform taken against the rig's own parent), else it raises
    `ValueError` naming the missing joints. The rest pose is the first clip of
    the main group, local = parent bind-inverse times bind; a node without a bind
    uses its frame-0 world. Clips: local = parent world-inverse times world per
    frame, the root's the full world (armature object included).
    """

    def __init__(self, index: str, cache: str, skip: str | None = None,
                 min_frames: int = 5, fps: float | None = 30) -> None:
        super().__init__()
        self.cache, self.min_frames, self.fps = Path(cache), min_frames, fps
        skipped = set()
        if skip is not None:
            for line in Path(skip).read_text().splitlines():
                line = line.split("#")[0].strip()
                if line:
                    skipped.add(line.split()[0])
        self._actions: dict[str, list[str]] = {}
        with open(index) as handle:
            for row in csv.DictReader(handle):
                if row["status"] != "linked":
                    continue
                species, action = row["normalized_name"].removesuffix(".fbx").split("-", 1)
                if f"{species}-{action}" not in skipped:
                    self._actions.setdefault(species, []).append(action)
        self._actions = {s: sorted(a) for s, a in sorted(self._actions.items())}
        self._rigs: dict[str, tuple] = {}
        self._offsets: dict[str, np.ndarray] = {}
        self.dropped_trees: dict[str, tuple[str, ...]] = {}

    def rigs(self, data=None) -> list[str]:
        return list(self._actions)

    def actions(self, data, manifest) -> list[str]:
        return self._actions[manifest.name]

    def _load(self, species: str, action: str, *, world: bool = True) -> dict:
        path = self.cache / species / f"{action}.npz"
        if not path.is_file():
            raise ValueError(f"{path}: not cached (scripts/build_fbx_cache.py)")
        with np.load(path) as z:
            skip = {"world", "frames"} if not world else set()
            d = {k: z[k] for k in z.files if k not in skip}
        raw = Path(str(d["source"]))
        if raw.is_file() and (raw.stat().st_size != int(d["source_size"])
                              or raw.stat().st_mtime != float(d["source_mtime"])):
            raise ValueError(f"{path}: stale, {raw} changed since it was cached")
        return d

    def _rig(self, species: str):
        """``(armature, main-group actions, representative cache)`` of a species."""
        if species not in self._rigs:
            armatures = {}
            for a in self._actions[species]:
                try:
                    armatures[a] = armature_of(self._load(species, a, world=False))
                except ValueError as error:   # `clip` raises the same for this action
                    self.dropped[f"{species}-{a}"] = str(error)
            if not armatures:
                raise ValueError(f"{species}: no clip could be loaded from the cache")
            groups = Counter(frozenset(x.names) for x in armatures.values())
            key = max(groups, key=lambda g: groups[g])   # first-inserted wins a tie
            first = next(a for a, x in armatures.items() if frozenset(x.names) == key)
            self._rigs[species] = (armatures[first], first)
            self.dropped_trees[species] = armatures[first].dropped_trees
        return self._rigs[species]

    def rest(self, data, manifest) -> Animation:
        armature, first = self._rig(manifest.name)
        d = self._load(manifest.name, first)
        bind = d["bind"][list(armature.nodes)]
        missing = np.isnan(bind).any(axis=(-2, -1))
        bind[missing] = d["world"][0][list(armature.nodes)][missing]
        local = _local(bind, armature.parents)[None]
        return _animation(local, None, armature.parents, armature.names, self.fps or float(d["fps"]))

    def clip(self, data, manifest, action) -> SourceClip:
        armature, _ = self._rig(manifest.name)
        d = self._load(manifest.name, action)
        label = f"{manifest.name}-{action}"
        fps = manifest.fps or self.fps
        if fps is not None and not math.isclose(fps, float(d["fps"]), rel_tol=1e-3):
            raise ValueError(f"the dataset requires {fps} fps but this clip is {float(d['fps']):.4f}")
        if len(d["frames"]) < self.min_frames:
            raise ValueError(f"{len(d['frames'])} frames, fewer than {self.min_frames}")
        names = [str(x) for x in d["names"]]
        missing = [n for n in armature.names if n not in names]
        if missing:
            raise ValueError(f"{label} lacks {len(missing)} of the rig's joints ({', '.join(missing[:4])}...)")
        world = d["world"][:, [names.index(n) for n in armature.names]]
        return SourceClip(action, _animation(_local(world, armature.parents), self._rest_offsets(data, manifest),
                                             armature.parents, armature.names, float(d["fps"])), None)

    def _rest_offsets(self, data, manifest) -> np.ndarray:
        if manifest.name not in self._offsets:
            self._offsets[manifest.name] = self.rest(data, manifest).offsets
        return self._offsets[manifest.name]

    def skin(self, data, manifest) -> dict[str, float] | None:
        armature, first = self._rig(manifest.name)
        d = self._load(manifest.name, first, world=False)
        if not len(d["skin_mesh"]):
            return {}   # no skin mesh: every joint unskinned, as UniMate treats it (None is a BVH, no skin information)
        row = int(np.argmax(d["skin_vertices"]))
        return {n: float(d["skin_max"][row, k]) for n, k in zip(armature.names, armature.nodes)}
