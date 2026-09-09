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

from poseydon.build.corpus import PerRigDirectory, read_labels, rest_action, write_labels
from poseydon.build.index import ClipRecord, CorpusIndex, clip_id
from poseydon.build.names import Humanize
from poseydon.build.prepare import RigTransform
from poseydon.core.animation import Animation
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
    #: Non-fatal, per-rig notices that do not withhold any artefact -- e.g. a
    #: mesh/skeleton joint-count mismatch (`_check_mesh_matches_skeleton`).
    #: A non-empty `warnings` list must never, by itself, fail the build.
    warnings: list[str] = field(default_factory=list)
    #: Rigs (or the index pass) that raised and produced NO artefact for that
    #: step. Distinct from `warnings` on purpose: `scripts/build_features.py`
    #: exits non-zero exactly when this is non-empty, so a rig that merely
    #: warns (Camel, Goat) must never land here.
    failures: list[str] = field(default_factory=list)


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


def _n_real_joints(anim: Animation) -> int:
    """BVH joint count minus End Sites -- the count an FBX skeleton, which has
    no End Site concept, is comparable against.

    An End Site is exactly a childless joint (see `EnforceRigid.fit` and
    `ScaleToMeanBoneLength.fit` in `prepare.py`, which use the same test): every
    real leaf gets one synthesized under it on BVH read, so a joint with no
    child in the prepared animation is never a real body joint.
    """
    has_child = np.zeros(anim.n_joints, dtype=bool)
    real = anim.parents >= 0
    has_child[anim.parents[real]] = True
    return int(has_child.sum())


def _check_mesh_matches_skeleton(rest: Animation, mesh_path: Path, rig: str) -> str | None:
    """Detect -- but do not refuse -- a `mesh.npz` whose joint count disagrees
    with the rest skeleton. Returns a warning message, or `None` if they agree.

    The FBX path (`process_dataset_truebones_fbx.py`) never runs `PromoteRoot`,
    so for the 14 rigs `PROMOTE_ROOT` names, `mesh.npz`'s skin weights are
    indexed by the UN-promoted joint order while `skeleton.npz`'s joints are
    promoted. Design spec 2026-09-08 Stage 2 says stage 2 must not trust
    positional alignment between the two -- but `skeleton.npz` and `stats.npz`
    never read `mesh.npz` at all (mesh folding does not exist yet; Task 5 left
    it out on purpose), so a mismatch here has no bearing on the TRAINING
    artefacts and must not withhold them. It matters only at the point mesh
    folding is implemented: whoever adds that MUST call this check (or its
    successor) and refuse to fold a disagreeing mesh -- do not silently trust
    positional alignment there either.
    """
    with np.load(mesh_path, allow_pickle=True) as mesh:
        mesh_n = len(mesh["joint_names"])
    rest_n = _n_real_joints(rest)
    if mesh_n != rest_n:
        return (
            f"{rig}: mesh.npz has {mesh_n} joints but the rest skeleton has "
            f"{rest_n} real joints -- the FBX path does not run PromoteRoot, "
            f"so this rig's mesh.npz is indexed against an un-promoted "
            f"skeleton and would need PromoteRoot's mapping to fold correctly "
            f"(design spec 2026-09-08 Stage 2, 'the FBX path does not "
            f"promote'). skeleton.npz and stats.npz are written from the BVH "
            f"rest pose regardless -- mesh.npz is not folded into them."
        )
    return None


def build_skeleton(config: BuildConfig, rig: str) -> str | None:
    """Rig-level geometry, names and the reduction map.

    Offsets come from the REST clip specifically: FaceAxis is clip-scoped and
    rotate_rig turns OFFSETS with the motion, so prepared clips of one rig do
    not share an OFFSET block. Which clip that is comes from the manifest via
    rest_action -- never from a filename match, since Crab's is `walk`.

    Returns a mesh/skeleton mismatch warning (see `_check_mesh_matches_skeleton`),
    or `None`. Either way, `skeleton.npz` is written -- a mismatch is reported,
    not refused.

    Loading requires ``allow_pickle=True``. `prepare/enforce_rigid/source_channels`
    is genuinely ragged (a tuple of channel names per joint, of differing length)
    and must stay an object array; there is no fixed-width encoding for it. See
    `RigTransform`'s docstring in `build/prepare.py` for the same trap on
    `prepare.npz`, which this file inherits by folding `rig_params` in.
    """
    manifest = config.manifest(rig)
    rest = BVH.read(
        config.root / "clips" / rig / f"{rest_action(manifest)}.bvh"
    ).to_animation()

    mesh_path = config.root / "rigs" / rig / "mesh.npz"
    mismatch = _check_mesh_matches_skeleton(rest, mesh_path, rig) if mesh_path.is_file() else None

    # This reduction must equal the one `build_stats` fits under, or
    # `skeleton.npz`'s `reduction_source_of` would describe a different lens
    # than `stats.npz`'s moments were measured through. Two things differ
    # between the calls, and neither can change which joints are removed:
    #
    # * a different CLIP. Prepared clips of one rig do not share an OFFSET
    #   block (FaceAxis is clip-scoped and turns offsets with the motion --
    #   see this docstring above), but `_next_removal` selects on
    #   `norm(offsets)`, `parents` and `names`, and a rotation leaves a norm
    #   unchanged.
    # * the rigid-body form. `build_stats` reduces
    #   `as_rigid_body(joint_translation="drop")`; that drops per-joint
    #   TRANSLATION channels and touches offsets/parents/names not at all.
    #
    # Load-bearing and derived nowhere else. If `as_rigid_body` ever starts
    # rewriting offsets, or `_next_removal` ever reads a rotation-dependent
    # quantity, this comment and its twin at `build_stats` are where it breaks.
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
            array = np.asarray(value)
            # `RestRelative.fit` stores `names` as a plain list of strings
            # boxed into an object array only so it round-trips through
            # `RigTransform.load(allow_pickle=True)` -- it is not ragged like
            # `source_channels`, so re-cast it to a fixed-width string array
            # here rather than carrying the object dtype (and the trap that
            # comes with it) into the training-facing file for no reason.
            if stage == "rest_relative" and key == "names" and array.dtype == object:
                array = array.astype("<U")
            payload[f"prepare/{stage}/{key}"] = array

    out = config.out / "rigs" / rig
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "skeleton.npz", **payload)
    return mismatch


def build_stats(config: BuildConfig, rig: str) -> None:
    """The one schema-dependent artefact."""
    manifest = config.manifest(rig)
    arrays, spec = [], None
    for path in config.corpus.clips(config.root, rig):
        anim = BVH.read(path).to_animation().as_rigid_body(joint_translation="drop")
        # The rigid-body form, and a different clip on every iteration --
        # unlike `build_skeleton`'s single rest clip. Removal is selected on
        # `norm(offsets)`, `parents` and `names`, none of which either
        # difference can change, so every one of these reductions equals the
        # one `skeleton.npz` records. See the full argument there.
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


def _split_of(label_path: Path, warnings: list[str] | None) -> str:
    """One clip's authored split, or `"train"` -- never an exception.

    `split` reaches `index.jsonl` verbatim and the read path filters on it by
    string equality, so a non-string (`split:` with no value parses as `None`;
    `split: 42` as an int) would write a row no split can ever select. Both
    that and an unparseable file degrade to the default and a warning.
    """
    try:
        split = read_labels(label_path).get("split", "train")
    except Exception as error:  # noqa: BLE001 - a bad label file is not a bad corpus
        _warn(warnings, f"{label_path}: {type(error).__name__}: {error}; using split=train")
        return "train"
    if not isinstance(split, str):
        _warn(warnings, f"{label_path}: split must be a string, got {split!r}; using split=train")
        return "train"
    return split


def _warn(warnings: list[str] | None, message: str) -> None:
    if warnings is None:
        print(f"warning: {message}", flush=True)
    else:
        warnings.append(message)


def build_index(config: BuildConfig, warnings: list[str] | None = None) -> CorpusIndex:
    """One flat row per clip, joining rig-level manifest fields.

    `tags` lives once, in the manifest, and is joined here -- rather than copied
    into every ClipRecord, which wrote `[quadruped, mammal]` twenty-two times
    for BrownBear.

    `split` comes from the clip's OWN label file (`build_clips` writes one
    `<action>.yaml` per clip, and `--relabel` exists specifically so an
    authored `split:` survives a rebuild) -- not hardcoded, so editing a label
    file's `split:` actually reaches the index. Falls back to `"train"` when
    the label file has no `split:` key, matching what `build_clips` writes by
    default.

    That read is guarded PER CLIP, and deliberately: one hand-edited label file
    with a stray tab in it must not cost the whole index. `build_all` wraps each
    rig for the same reason, but `build_index` runs once, after that loop -- an
    exception escaping here discards a completed 73-rig build's index entirely.
    A bad label file yields the default split and a warning in `warnings`.
    """
    index = CorpusIndex()
    for rig in config.corpus.rigs(config.root):
        manifest = config.manifest(rig)
        for path in sorted((config.out / "clips" / rig).glob("*.npz")):
            with np.load(path) as data:
                n_frames = int(data["rotations"].shape[0])
                fps = float(data["fps"])
            split = _split_of(path.with_suffix(".yaml"), warnings)
            index.add(
                ClipRecord(
                    clip_id=clip_id(rig, path.stem),
                    skeleton=rig,
                    action=path.stem,
                    split=split,
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
    per-rig guard, leaving a partial corpus behind. `build_index` gets the
    same guard for the same reason -- one malformed manifest must not discard
    a completed multi-rig build.

    Convention (recorded here because the four passes do not agree on error
    shape, and that disagreement must not spread further): `build_clips`
    raises and aborts its rig; `build_skeleton` returns a warning string for a
    non-fatal condition instead of raising; `build_stats` raises `ValueError`
    on an empty rig. All of that is caught here, per rig, and sorted into
    `result.warnings` (non-fatal, every artefact for that rig still written)
    or `result.failures` (that rig produced no artefact for the step that
    raised). `build_index` is not per-rig, so its own failure is caught once,
    after the loop, and also lands in `result.failures`.
    """
    result = BuildResult()
    for rig in rigs or config.corpus.rigs(config.root):
        try:
            if not stats_only:
                result.clips += build_clips(config, rig)
                mismatch = build_skeleton(config, rig)
                if mismatch is not None:
                    result.warnings.append(mismatch)
            build_stats(config, rig)
            result.rigs += 1
        except Exception as error:  # noqa: BLE001 - collect, don't abort the corpus
            result.failures.append(f"{rig}: {type(error).__name__}: {error}")

    if not stats_only:
        try:
            build_index(config, result.warnings)
        except Exception as error:  # noqa: BLE001 - collect, don't discard a completed build
            result.failures.append(f"index: {type(error).__name__}: {error}")
    return result
