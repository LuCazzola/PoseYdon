"""Raw Truebones BVH -> T-pose-relative, +Z-facing clips, native hierarchy kept.

For every skeleton with a manifest in ``--manifest-dir`` (the hand-authored
subset under ``data/truebones/skeletons/``), reads every raw ``.bvh`` clip
under ``--raw-root/<Skeleton>/``, removes that rig's baked-in bind rotation
(:func:`poseydon.preproc.rest_pose.make_anim_rest_relative` against a T-pose-derived
rest reference), rotates the whole rig so its facing direction points +Z
(:func:`poseydon.ingest.align.rotate_to_face_axis`, using the manifest's
``facing`` joint pairs), and writes the result back out as an ordinary BVH under
``--out-dir/<Skeleton>/`` (default ``data/truebones/bvh``), same filename,
joint count, names and parents as the source.

No joint drop/merge/retarget, scaling, centring or grounding happens here --
those are separate concerns from bind-pose removal and facing alignment.
Y+-up needs no transform: BVH and ``Anim`` are already Y-up throughout this
codebase.

Run inside the test container:
    docker compose run --rm test python scripts/process_dataset_truebones.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from poseydon.core.animation import Animation
from poseydon.core.rotations import QUAT_IDENTITY
from poseydon.core.skeleton import SkeletonManifest, resolve
from poseydon.ingest.align import rotate_to_face_axis
from poseydon.io.bvh import BVH
from poseydon.build.prepare import RestRelative


def load_raw_clip(path: Path) -> Animation:
    """One raw Biped clip, with its per-joint translation kept.

    These rigs animate joint translation as well as rotation, and it is real
    motion: measured against the same clips read through Blender, dropping it
    moves joints by ~10% of skeleton size, while keeping it agrees to 0.2%.
    So the corpus keeps it and stays faithful to the source; the rigid-bone
    assumption is applied where it is actually required, at ingest, by an
    explicit ``as_rigid_body(joint_translation="drop")``.
    """
    return BVH.read(path).to_animation()

DEFAULT_RAW_ROOT = Path("data/truebones/Truebone_Z-OO")
DEFAULT_MANIFEST_DIR = Path("data/truebones/skeletons")
DEFAULT_OUT_DIR = Path("data/truebones/bvh")
TARGET_AXIS = "+Z"
_TARGET_FORWARD = np.array([0.0, 0.0, 1.0])




def find_tpose(clip_paths: list[Path]) -> Path | None:
    """The species' rest-pose file, by the reference's ``find_tpos_path`` rule.

    Matches ``"tpos"`` rather than ``"tpose"`` (Truebones spells it ``Tpose``,
    ``TPOSE`` and ``Tpos``), then falls back to an ``idle`` clip, whose first
    frame is a far better rest approximation than an arbitrary one.

    This must stay in step with the same rule in
    ``tools/reference/generate_truebones_manifests.py``: that tool reads a
    species' facing joints out of whichever file it calls the T-pose, so a
    manifest built from one file and applied against another would resolve
    joint names that were never checked together.
    """
    for path in clip_paths:
        if "tpos" in path.name.lower():
            return path
    for path in clip_paths:
        if path.name.lower().lstrip("_").startswith("idle"):
            return path
    return None


def resolve_rest_anim(
    clip_paths: list[Path],
) -> tuple[Animation, Path, bool]:
    """The single-frame rest reference to remove every clip's bind rotation against.

    Prefers the raw T-pose file, recovered exactly via ``establish_rest_pose``.
    Falls back to frame 0 of the first clip when the species has no T-pose
    file -- an approximation (that frame need not be a true rest pose),
    flagged by the returned bool.
    """
    tpose_path = find_tpose(clip_paths)
    if tpose_path is not None:
        return establish_rest_pose(tpose_path), tpose_path, False

    fallback_path = clip_paths[0]
    return load_raw_clip(fallback_path).slice(0, 1), fallback_path, True


def process_species(
    manifest: SkeletonManifest, raw_root: Path, out_dir: Path
) -> tuple[int, list[str]]:
    """Process every raw clip for one skeleton. Returns (n_written, warnings)."""
    warnings: list[str] = []
    species_dir = raw_root / manifest.name
    if not species_dir.is_dir():
        return 0, [f"{manifest.name}: no raw directory at {species_dir}, skipped"]

    clip_paths = sorted(species_dir.glob("*.bvh"))
    if not clip_paths:
        return 0, [f"{manifest.name}: no .bvh files found in {species_dir}, skipped"]

    rest_anim, rest_source, is_fallback = resolve_rest_anim(clip_paths)
    if is_fallback:
        warnings.append(
            f"{manifest.name}: no T-pose file found; using frame 0 of "
            f"{rest_source.name} as an approximate rest pose"
        )

    dest_dir = out_dir / manifest.name
    dest_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for clip_path in clip_paths:
        # One bad clip must not cost a whole species. Some Truebones species
        # ship clips rigged differently from their own T-pose (Elephant's
        # __Take_001 has 43 joints against the T-pose's 51), so the manifest
        # cannot name joints that exist in both; those clips are reported and
        # skipped rather than aborting the other twenty.
        try:
            anim = load_raw_clip(clip_path)
            # Catch a stray frame rate here, where the file it came from is
            # still named, rather than letting it reach ingest. Relative
            # tolerance because a BVH stores frame TIME, so 30 fps is a
            # rounded repeating decimal on disk (0.033333 -> 30.00003).
            if manifest.fps is not None and not math.isclose(
                manifest.fps, anim.fps, rel_tol=1e-3
            ):
                raise ValueError(
                    f"manifest requires {manifest.fps} fps but this clip is {anim.fps:.4f}"
                )
            anim = make_anim_rest_relative(anim, rest_anim)
            resolved = resolve(manifest, anim.names)
            anim = rotate_to_face_axis(
                anim, resolved.facing_indices, TARGET_AXIS, manifest.extra_yaw_deg
            )
            BVH.from_animation(anim).write(dest_dir / clip_path.name)
        except Exception as error:  # noqa: BLE001 - collect, don't abort the corpus
            warnings.append(
                f"{manifest.name}/{clip_path.name}: {type(error).__name__}: {error}"
            )
            continue
        n_written += 1

    warnings.extend(_sanity_check(manifest, dest_dir, rest_source, is_fallback))
    return n_written, warnings


def rest_skeleton_positions(anim: Animation) -> np.ndarray:
    """``(J, 3)`` joint positions of the REST skeleton: OFFSETs, no rotations.

    This is what a DCC tool draws for the bind/rest pose -- Blender's Edit
    Mode -- and it reads the ``OFFSET`` block ONLY. Checking it is not the
    same as checking ``Anim.global_positions()``, which applies the motion
    channels on top; a rig whose rest geometry and motion disagree passes
    the second check and fails this one.
    """
    positions = np.zeros((anim.n_joints, 3))
    for joint in range(1, anim.n_joints):
        positions[joint] = positions[anim.parents[joint]] + anim.offsets[joint]
    return positions


def _sanity_check(
    manifest: SkeletonManifest,
    dest_dir: Path,
    rest_source: Path,
    is_fallback: bool,
) -> list[str]:
    """Print diagnostics for the written files; warn (not fail) if one looks off.

    Reads the BVHs back off disk rather than re-deriving them in memory, so
    anything the write/read round trip itself breaks shows up here too.
    """
    warnings: list[str] = []
    written = sorted(dest_dir.glob("*.bvh"))
    if not written:
        return warnings

    subject = dest_dir / rest_source.name if not is_fallback else written[0]
    anim = BVH.read(subject).to_animation()
    resolved = resolve(manifest, anim.names)
    label = f"  [{manifest.name}] {subject.name}"

    # Motion and rest geometry must BOTH face +Z; checking only the first
    # leaves a rig whose Edit-Mode skeleton still points the old way.
    def facing_dot(positions: np.ndarray, facing_indices) -> float:
        """How well ``positions``' forward axis agrees with +Z. 1.0 is exact."""
        across = np.zeros(3)
        for right, left in facing_indices:
            across += positions[right] - positions[left]
        across = across / np.linalg.norm(across)
        forward = np.cross(np.array([0.0, 1.0, 0.0]), across)
        return float(np.dot(forward / np.linalg.norm(forward), _TARGET_FORWARD))

    posed_dot = facing_dot(anim.global_positions()[0], resolved.facing_indices)
    rest_dot = facing_dot(rest_skeleton_positions(anim), resolved.facing_indices)
    print(f"{label} frame-0 forward . +Z = {posed_dot:.4f}   rest-skeleton forward . +Z = {rest_dot:.4f}")
    for what, value in (("frame-0 pose", posed_dot), ("rest skeleton", rest_dot)):
        if value < 0.99:
            warnings.append(
                f"{manifest.name}: {what} is not aligned to +Z ({value:.4f}) in {subject.name}"
            )

    if is_fallback:
        return warnings

    # With the whole rig rotated (rotate_to_face_axis), EVERY joint --
    # the root included -- reads as identity at the rest frame.
    identity_err = float(np.abs(np.abs((anim.rotations[0] * QUAT_IDENTITY).sum(axis=-1)) - 1.0).max())
    print(f"{label} T-pose frame max |1 - dot(rot, identity)| = {identity_err:.4f}")
    if identity_err > 1e-3:
        warnings.append(
            f"{manifest.name}: T-pose frame does not read as identity after bind removal "
            f"(max deviation {identity_err:.4f})"
        )

    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--skeletons",
        nargs="*",
        default=None,
        help="Skeleton names to process (default: every manifest in --manifest-dir)",
    )
    args = parser.parse_args()

    # `_`-prefixed files are shared fragments pulled in via `base:`, not
    # skeletons -- the same convention `ingest.pipeline.available_skeletons` uses.
    manifest_paths = sorted(
        p for p in args.manifest_dir.glob("*.yaml") if not p.stem.startswith("_")
    )
    if args.skeletons is not None:
        wanted = set(args.skeletons)
        manifest_paths = [p for p in manifest_paths if p.stem in wanted]

    total_written = 0
    all_warnings: list[str] = []
    for manifest_path in manifest_paths:
        manifest = SkeletonManifest.load(manifest_path)
        print(f"{manifest.name}:")
        n_written, warnings = process_species(manifest, args.raw_root, args.out_dir)
        print(f"  wrote {n_written} clips -> {args.out_dir / manifest.name}")
        total_written += n_written
        all_warnings.extend(warnings)

    print(f"\ntotal: {total_written} clips written across {len(manifest_paths)} skeletons")
    if all_warnings:
        print(f"\n{len(all_warnings)} warning(s):")
        for warning in all_warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
