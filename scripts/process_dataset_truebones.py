"""Raw Truebones BVH -> canonical, invertible clips under the entity-first layout.

For every rig with a manifest under ``--out-root/rigs/<Rig>/manifest.yaml``,
reads every raw ``.bvh`` clip under ``--raw-root/<Rig>/``, runs it through
:class:`poseydon.build.prepare.PrepareChain` (bind-rotation removal, facing
alignment, rigid-bone enforcement, XZ centring, uniform scaling, grounding),
and writes the result to ``--out-root/clips/<Rig>/<action>.bvh``. The fitted
transform is recorded to ``--out-root/rigs/<Rig>/prepare.npz`` so the corpus
can be turned back into what it came from.

Run inside the test container:
    docker compose run --rm test python scripts/process_dataset_truebones.py
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np

from poseydon.build.prepare import (
    CLIP,
    CentreXZ,
    EnforceRigid,
    FaceAxis,
    PrepareChain,
    PutOnGround,
    RestRelative,
    RigTransform,
    ScaleToMeanBoneLength,
)
from poseydon.core.rotations import QUAT_IDENTITY
from poseydon.core.skeleton import SkeletonManifest, resolve
from poseydon.ingest.index import action_slug, strip_skeleton_prefix
from poseydon.ingest.pipeline import available_rigs
from poseydon.io.bvh import BVH

DEFAULT_RAW_ROOT = Path("data/truebones/source")
DEFAULT_OUT_ROOT = Path("data/truebones")
TARGET_AXIS = "+Z"
_TARGET_FORWARD = np.array([0.0, 0.0, 1.0])

def _chain_for(source_channels: tuple[tuple[str, ...], ...]) -> PrepareChain:
    """Build the chain with `EnforceRigid` recording THIS rig's real channels.

    The layout differs per rig, so it cannot live on a module-level constant;
    `EnforceRigid.fit` records `source_channels` verbatim when given one rather
    than inferring a layout from the (often single-frame, under-declaring)
    rest pose -- see `EnforceRigid`'s docstring.
    """
    return PrepareChain(
        (
            RestRelative(),
            FaceAxis(axis=TARGET_AXIS),
            EnforceRigid(joint_translation="drop", source_channels=source_channels),
            CentreXZ(),
            ScaleToMeanBoneLength(),
            PutOnGround(),
        )
    )


def _pick_by_name(clip_paths: list[Path]) -> Path | None:
    """The reference's ``find_tpos_path`` naming rule, applied within a set.

    Matches ``"tpos"`` rather than ``"tpose"`` (Truebones spells it ``Tpose``,
    ``TPOSE`` and ``Tpos``), then falls back to an ``idle`` clip, whose first
    frame is a far better rest approximation than an arbitrary one, then the
    first file in the set.
    """
    for path in clip_paths:
        if "tpos" in path.name.lower():
            return path
    for path in clip_paths:
        if path.name.lower().lstrip("_").startswith("idle"):
            return path
    return clip_paths[0] if clip_paths else None


def find_tpose(clip_paths: list[Path]) -> tuple[Path | None, list[str]]:
    """The species' rest-pose file, chosen from the MODAL joint set.

    The naming rule alone (``_pick_by_name`` over every clip, unfiltered)
    just takes whichever file happens to be named ``tpos``/``idle`` first,
    with no regard for whether that file's SKELETON agrees with the rest of
    the corpus. For four rigs (Ant, Crab, Deer, Jaguar) the named T-pose file
    is a minority outlier -- it disagrees with almost every clip, and every
    disagreeing clip is then rejected by ``RestRelative``. The file's NAME is
    not evidence of anything; its joint set, compared against the corpus, is.

    So: compute every clip's joint-name tuple (via ``BVH.read_names``, which
    parses only the HIERARCHY block -- reading every clip in full just for
    this would double the parse cost of the whole build), take the most
    common one, and apply the naming rule only among the files that carry it.
    A rig whose OWN majority disagrees with its T-pose file (a genuine corpus
    oddity, distinct from the four above) gets a warning naming both.

    Returns ``(chosen_path, warnings)``; ``chosen_path`` is ``None`` only when
    ``clip_paths`` is empty.
    """
    warnings: list[str] = []
    if not clip_paths:
        return None, warnings

    names_by_path = {path: BVH.read_names(path) for path in clip_paths}
    counts = Counter(names_by_path.values())
    modal_names, _ = counts.most_common(1)[0]
    modal_paths = [path for path in clip_paths if names_by_path[path] == modal_names]

    old_pick = _pick_by_name(clip_paths)
    chosen = _pick_by_name(modal_paths)

    if old_pick is not None and chosen is not None and old_pick != chosen:
        warnings.append(
            f"rest-pose file chosen by name ({old_pick.name}) is not in the modal "
            f"joint set ({len(modal_paths)}/{len(clip_paths)} clips); using "
            f"{chosen.name} instead"
        )

    return chosen, warnings


def process_species(
    manifest: SkeletonManifest, raw_root: Path, out_root: Path
) -> tuple[int, list[str]]:
    """Prepare every raw clip for one rig. Returns (n_written, warnings)."""
    warnings: list[str] = []
    species_dir = raw_root / manifest.name
    if not species_dir.is_dir():
        return 0, [f"{manifest.name}: no raw directory at {species_dir}, skipped"]

    clip_paths = sorted(species_dir.glob("*.bvh"))
    if not clip_paths:
        return 0, [f"{manifest.name}: no .bvh files found in {species_dir}, skipped"]

    rest_path, tpose_warnings = find_tpose(clip_paths)
    warnings.extend(f"{manifest.name}: {message}" for message in tpose_warnings)
    if rest_path is None:
        rest_path = clip_paths[0]
        warnings.append(
            f"{manifest.name}: no T-pose file found; using {rest_path.name} as an "
            "approximate rest pose"
        )

    rest_bvh = BVH.read(rest_path)
    rest = rest_bvh.to_animation()
    chain = _chain_for(rest_bvh.channels)
    rig_params = chain.fit_rig(rest, resolve(manifest, rest.names))

    clips_dir = out_root / "clips" / manifest.name
    clips_dir.mkdir(parents=True, exist_ok=True)

    clip_params: dict[str, dict] = {}
    n_written = 0
    for clip_path in clip_paths:
        # One bad clip must not cost a whole species. Some Truebones species
        # ship clips rigged differently from their own T-pose (Elephant's
        # __Take_001 has 43 joints against the T-pose's 51); those are reported
        # and skipped rather than aborting the other twenty.
        try:
            source = BVH.read(clip_path).to_animation()
            if manifest.fps is not None and not math.isclose(
                manifest.fps, source.fps, rel_tol=1e-3
            ):
                raise ValueError(
                    f"manifest requires {manifest.fps} fps but this clip is "
                    f"{source.fps:.4f}"
                )
            resolved = resolve(manifest, source.names)
            prepared, params = chain.apply(source, resolved, rig_params)

            action = strip_skeleton_prefix(action_slug(clip_path.stem), manifest.name)
            BVH.from_animation(prepared).write(clips_dir / f"{action}.bvh")
            clip_params[action] = {
                stage.name: params[stage.name] for stage in chain.stages
                if stage.scope == CLIP
            }
        except Exception as error:  # noqa: BLE001 - collect, don't abort the corpus
            warnings.append(
                f"{manifest.name}/{clip_path.name}: {type(error).__name__}: {error}"
            )
            continue
        n_written += 1

    RigTransform(rig_params=rig_params, clip_params=clip_params).save(
        out_root / "rigs" / manifest.name / "prepare.npz"
    )
    warnings.extend(_sanity_check(manifest, clips_dir, rest_path))
    return n_written, warnings


def _sanity_check(
    manifest: SkeletonManifest, clips_dir: Path, rest_source: Path
) -> list[str]:
    """Print diagnostics for the written files; warn (not fail) if one looks off.

    Reads the BVHs back off disk rather than re-deriving them in memory, so
    anything the write/read round trip itself breaks shows up here too.
    """
    warnings: list[str] = []
    written = sorted(clips_dir.glob("*.bvh"))
    if not written:
        return warnings

    # The identity check below only holds at the rest frame itself, so this
    # must read back the rest source's own written clip -- not just any
    # clip -- by the same action-slug it was written under.
    rest_action = strip_skeleton_prefix(action_slug(rest_source.stem), manifest.name)
    subject = clips_dir / f"{rest_action}.bvh"
    if not subject.is_file():
        subject = written[0]

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

    def rest_skeleton_positions(anim) -> np.ndarray:
        """``(J, 3)`` joint positions of the REST skeleton: OFFSETs, no rotations."""
        positions = np.zeros((anim.n_joints, 3))
        for joint in range(1, anim.n_joints):
            positions[joint] = positions[anim.parents[joint]] + anim.offsets[joint]
        return positions

    posed_dot = facing_dot(anim.global_positions()[0], resolved.facing_indices)
    rest_dot = facing_dot(rest_skeleton_positions(anim), resolved.facing_indices)
    print(
        f"{label} frame-0 forward . +Z = {posed_dot:.4f}   "
        f"rest-skeleton forward . +Z = {rest_dot:.4f}"
    )
    for what, value in (("frame-0 pose", posed_dot), ("rest skeleton", rest_dot)):
        if value < 0.99:
            warnings.append(
                f"{manifest.name}: {what} is not aligned to +Z ({value:.4f}) in "
                f"{subject.name}"
            )

    # With the whole rig rotated (FaceAxis), EVERY joint -- the root included --
    # reads as identity at the rest frame.
    identity_err = float(
        np.abs(np.abs((anim.rotations[0] * QUAT_IDENTITY).sum(axis=-1)) - 1.0).max()
    )
    print(f"{label} T-pose frame max |1 - dot(rot, identity)| = {identity_err:.4f}")
    if identity_err > 1e-3:
        warnings.append(
            f"{manifest.name}: T-pose frame does not read as identity after bind "
            f"removal (max deviation {identity_err:.4f})"
        )

    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--rigs",
        nargs="*",
        default=None,
        help="Rig names to process (default: every rig under --out-root/rigs)",
    )
    args = parser.parse_args()

    rig_names = available_rigs(args.out_root / "rigs")
    if args.rigs is not None:
        wanted = set(args.rigs)
        rig_names = [name for name in rig_names if name in wanted]

    total_written = 0
    all_warnings: list[str] = []
    for rig_name in rig_names:
        manifest = SkeletonManifest.load(args.out_root / "rigs" / rig_name / "manifest.yaml")
        print(f"{manifest.name}:")
        n_written, warnings = process_species(manifest, args.raw_root, args.out_root)
        print(f"  wrote {n_written} clips -> {args.out_root / 'clips' / manifest.name}")
        total_written += n_written
        all_warnings.extend(warnings)

    print(f"\ntotal: {total_written} clips written across {len(rig_names)} rigs")
    if all_warnings:
        print(f"\n{len(all_warnings)} warning(s):")
        for warning in all_warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
