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
    PromoteRoot,
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

def _chain_for(source_channels: tuple[tuple[str, ...], ...], rig: str = "") -> PrepareChain:
    """Build the chain with `EnforceRigid` recording THIS rig's real channels.

    The layout differs per rig, so it cannot live on a module-level constant;
    `EnforceRigid.fit` records `source_channels` verbatim when given one rather
    than inferring a layout from the (often single-frame, under-declaring)
    rest pose -- see `EnforceRigid`'s docstring.
    """
    return PrepareChain(
        (
            RestRelative(),
            # Second, and the position is contractual. After RestRelative,
            # whose _check requires the clip to carry exactly the joints its
            # rest pose declares -- a promoted clip would fail it. Before
            # EnforceRigid, so the translation channel EnforceRigid preserves
            # is the PROMOTED root's rather than the locator's. And before
            # ScaleToMeanBoneLength: the connectors this removes are long
            # (6.47 bone lengths on Pirrana, 4.95 on Bear) and currently enter
            # the mean, so promoting first changes the canonical scale for
            # these 14 rigs -- a correction, since a locator-to-body connector
            # is not an anatomical bone.
            PromoteRoot(target=PROMOTE_ROOT.get(rig)),
            FaceAxis(axis=TARGET_AXIS),
            EnforceRigid(joint_translation="drop", source_channels=source_channels),
            CentreXZ(),
            ScaleToMeanBoneLength(),
            PutOnGround(),
        )
    )


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


#: Joint promoted to root, for the 14 rigs that root at a ground locator.
#:
#: Measured on the rest pose as height fraction (rootY - minY) / (maxY - minY),
#: the corpus splits absolutely: these 14 at f <= 0.005, the other 59 at
#: f >= 0.189, nothing between. Thirteen entries are the first branching joint
#: along the root's single-child chain. Tukan is authored because its root
#: branches straight into the real skeleton (N_ALL -> locator) and a dead MESH
#: subtree of geometry-holder nodes, so no chain rule reaches it.
#:
#: The gate is the locator classification, NEVER the presence of an offset:
#: Lynx and BrownBear have a correct root on the pelvis and a chain continuing
#: to Bip01_Spine at 0.79 and 0.89 bone lengths, and an offset-keyed rule would
#: promote their root onto the spine. A rig absent from this table is untouched.
PROMOTE_ROOT: dict[str, str] = {
    "Bear": "NPC_Pelvis",
    "Camel": "Bip01",
    "Crow": "_00",
    "Dog": "Bip01_Pelvis",
    "Dog-2": "Bip01_Pelvis",
    "Horse": "Bip01_Pelvis",
    "Pirrana": "locator",
    "Pteranodon": "jt_Cog_C",
    "Raptor3": "jt_Cog_C",
    "SabreToothTiger": "Sabrecat__pelv_",
    "Scorpion-2": "jt_Cog_C",
    "Spider": "_body_",
    "Trex": "jt_Cog_C",
    "Tukan": "locator",
}


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

    try:
        rest_path = rest_source(manifest, clip_paths)
    except ValueError as error:
        return 0, [f"{manifest.name}: {error}"]

    rest_bvh = BVH.read(rest_path)
    rest = rest_bvh.to_animation()
    chain = _chain_for(rest_bvh.channels, manifest.name)
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
    manifest: SkeletonManifest, clips_dir: Path, rest_path: Path
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
    # clip -- by the same action-slug it was written under. `rest_action`
    # is the module's one source of truth for that slug; re-deriving it
    # inline here would be a second copy of the rule this plan exists to
    # de-duplicate.
    rest_clip_action = rest_action(manifest)
    subject = clips_dir / f"{rest_clip_action}.bvh"
    if not subject.is_file():
        # Falling back silently would measure facing/identity on an
        # arbitrary clip and report plausible-looking numbers under a slug
        # drift -- name both files so the substitution is visible.
        warnings.append(
            f"{manifest.name}: expected rest clip {subject.name} is missing "
            f"from {clips_dir}; sanity-checking {written[0].name} instead"
        )
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
