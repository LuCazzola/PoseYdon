"""Raw BVH -> canonical, invertible clips under the entity-first layout (stage 1).

For every rig with a manifest under the dataset's ``paths.rigs``, reads every
raw clip under ``paths.raw/source/<Rig>/``, runs it through the dataset's
``prepare:`` chain (`poseydon.build.prepare.chain_from`) and writes the result
to ``paths.root/clips/<Rig>/<action>.bvh``. The fitted transform, and the chain
itself, go to ``rigs/<Rig>/preproc.npz`` so the corpus can be turned back into
what it came from. The dataset is `configs/build.yaml`'s ``data`` group:

    docker compose run --rm test python scripts/process_dataset_truebones.py [--rigs Lion ...] [data=truebones_bone]
"""

from __future__ import annotations

import argparse
import dataclasses
import warnings as warnings_module
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from poseydon.build.corpus import rest_action
from poseydon.build.pipeline import compose_build
from poseydon.build.prepare import (
    CLIP,
    BoneRelative,
    EnforceRigid,
    PrepareChain,
    RigEvidence,
    RigTransform,
    chain_from,
)
from poseydon.build.report import PreparationReport
from poseydon.build.sources import SOURCES
from poseydon.core.rotations import QUAT_IDENTITY, euler_to_quat, quat_apply, quat_inverse
from poseydon.core.skeleton import SkeletonManifest, resolve
from poseydon.ingest.pipeline import available_rigs
from poseydon.io.bvh import BVH

TARGET_AXIS = "+Z"
_TARGET_FORWARD = np.array([0.0, 0.0, 1.0])


def chain_for(prepare, source_channels: tuple[tuple[str, ...], ...] | None) -> PrepareChain:
    """The dataset's chain, with THIS rig's real channel layout given to
    `EnforceRigid`, whose `fit` records it verbatim rather than inferring a
    layout from the (often single-frame, under-declaring) rest pose."""
    chain = chain_from(prepare)
    stages = tuple(
        dataclasses.replace(stage, source_channels=source_channels) if isinstance(stage, EnforceRigid) else stage
        for stage in chain.stages
    )
    return PrepareChain(stages, chain.entries)


def process_species(manifest: SkeletonManifest, data: DictConfig) -> tuple[int, list[str]]:
    """Prepare every raw clip for one rig. Returns (n_written, warnings)."""
    warnings: list[str] = []
    out_root = Path(data.paths.root)
    source = SOURCES.build(data.source)
    try:
        rest = source.rest(data, manifest)
        rest_channels = source.rest_channels(data, manifest)
        actions = source.actions(data, manifest)
    except ValueError as error:
        return 0, [f"{manifest.name}: {error}"]

    chain = chain_for(data.prepare, rest_channels)
    # A stage that decides from the whole rig (`prune_unskinned`) reads every clip
    # and the skin; those clips are then written from the same load.
    loaded: dict[str, object] = {}
    evidence = RigEvidence()
    report = PreparationReport.start("truebones", manifest.name, manifest.authority)
    try:
        if any(stage.uses_evidence for stage in chain.stages):
            loaded = {c.action: c for c in source.clips(data, manifest)}
            evidence = RigEvidence(tuple(c.anim for c in loaded.values()), source.skin(data, manifest))
        with warnings_module.catch_warnings(record=True) as caught:   # e.g. `ground: mesh` falling back
            warnings_module.simplefilter("always")
            rig_params = chain.fit_rig(rest, resolve(manifest, rest.names), audit=report, evidence=evidence)
    except ValueError as error:
        return 0, [f"{manifest.name}: {error}"]
    warnings.extend(str(w.message) for w in caught)

    clips_dir = out_root / "clips" / manifest.name
    clips_dir.mkdir(parents=True, exist_ok=True)

    clip_params: dict[str, dict] = {}
    n_written = 0
    for action in actions:
        # One bad clip must not cost a whole species. Some Truebones species
        # ship clips rigged differently from their own T-pose (Elephant's
        # __Take_001 has 43 joints against the T-pose's 51); those are reported
        # and skipped rather than aborting the other twenty.
        try:
            clip = loaded.get(action) or source.clip(data, manifest, action)
            resolved = resolve(manifest, clip.anim.names)
            prepared, params = chain.apply(clip.anim, resolved, rig_params, audit=report)

            BVH.from_animation(prepared).write(clips_dir / f"{action}.bvh")
            clip_params[action] = {
                stage.name: params[stage.name] for stage in chain.stages
                if stage.scope == CLIP
            }
        except Exception as error:  # noqa: BLE001 - collect, don't abort the corpus
            warnings.append(
                f"{manifest.name}/{action}: {type(error).__name__}: {error}"
            )
            continue
        n_written += 1

    report.write(out_root / "rigs" / manifest.name / "report.json")
    RigTransform(rig_params=rig_params, clip_params=clip_params, chain=chain.entries).save(
        out_root / "rigs" / manifest.name / "preproc.npz"
    )
    warnings.extend(_sanity_check(manifest, clips_dir, rig_params.get("bone_relative")))
    return n_written, warnings


def _sanity_check(
    manifest: SkeletonManifest, clips_dir: Path,
    bone_relative: dict[str, np.ndarray] | None,
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
    # A rig without `rest_pose:` (`fbx_cache`) has no rest clip on disk: any
    # clip is read, and the identity check below is skipped.
    has_rest_clip = manifest.rest_pose is not None
    subject = clips_dir / f"{rest_action(manifest)}.bvh" if has_rest_clip else written[0]
    if has_rest_clip and not subject.is_file():
        # Falling back silently would measure facing/identity on an
        # arbitrary clip and report plausible-looking numbers under a slug
        # drift -- name both files so the substitution is visible.
        warnings.append(
            f"{manifest.name}: expected rest clip {subject.name} is missing "
            f"from {clips_dir}; sanity-checking {written[0].name} instead"
        )
        subject = written[0]

    # Checked in the rest-relative view, where identity is the rest pose: a
    # bone-relative corpus is undone first (the identity on any other).
    anim = BVH.read(subject).to_animation()
    if bone_relative is not None:
        anim = BoneRelative().invert(anim, bone_relative)
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
        # `extra_yaw_deg` turns the pairs' forward into the rig's (a snake's pair
        # runs along its head); undo it so the check reads the rig's own facing.
        yaw = euler_to_quat(np.array([0.0, manifest.extra_yaw_deg, 0.0]), "ZYX")
        forward = quat_apply(quat_inverse(yaw), forward)
        return float(np.dot(forward / np.linalg.norm(forward), _TARGET_FORWARD))

    posed_dot = facing_dot(anim.global_positions()[0], resolved.facing_indices)
    def rest_skeleton_positions(anim) -> np.ndarray:
        """``(J, 3)`` joint positions of the REST skeleton: OFFSETs, no rotations."""
        positions = np.zeros((anim.n_joints, 3))
        for joint in range(1, anim.n_joints):
            positions[joint] = positions[anim.parents[joint]] + anim.offsets[joint]
        return positions

    rest = rest_skeleton_positions(anim)
    rest_dot = facing_dot(rest, resolved.facing_indices)
    # Upright, or a wrong or missing `up_axis:` in the manifest. On its side,
    # the right-to-left direction across hips and shoulders turns vertical;
    # upside down, the feet sit clearly above the hips and shoulders.
    across = sum(rest[r] - rest[l] for r, l in resolved.facing_indices)
    sideways = abs(across[1]) / np.linalg.norm(across)
    extent = np.ptp(rest, axis=0).max()
    body = rest[[j for pair in resolved.facing_indices for j in pair], 1].mean()
    feet = rest[list(resolved.foot_indices), 1].mean() if resolved.foot_indices else body
    if sideways > 0.5 or feet - body > 0.1 * extent:
        warnings.append(
            f"{manifest.name}: rest skeleton is not upright (right-left axis "
            f"{sideways:.2f} vertical, feet {(feet - body) / extent:+.2f} of extent "
            "above hips/shoulders); check `up_axis` and the facing pairs in the manifest"
        )
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

    # Every joint but the root (which carries `FaceAxis`'s turn) reads as
    # identity at the rest frame.
    if has_rest_clip:
        identity_err = float(
            np.abs(np.abs((anim.rotations[0, 1:] * QUAT_IDENTITY).sum(axis=-1)) - 1.0).max()
        )
        print(f"{label} T-pose frame max |1 - dot(rot, identity)| = {identity_err:.4f}")
        if identity_err > 1e-3:
            warnings.append(
                f"{manifest.name}: T-pose frame does not read as identity after bind "
                f"removal (max deviation {identity_err:.4f})"
            )

    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rigs", nargs="*", default=None, help="rig names to process (default: every rig under paths.rigs)")
    parser.add_argument("--config-name", default="build")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. data=truebones_bone")
    args = parser.parse_args()
    data = compose_build(args.overrides, config_name=args.config_name).data

    rig_names = available_rigs(data.paths.rigs)
    if args.rigs is not None:
        wanted = set(args.rigs)
        rig_names = [name for name in rig_names if name in wanted]

    total_written = 0
    all_warnings: list[str] = []
    for rig_name in rig_names:
        manifest = SkeletonManifest.load(Path(data.paths.rigs) / rig_name / "manifest.yaml")
        print(f"{manifest.name}:")
        n_written, warnings = process_species(manifest, data)
        print(f"  wrote {n_written} clips -> {Path(data.paths.root) / 'clips' / manifest.name}")
        total_written += n_written
        all_warnings.extend(warnings)

    print(f"\ntotal: {total_written} clips written across {len(rig_names)} rigs")
    if all_warnings:
        print(f"\n{len(all_warnings)} warning(s):")
        for warning in all_warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
