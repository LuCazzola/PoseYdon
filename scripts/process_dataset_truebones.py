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
import json
import shutil
import warnings as warnings_module
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from poseydon.build.clips import CLIP_OPS, MaxLength, apply_clip_ops
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
    tree_diameter,
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


def process_species(
    manifest: SkeletonManifest, data: DictConfig, dropped: dict[str, str] | None = None,
    prepared: dict[str, dict] | None = None,
) -> tuple[int, list[str], int]:
    """Prepare every raw clip for one rig. Returns (n_written, warnings, n_frames).

    The dataset's ``clips:`` ops (`poseydon.build.clips`) trim and filter around
    ``prepare:``. A clip they drop is written nowhere; it lands, with its reason,
    in the rig's ``report.json`` and in ``dropped`` (``Rig-action`` -> reason).
    ``prepared``, if given, receives the rig's prepared rest skeleton (joint names
    and tree diameter) for `--check-against`.
    """
    warnings: list[str] = []
    dropped = {} if dropped is None else dropped
    ops = CLIP_OPS.build_all(data.get("clips"))
    windowed = any(isinstance(op, MaxLength) and op.mode == "windows" for op in ops)
    out_root = Path(data.paths.root)
    source = SOURCES.build(data.source)
    try:
        rest = source.rest(data, manifest)
        rest_channels = source.rest_channels(data, manifest)
        actions = source.actions(data, manifest)
    except ValueError as error:
        return 0, [f"{manifest.name}: {error}"], 0

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
        return 0, [f"{manifest.name}: {error}"], 0
    warnings.extend(str(w.message) for w in caught)
    dropped.update(source.dropped)   # empty for `bvh_files`: `data=truebones` writes what it always did
    report.dropped.update(source.dropped)

    # A rig-level op judges the skeleton as the clips will have it; a rest pose
    # that is none of the source's clips (a bind pose, an authored T-pose) is
    # prepared to be written as one.
    rest_name = rest_action(manifest)
    own_rest = rest_name not in actions
    if ops or own_rest:
        try:
            prepared_rest, rest_params = chain.apply(rest, resolve(manifest, rest.names), rig_params)
        except Exception as error:  # noqa: BLE001 - one rig must not abort the corpus
            return 0, [f"{manifest.name}: {type(error).__name__}: {error}"], 0
        if prepared is not None:
            prepared[manifest.name] = {"joints": list(prepared_rest.names),
                                       "diameter": tree_diameter(prepared_rest.offsets, prepared_rest.parents)}
        for op in ops:
            if reason := op.rig(prepared_rest, manifest.name):
                dropped[manifest.name] = report.dropped[manifest.name] = reason
                shutil.rmtree(out_root / "clips" / manifest.name, ignore_errors=True)   # an earlier run's
                (out_root / "rigs" / manifest.name / "preproc.npz").unlink(missing_ok=True)
                report.write(out_root / "rigs" / manifest.name / "report.json")
                return 0, [f"{manifest.name}: skipped, {reason}"], 0

    # A leaf joint's rotation is lost in a plain BVH, so the files keep it
    # (`BVH.from_animation`, `leaf_rotations`): an FBX has no End Sites, so its
    # wing tips, toes and ear tips are leaves, as are the joints `prune_unskinned`
    # leaves childless. A BVH source's leaves are End Sites, which come back as they were.
    clips_dir = out_root / "clips" / manifest.name
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_params: dict[str, dict] = {}
    if own_rest:   # written as the one-frame clip `rest_action` names
        BVH.from_animation(prepared_rest, leaf_rotations=True).write(clips_dir / f"{rest_name}.bvh")
        clip_params[rest_name] = {   # stage 2 reads its facing like any clip's
            stage.name: rest_params[stage.name] for stage in chain.stages if stage.scope == CLIP
        } | {"frames": {"window": np.array((0, 1, 1))}}

    cropped: set[str] = set()   # clips a `clips:` op cut, whose lowest joint is no longer the grounded one
    n_written = n_frames = 0
    for action in actions:
        # One bad clip must not cost a whole species. Some Truebones species
        # ship clips rigged differently from their own T-pose (Elephant's
        # __Take_001 has 43 joints against the T-pose's 51); those are reported
        # and skipped rather than aborting the other twenty.
        clip_id = f"{manifest.name}-{action}"
        kept, reason = [], None
        try:
            clip = loaded.get(action) or source.clip(data, manifest, action)
            resolved = resolve(manifest, clip.anim.names)
            # Head trim and fps halving first, so `face_axis` fits on the trimmed
            # clip's frame 0; the other ops judge the prepared clip.
            trimmed = apply_clip_ops(ops, clip.anim, clip_id, "source")
            if isinstance(trimmed, str):
                reason = trimmed
            else:
                for sub, window in trimmed:
                    prepared, params = chain.apply(sub, resolved, rig_params, audit=report)
                    result = apply_clip_ops(ops, prepared, clip_id, "prepared", base=window)
                    if isinstance(result, str):
                        reason = result
                    else:
                        kept += [(piece, w, params) for piece, w in result]
        except Exception as error:  # noqa: BLE001 - collect, don't abort the corpus
            warnings.append(
                f"{manifest.name}/{action}: {type(error).__name__}: {error}"
            )
            continue
        if not kept:
            dropped[clip_id] = report.dropped[clip_id] = reason
            continue
        for i, (piece, window, params) in enumerate(kept):
            name = f"{action}_w{i:03d}" if windowed else action
            BVH.from_animation(piece, leaf_rotations=True).write(clips_dir / f"{name}.bvh")
            clip_params[name] = {
                stage.name: params[stage.name] for stage in chain.stages
                if stage.scope == CLIP
            }
            # Source-frame `range(start, stop, step)` this clip came from.
            clip_params[name]["frames"] = {"window": np.array(window)}
            if tuple(window) != (0, clip.anim.n_frames, 1):
                cropped.add(name)
            n_written += 1
            n_frames += piece.n_frames

    # What an earlier run wrote and this one did not (a clip now dropped or skipped,
    # `_wNNN` windows of another `max_length` mode, an action no longer listed) goes,
    # stage 2's files of the same clip with it: stage 2 reads every BVH here.
    for path in clips_dir.iterdir():
        if path.suffix in (".bvh", ".npz", ".yaml") and path.stem not in clip_params:
            path.unlink()

    report.write(out_root / "rigs" / manifest.name / "report.json")
    RigTransform(rig_params=rig_params, clip_params=clip_params, chain=chain.entries).save(
        out_root / "rigs" / manifest.name / "preproc.npz"
    )
    warnings.extend(_sanity_check(manifest, clips_dir, rig_params.get("bone_relative"), chain, cropped))
    return n_written, warnings, n_frames


def _sanity_check(
    manifest: SkeletonManifest, clips_dir: Path,
    bone_relative: dict[str, np.ndarray] | None, chain: PrepareChain, cropped: set[str] = frozenset(),
) -> list[str]:
    """Print diagnostics for the written files; warn (not fail) if one looks off.

    Reads the BVHs back off disk rather than re-deriving them in memory, so
    anything the write/read round trip itself breaks shows up here too. Checks
    that belong to a stage the chain lacks (the rest skeleton facing needs
    `face_rest_axis`) are skipped for it.
    """
    stages = {stage.name: stage for stage in chain.stages}
    warnings: list[str] = []
    written = sorted(clips_dir.glob("*.bvh"))
    if not written:
        return warnings

    # The identity check below only holds at the rest frame itself, so this
    # must read back the rest source's own written clip -- not just any
    # clip -- by the same action-slug it was written under. `rest_action`
    # is the module's one source of truth for that slug (a rig without
    # `rest_pose:` has its bind pose, `bind_pose.bvh`); re-deriving it
    # inline here would be a second copy of the rule this plan exists to
    # de-duplicate.
    subject = clips_dir / f"{rest_action(manifest)}.bvh"
    if not subject.is_file():
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

    # UniMate grounds before trimming, so a cropped clip may sit above y = 0.
    for clip in written if "ground_clip" in stages else ():
        if clip.stem in cropped or clip.stem == rest_action(manifest):
            continue
        low = float(BVH.read(clip).to_animation().global_positions()[..., 1].min())
        if abs(low) > 1e-4:
            warnings.append(f"{manifest.name}/{clip.stem}: lowest joint is at y = {low:.5f}, not 0")
    scale = stages.get("scale")
    if scale is not None and scale.mode == "diameter":
        diameter = tree_diameter(anim.offsets, anim.parents)
        if abs(diameter - scale.target) > 1e-3 * scale.target:
            warnings.append(f"{manifest.name}: rest diameter is {diameter:.4f}, not {scale.target}")
    # Upright, or a wrong or missing `up_axis:` in the manifest. On its side,
    # the right-to-left direction across hips and shoulders turns vertical;
    # upside down, the feet sit clearly above the hips and shoulders. Read on
    # the posed rest frame: without `rest_relative` (`data=unimate`) the rest
    # rotations carry the shape and the offsets alone are not the rest pose.
    pose = anim.global_positions()[0]
    across = sum(pose[r] - pose[l] for r, l in resolved.facing_indices)
    sideways = abs(across[1]) / np.linalg.norm(across)
    extent = np.ptp(pose, axis=0).max()
    body = pose[[j for pair in resolved.facing_indices for j in pair], 1].mean()
    feet = pose[list(resolved.foot_indices), 1].mean() if resolved.foot_indices else body
    if sideways > 0.5 or feet - body > 0.1 * extent:
        warnings.append(
            f"{manifest.name}: rest frame is not upright (right-left axis "
            f"{sideways:.2f} vertical, feet {(feet - body) / extent:+.2f} of extent "
            "above hips/shoulders); check `up_axis` and the facing pairs in the manifest"
        )
    # The rest skeleton faces +Z only where `face_rest_axis` turned it.
    checks = [("frame-0 pose", posed_dot)]
    if "face_rest_axis" in stages:
        checks.append(("rest skeleton", facing_dot(rest_skeleton_positions(anim), resolved.facing_indices)))
    print(f"{label} " + "   ".join(f"{what} forward . +Z = {value:.4f}" for what, value in checks))
    for what, value in checks:
        if value < 0.99:
            warnings.append(
                f"{manifest.name}: {what} is not aligned to +Z ({value:.4f}) in "
                f"{subject.name}"
            )

    # Every joint but the root (which carries `FaceAxis`'s turn) reads as
    # identity at the rest frame. Only `rest_relative` makes it so: a chain
    # without it (`data=unimate`, whose bind pose keeps its joints' own frames)
    # skips the check.
    if "rest_relative" in stages:
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


def _float32_end_frames(end: float) -> int:
    """Frames a Blender action ending at key-time ``end`` (frames) gets WITHOUT
    UniMate's ``FRAME_RANGE_EPS``: Blender holds the time as float32, seconds first
    (``f32(f32(ktime / 46186158000) * 30)``), and a plain ``ceil`` adds a frame to
    ``119.0000076``. Measured to reproduce ``clip_frames.json`` on all 1,097 clips."""
    ktime = round(end * 46186158000 / 30)
    return int(np.ceil(np.float32(np.float32(ktime / 46186158000) * np.float32(30)))) + 1


def check_against(reference: Path, data: DictConfig, prepared: dict[str, dict], dropped: dict[str, str],
                  n_written: int, n_frames: int) -> None:
    """Compare the build with UniMate's published Truebones sidecars; print four lines.

    Per-rig and per-clip mismatches, each with the stage or filter responsible,
    go to ``<paths.root>/replication.json``. Joint sets compare the prepared rest
    skeleton's names with ``joint_names.json`` (as sets); export frames compare
    the cached source clip's frame count (before any filter) with
    ``clip_frames.json``.
    """
    names = json.loads((reference / "joint_names.json").read_text())
    expected_frames = json.loads((reference / "clip_frames.json").read_text())
    joints = {}
    for rig, expected in names.items():
        got = prepared.get(rig, {}).get("joints")
        if got is None:
            joints[rig] = {"stage": "rig skipped before the rest skeleton was prepared"}
        elif set(got) != set(expected):
            joints[rig] = {"stage": "source rig cleanup (prune_unskinned / armature rule)",
                           "missing": sorted(set(expected) - set(got)), "extra": sorted(set(got) - set(expected))}
    cache = Path(data.source.cache)
    frames = {}
    for clip, expected in expected_frames.items():
        rig, action = clip.split("-", 1)
        with np.load(cache / rig / f"{action}.npz") as z:
            got, end = len(z["frames"]), float(z["key_range"][1])
        if got != expected:
            frames[clip] = {"stage": "frame range (source)", "ours": got, "unimate": expected,
                            "explained_by_float32_end_noise": _float32_end_frames(end) == expected}
    diameters = {rig: p["diameter"] for rig, p in prepared.items() if abs(p["diameter"] - 2.0) > 1e-9}
    dropped_by = {}
    skip = reference / "filtered_clips.txt"
    for line in skip.read_text().splitlines() if skip.is_file() else ():
        if line.split("#")[0].strip():
            dropped_by.setdefault("skip list", []).append(line.split()[0])
    for clip, reason in dropped.items():
        dropped_by.setdefault(reason.split(":")[0], []).append(clip)
    Path(data.paths.root, "replication.json").write_text(json.dumps(
        {"joint_sets": joints, "export_frames": frames, "diameter": diameters, "dropped_by_filter": dropped_by},
        indent=2, sort_keys=True) + "\n")
    print(f"joint sets: {len(names) - len(joints)}/{len(names)} match")
    explained = sum(f["explained_by_float32_end_noise"] for f in frames.values())
    print(f"export frames: {len(expected_frames) - len(frames)}/{len(expected_frames)} match, "
          f"{explained} of the {len(frames)} others explained (+1 frame, float32 end noise)")
    print(f"training clips: {n_written}/1023, frames: {n_frames}/95491")
    print(f"diameter: {len(prepared) - len(diameters)}/{len(names)} at 2.0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rigs", nargs="*", default=None, help="rig names to process (default: every rig under paths.rigs)")
    parser.add_argument("--check-against", type=Path, default=None, metavar="DIR",
                        help="UniMate's published sidecars (data=unimate): print the replication report, "
                             "write <paths.root>/replication.json")
    parser.add_argument("--config-name", default="build")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. data=truebones_bone")
    args = parser.parse_args()
    data = compose_build(args.overrides, config_name=args.config_name).data

    rig_names = available_rigs(data.paths.rigs)
    if args.rigs is not None:
        wanted = set(args.rigs)
        rig_names = [name for name in rig_names if name in wanted]

    total_written = total_frames = 0
    dropped: dict[str, str] = {}
    prepared: dict[str, dict] = {}
    all_warnings: list[str] = []
    for rig_name in rig_names:
        manifest = SkeletonManifest.load(Path(data.paths.rigs) / rig_name / "manifest.yaml")
        print(f"{manifest.name}:")
        n_written, warnings, n_frames = process_species(manifest, data, dropped, prepared)
        print(f"  wrote {n_written} clips -> {Path(data.paths.root) / 'clips' / manifest.name}")
        total_written += n_written
        total_frames += n_frames
        all_warnings.extend(warnings)

    print(f"\ntotal: {total_written} clips written across {len(rig_names)} rigs")
    if data.get("clips"):
        Path(data.paths.root, "dropped.json").write_text(json.dumps(dropped, indent=2, sort_keys=True) + "\n")
        print(f"training clips: {total_written}, frames: {total_frames}; dropped: {len(dropped)}")
    if args.check_against is not None:
        check_against(args.check_against, data, prepared, dropped, total_written, total_frames)
    if all_warnings:
        print(f"\n{len(all_warnings)} warning(s):")
        for warning in all_warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
