"""One verifier case: the stage files of a built round trip, and the frames to judge."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Case:
    """``files`` are the files under test by name; ``clip_fbx`` is the clip's own vendor
    file (joint motion) and ``skin_fbx`` the rig's reference (mesh, skin, bind). Frames
    are positions in the file under test and in ``clip_fbx``."""

    data_name: str
    rig: str
    action: str
    files: dict[str, Path]
    clip_fbx: Path
    skin_fbx: Path
    frames: np.ndarray
    source_frames: np.ndarray
    window: tuple[int, int, int] | None = None  # the clip's recorded (start, stop, step) over the vendor keys
    floor: bool = False  # the file is the source clip written with no preparation (`build --floor`)


@dataclass
class LossRecord:
    """What the pipeline recorded about a rig: the lossy stages a clip went through, in order, with
    the parameters their laws need, the asset the returned file is written onto and where the dataset's
    rest pose comes from (`rest_asset: true` in its source config: the asset; else the vendor file's bind)."""

    stages: list[str]
    params: dict[str, dict]
    asset: Path  # the FBX the returned file is written onto
    bind_of: Path | None  # the file whose bind is the dataset's rest pose; None where the asset's pose is it
    reduction: tuple[str, ...] = ()  # the stages that are the reduction's rules: the model view, not the prepared clip


def loss_record(case: Case) -> LossRecord:
    """Read the rig's preparation and reduction records (the only place the verifier reads them)."""
    from poseydon.build.pipeline import compose_build  # noqa: PLC0415 - the SDK-free tools stay importable without it
    from poseydon.export import RigRecord  # noqa: PLC0415
    from poseydon.features.reduce import load_still  # noqa: PLC0415

    data = compose_build([f"data={case.data_name}"]).data
    root = Path(data.paths.root)
    record = RigRecord.load(root, case.rig)
    stages = [s.name for s in record.chain.stages]
    params: dict[str, dict] = {s: {} for s in stages}
    promote = record.params.get("promote_root", {})
    if "promote_root" in params:
        params["promote_root"] = {"chain": [str(n) for n in promote["names"][:-1]],
                                  "dead": [str(n) for n in promote["dead_names"]]}
    prune = record.params.get("prune_unskinned")
    if prune is not None:
        params["prune_unskinned"] = {"removed": [str(n) for n in prune["removed"]],
                                     "kind": [str(k) for k in prune["kind"]]}
    with np.load(root / "rigs" / case.rig / "skeleton.npz", allow_pickle=True) as z:
        still = load_still(z)
    # The rules the rig recorded are its stages: one with no law reaches MissingLaw even if it removed nothing.
    for rule in record.reduction_rules:
        stages.append(rule.name)
        params[rule.name] = {"names": [], "parents": [], "rotations": []}
    for op in record.reduction.ops:
        rule = "fold_into_root" if op.kind == "collapse" else "drop_still" if op.name in still else "drop_end_sites"
        if rule in params:
            params[rule]["names"].append(op.name)
            params[rule]["parents"].append(op.parent)
            params[rule]["rotations"].append(list(op.rotation))
    if "fold_into_root" in params:
        params["fold_into_root"]["relative"] = "rest_relative" in stages
    if "drop_still" in params:  # up_axis, then face_rest_axis, conjugated every recorded local rotation
        turn = Rotation.identity()
        for stage in ("up_axis", "face_rest_axis"):
            if stage in record.params:
                turn = Rotation.from_quat(np.asarray(record.params[stage]["rotation"])) * turn
        params["drop_still"]["turn"] = turn.as_matrix().tolist()
    return LossRecord(stages, params, record.asset, None if data.source.get("rest_asset", False) else case.skin_fbx,
                      tuple(rule.name for rule in record.reduction_rules))


def recorded_space(case: Case, stage: str) -> np.ndarray:
    """The 4x4 similarity `RigRecord.placement` puts a stage file's scene in (``2_prepared`` or ``3_model``)."""
    from poseydon.build.pipeline import compose_build  # noqa: PLC0415
    from poseydon.core.animation import RigidBodyAnimation  # noqa: PLC0415
    from poseydon.export import RigRecord  # noqa: PLC0415
    from poseydon.features.reduce import apply_reduction  # noqa: PLC0415

    root = Path(compose_build([f"data={case.data_name}"]).data.paths.root)
    record = RigRecord.load(root, case.rig)
    prepared = RigidBodyAnimation.load(root / "clips" / case.rig / f"{case.action}.npz").as_rigid_body(joint_translation="drop")
    anim = prepared if stage == "2_prepared" else apply_reduction(prepared, record.reduction)
    return record.placement(anim, facing_from=case.action)[1]


@dataclass
class ZeroRecord:
    """What the pipeline declares about a rig's zero: the preparation stages, the rig's `rest.npz`, the
    prepared joints, and the manifest's definition of the rest pose's frame (``ground`` is the mode of
    the rig's `ground` stage, None where the chain has none)."""

    stages: tuple[str, ...]
    rest: Path
    joints: tuple[str, ...]
    facing: tuple[tuple[str, str], ...]  # (right, left) pairs
    extra_yaw_deg: float
    up_axis: str
    ground: str | None
    fixed: tuple[str, ...]  # joints a file holds by structure (`RigRecord.fixed`)
    space: np.ndarray  # the 4x4 similarity `RigRecord.placement` puts the zero in
    rest_pose: Path | None  # the manifest's authored or vendor rest-pose BVH


def zero_clip(record):
    """The zero pose: identity rotations on the rig's `skeleton.npz` offsets, the root where its rest-pose clip
    (`rest_action`) stands at frame 0."""
    from poseydon.core.animation import RigidBodyAnimation  # noqa: PLC0415
    from poseydon.core.rotations import QUAT_IDENTITY  # noqa: PLC0415

    with np.load(record.corpus / "rigs" / record.rig / "skeleton.npz", allow_pickle=True) as z:
        offsets, parents, names = z["offsets"], z["parents"].astype(np.int32), tuple(str(n) for n in z["names"])
        clip = record.corpus / "clips" / record.rig / f"{z['rest_action']}.npz"
    if not clip.is_file():
        raise FileNotFoundError(f"{record.rig}: the rest clip {clip} is missing, so the zero has no root")
    return RigidBodyAnimation.from_root_motion(
        rotations=np.tile(QUAT_IDENTITY, (1, len(names), 1)), root_pos=RigidBodyAnimation.load(clip).root_pos[:1].copy(),
        offsets=offsets, parents=parents, names=names, fps=30.0)


def zero_record(case: Case) -> ZeroRecord:
    from poseydon.build.pipeline import compose_build  # noqa: PLC0415
    from poseydon.build.sources import REST_CACHE, rig_cache  # noqa: PLC0415
    from poseydon.core.skeleton import SkeletonManifest  # noqa: PLC0415
    from poseydon.export import RigRecord  # noqa: PLC0415

    data = compose_build([f"data={case.data_name}"]).data
    root = Path(data.paths.root)
    manifest = SkeletonManifest.load(root / "rigs" / case.rig / "manifest.yaml")
    stage = next((s for s in data.prepare if s.type == "ground"), None)
    with np.load(root / "rigs" / case.rig / "skeleton.npz", allow_pickle=True) as z:
        joints = tuple(str(n) for n in z["names"])
    record = RigRecord.load(root, case.rig)
    pose = None if manifest.rest_pose is None else Path(data.paths.raw) / "source" / case.rig / manifest.rest_pose
    return ZeroRecord(tuple(s.name for s in record.chain.stages),
                      rig_cache(root, case.rig) / REST_CACHE, joints,
                      tuple((p.right, p.left) for p in manifest.facing), manifest.extra_yaw_deg, manifest.up_axis,
                      None if stage is None else str(manifest.ground or stage.get("mode", "joints")), record.fixed,
                      record.placement(zero_clip(record))[1], pose)


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def commit(root: str | Path = ".") -> str:
    """HEAD of the checkout, read from `.git` (the containers have no git binary)."""
    git = Path(root) / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        if (git / ref).is_file():
            return (git / ref).read_text().strip()
        packed = (git / "packed-refs").read_text().splitlines()
        return next(line.split()[0] for line in packed if line.endswith(" " + ref))
    except (OSError, StopIteration):
        return "unknown"


def sample_frames(world: np.ndarray, n: int) -> np.ndarray:
    """``n`` evenly spaced frames plus the frame of largest joint speed, sorted, unique.
    ``world`` is ``(F, N, 4, 4)``."""
    count = len(world)
    picked = set(np.linspace(0, count - 1, min(n, count)).round().astype(int))
    if count > 1:
        speed = np.linalg.norm(np.diff(world[..., :3, 3], axis=0), axis=-1).max(axis=1)
        picked.add(int(speed.argmax()) + 1)
    return np.array(sorted(picked), dtype=int)


def load_case(run_dir: str | Path) -> Case:
    meta = json.loads((Path(run_dir) / "case.json").read_text())
    return Case(meta["data"], meta["rig"], meta["action"],
                {k: Path(v) for k, v in meta["files"].items()}, Path(meta["clip_fbx"]), Path(meta["skin_fbx"]),
                np.array(meta["frames"], dtype=int), np.array(meta["source_frames"], dtype=int),
                tuple(meta["window"]) if meta.get("window") else None, bool(meta.get("floor", False)))
