"""A generated clip -> an FBX on the rig the user authored.

The last arrow of the pipeline, and the one the invertible preparation exists
for. `scripts/process_dataset_truebones.py` records every fitted parameter to
`rigs/<Rig>/prepare.npz` precisely so this direction is possible; stage 2
folds the rig-scoped half into `rigs/<Rig>/skeleton.npz`. This reads them
back:

    generated clip          reduced joints, prepared frame (what a model emits)
      invert_reduction      every joint the corpus keeps
      PrepareChain.invert   the source's names, hierarchy, offsets, units, yaw
      FBX.from_animation    an armature, keyed, exported

Run it in the `fbx` container, which is the only image with Blender:

    docker compose run --rm fbx blender -b --python scripts/anim_2_fbx.py -- \\
        runs/<run>/retargets/step-75000/Flamingo_onelegbent__to__Scorpion.npz \\
        --out out/scorpion.fbx

`--skin` exports the character rather than its skeleton. It drives the rig's
own source asset -- mesh, skin weights, materials intact -- instead of building
a bare armature, and stops the inversion one stage short so the clip is still
measured from the T-pose the bind pose also means. Off by default: a skeleton
needs only the corpus, while this needs the source `.fbx` to still be there.

Two limits, both inherited and both deliberate:

* **Per-joint translation does not come back.** `EnforceRigid` replaced it
  with the rest pose's, so non-root joints return pinned to their rest
  offset. For a generated clip nothing is lost -- the model never produced
  per-joint translation to begin with.
* **Facing is not recoverable.** `FaceAxis` is the one clip-scoped stage and
  a generated clip has no recorded yaw, so the export faces +Z unless
  `--facing-from <action>` borrows a real clip's.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from poseydon.build.index import is_all_takes_bundle
from poseydon.build.prepare import PrepareChain
from poseydon.core.animation import Animation, RigidBodyAnimation
from poseydon.core.rotations import QUAT_IDENTITY
from poseydon.features.reduce import JointReduction, build_reduction, invert_reduction
from poseydon.io.bvh import BVH
from scripts.process_dataset_truebones import _chain_for

DEFAULT_CORPUS = Path("data/truebones")


def rig_transform(
    corpus: Path, rig: str
) -> tuple[PrepareChain, dict[str, dict], JointReduction]:
    """The chain that prepared `rig`, its rig-scoped parameters, and its reduction.

    Everything comes from `rigs/<Rig>/skeleton.npz`, which `build_skeleton`
    made self-contained for exactly this reason -- so undoing a preparation
    needs neither Blender nor the raw corpus, only the artefact the training
    run itself read.

    The chain is the production one (`_chain_for`), not a private copy of the
    stage list. `PrepareChain.invert` reads nothing but `params`, so the
    stages' own fields do not affect the result -- but the ORDER does, and a
    second list of seven stages that has to stay in step with the first is
    exactly the kind of duplication this codebase exists to avoid.

    The reduction is rebuilt rather than loaded: `reduction_source_of` is a
    reindex map and a COLLAPSE removal composes a rotation, which no reindex
    map carries. The stored map is used to CHECK the rebuild, the same way
    `MotionDataset._reduction` does.

    `enforce_rigid/source_channels` is deliberately NOT read. It is the one
    genuinely ragged array in the file, so it is stored as a pickled object
    array (`RigTransform`'s docstring says so), and this script runs in the
    `fbx` image -- whose Blender-linked numpy is older than the one that wrote
    the corpus, and which therefore cannot unpickle it at all
    (`ModuleNotFoundError: numpy._core`). Nothing here needs it: it records a
    BVH file's CHANNELS declaration for `BVH.from_animation`, and both
    `EnforceRigid.apply` and `.invert` ignore their parameters entirely. Not
    reading it is what keeps this path free of pickle, and so free of the
    numpy version skew between the two images.
    """
    path = corpus / "rigs" / rig / "skeleton.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"no prepared rig `{rig}` at {path}; `poseydon list` names the rigs "
            "this corpus has"
        )

    with np.load(path) as data:
        params: dict[str, dict] = {}
        for key in data.files:
            if not key.startswith("prepare/"):
                continue
            _, stage, field = key.split("/")
            params.setdefault(stage, {})
            if (stage, field) != ("enforce_rigid", "source_channels"):
                params[stage][field] = data[key]
        rest = RigidBodyAnimation.from_root_motion(
            rotations=np.tile(QUAT_IDENTITY, (1, len(data["parents"]), 1)),
            root_pos=np.zeros((1, 3)),
            offsets=data["offsets"],
            parents=data["parents"].astype(np.int32),
            names=tuple(str(n) for n in data["names"]),
            fps=1.0,
        )
        stored = tuple(int(i) for i in data["reduction_source_of"])

    # `build_reduction` selects on `norm(offsets)`, `parents` and `names`, so
    # the identity rotations above cannot change which joints it removes --
    # the same invariant `build_skeleton`, `build_stats` and the read path all
    # rely on, stated in full at `build/pipeline.py::build_skeleton`.
    reduction = build_reduction(rest)
    if reduction.source_of != stored:
        raise ValueError(
            f"{rig}: the reduction rebuilt here does not match the one "
            f"`skeleton.npz` records, so the joints a model generated do not "
            f"mean what this would put them back as. Rebuild stage 2."
        )

    # `None` is `EnforceRigid`'s documented fallback for "a caller with no file
    # to hand", and is the accurate answer here: this one has no BVH to declare
    # channels for. It changes nothing about the inverse -- see above.
    return _chain_for(None, rig), params, reduction


def recorded_facing(corpus: Path, rig: str, action: str) -> np.ndarray:
    """The yaw `FaceAxis` fitted for one real clip of `rig`.

    Read from `rigs/<Rig>/prepare.npz`, which is the only file that keeps the
    clip-scoped half of the transform; `skeleton.npz` folds in the rig-scoped
    stages alone.
    """
    path = corpus / "rigs" / rig / "prepare.npz"
    key = f"clip/{action}/face_axis/rotation"
    with np.load(path, allow_pickle=True) as data:
        if key not in data.files:
            known = sorted(
                f.split("/")[1] for f in data.files if f.startswith("clip/")
            )
            raise KeyError(
                f"{rig} has no recorded clip `{action}`; it has: {', '.join(known)}"
            )
        return data[key]


def unprepare(
    anim: RigidBodyAnimation,
    rig: str,
    corpus: Path = DEFAULT_CORPUS,
    facing: np.ndarray | None = None,
    stop_before: str | None = None,
) -> Animation:
    """A generated clip, on the joints and in the frame the user supplied.

    `facing` defaults to identity, which leaves the character facing +Z: the
    honest answer for motion a model generated, which was never oriented any
    other way. Pass `recorded_facing(...)` to land it like a real take.

    `stop_before` names a stage to leave UNDONE, stopping the inversion short.
    Only `"rest_relative"` is useful, and only for `--skin`: see the loop below.
    """
    chain, params, reduction = rig_transform(corpus, rig)

    expected = tuple(reduction.source_names[i] for i in reduction.source_of)
    if tuple(anim.names) != expected:
        raise ValueError(
            f"this clip's {anim.n_joints} joints are not {rig}'s reduced joint "
            f"set ({len(expected)} joints, starting {expected[:3]}); it was "
            "generated on a different rig, or against a different corpus build"
        )

    params = {
        **params,
        "face_axis": {"rotation": QUAT_IDENTITY if facing is None else facing},
    }

    # `PrepareChain.invert` with a stopping point. Written out rather than
    # called because `stop_before` is the whole difference between the two
    # exports: the skinned one stops at `rest_relative`, which is the stage
    # that made identity rotation mean "T-pose" -- the one thing an FBX bind
    # pose and a prepared clip already agree on. Undoing it would hand
    # `FBX.apply_animation` rotations measured from a different zero.
    current = invert_reduction(anim, reduction)
    for stage in reversed(chain.stages):
        if stage.name == stop_before:
            return current
        current = stage.invert(current, params[stage.name])
    return current


def load_generated(path: Path) -> tuple[RigidBodyAnimation, str | None]:
    """A generated clip, plus the rig it names if the file records one.

    Rotations are what an FBX needs, and the `.npz` a retarget writes carries
    POSITIONS -- `predicted_positions` and `solved_positions` -- because that
    is what its metrics are measured on. The rotations for the very same clip
    are in the `.bvh` written beside it, so that is where they are read from
    rather than solved for a second time: an IK pass here would produce a
    different animation from the one the run already published under the same
    name.
    """
    if path.suffix == ".bvh":
        return BVH.read(path).to_animation().as_rigid_body(joint_translation="drop"), None

    with np.load(path, allow_pickle=True) as data:
        keys = set(data.files)
        rig = str(data["target_rig"]) if "target_rig" in keys else None

    if "rotations" in keys:
        return Animation.load(path).as_rigid_body(joint_translation="drop"), rig

    sibling = path.with_suffix(".bvh")
    if not sibling.is_file():
        raise ValueError(
            f"{path} carries joint positions but no rotations, and there is no "
            f"{sibling.name} beside it to take them from. It was written with "
            "`reconstruct=positions`, which yields no rotations at all -- an "
            "FBX cannot be built from it. Re-sample with `reconstruct.name=fk` "
            "or `positions_ik`."
        )
    return BVH.read(sibling).to_animation().as_rigid_body(joint_translation="drop"), rig


def source_asset(corpus: Path, rig: str) -> Path:
    """The rig's own FBX, the one `--skin` drapes generated motion over.

    Any per-clip file will do: what is wanted is the bind pose, the mesh and
    the skin weights, all of which describe the CHARACTER, and the animation it
    happens to carry is overwritten. The all-takes compilation is excluded the
    way every other consumer excludes it -- through `is_all_takes_bundle`, not
    a suffix or substring test on the stem, both of which cost real files (see
    that function).
    """
    directory = corpus / "source" / rig
    candidates = sorted(
        path
        for path in directory.glob("*.fbx")
        if not is_all_takes_bundle(path.stem, rig)
    )
    if not candidates:
        raise FileNotFoundError(
            f"no per-clip source .fbx for {rig} under {directory}, so there is "
            "no mesh to skin; export without --skin, or pass --source-fbx"
        )
    return candidates[0]


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        # Blender swallows its own arguments; everything after `--` is ours.
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Turn a generated clip back into an FBX on the source rig."
    )
    parser.add_argument("input", type=Path, help="a generated .npz or .bvh")
    parser.add_argument("--out", type=Path, required=True, help="the .fbx to write")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--rig",
        default=None,
        help="the rig to return this clip on (default: the one the .npz names)",
    )
    parser.add_argument(
        "--facing-from",
        default=None,
        metavar="ACTION",
        help="borrow this clip's recorded yaw instead of facing +Z",
    )
    parser.add_argument(
        "--skin",
        action="store_true",
        help="drive the rig's SOURCE asset instead, so the export carries its "
        "mesh, skin weights and materials (default: a bare skeleton)",
    )
    parser.add_argument(
        "--source-fbx",
        type=Path,
        default=None,
        help="the asset --skin should drive (default: the rig's first "
        "per-clip source .fbx)",
    )
    args = parser.parse_args(argv)

    anim, recorded_rig = load_generated(args.input)
    rig = args.rig or recorded_rig
    if rig is None:
        print(
            f"{args.input} does not record which rig it is on; pass --rig",
            file=sys.stderr,
        )
        return 1

    facing = (
        recorded_facing(args.corpus, rig, args.facing_from)
        if args.facing_from
        else None
    )
    # `--skin` drives the source asset, whose bind pose IS the T-pose, so it
    # needs the clip measured from that same zero -- `rest_relative` left
    # undone. The bare skeleton is built standalone and wants the source's own
    # bind rotations back, so it undoes everything.
    restored = unprepare(
        anim,
        rig,
        args.corpus,
        facing=facing,
        stop_before="rest_relative" if args.skin else None,
    )

    # Imported here, not at the top: `poseydon.io.fbx` needs Blender, and
    # everything above it is testable without one.
    from poseydon.io.fbx import FBX

    if args.skin:
        from poseydon.export import CharacterExporter
        raw_root = args.corpus.parent / "_raw" / "truebones"
        exporter = CharacterExporter.load(args.corpus, raw_root, rig)
        scene = exporter.original_fbx(anim)
        if args.source_fbx is not None:
            scene = FBX.read(args.source_fbx)
            residual = scene.apply_animation(restored)
        else:
            residual = 0.0
        scene.write(args.out, embed_textures=True)
        skin = f", character-aware source FBX (bind fit {residual:.2e} bone lengths)"
    else:
        FBX.from_animation(restored, name=rig).write(args.out)
        skin = ""
    print(
        f"wrote {args.out}  ({rig}, {restored.n_frames} frames @ {restored.fps:g} fps, "
        f"{restored.n_joints} joints{skin})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
