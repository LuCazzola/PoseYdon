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


def _exporter(corpus: Path, rig: str):
    from poseydon.export import CharacterExporter
    raw = corpus.parent / "_raw" / "truebones"
    return CharacterExporter.load(corpus, raw, rig)


def rig_transform(corpus: Path, rig: str):
    """Compatibility view of the canonical CharacterExporter state."""
    exporter = _exporter(corpus, rig)
    return exporter.chain, exporter.params, exporter.reduction


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
    """Compatibility wrapper around CharacterExporter's inverse path."""
    exporter = _exporter(Path(corpus), rig)
    if facing is None:
        return exporter._restore(anim, stop_before=stop_before)
    current = invert_reduction(anim, exporter.reduction)
    params = {name: dict(value) for name, value in exporter.params.items()}
    params["face_axis"] = {"rotation": facing}
    for stage in reversed(exporter.chain.stages):
        if stage.name == stop_before:
            break
        current = stage.invert(current, params.get(stage.name, {}))
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
    """Compatibility wrapper returning CharacterExporter's chosen source FBX."""
    path = _exporter(Path(corpus), rig).source_fbx
    if path is None:
        raise FileNotFoundError(f"no source FBX for {rig}")
    return path


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

    from poseydon.export import CharacterExporter
    exporter = CharacterExporter.load(
        args.corpus, args.corpus.parent / "_raw" / "truebones", rig
    )
    if args.skin:
        # CharacterExporter owns reduction expansion, preparation inversion,
        # bind-pose alignment, and mesh-bearing source selection.
        scene = exporter.original_fbx(anim, facing_from=args.facing_from)
        if args.source_fbx is not None:
            from poseydon.io.fbx import FBX
            scene = FBX.read(args.source_fbx)
            scene.apply_animation(exporter._restore(anim, "rest_relative", args.facing_from), allow_undriven=True)
        scene.write(args.out, embed_textures=True)
        restored = exporter._restore(anim, "rest_relative", args.facing_from)
        skin = f", character-aware source FBX ({restored.n_frames} frames)"
    else:
        restored_bvh = exporter.original_bvh(anim, facing_from=args.facing_from)
        from poseydon.io.fbx import FBX
        FBX.from_animation(restored_bvh.to_animation(), name=rig).write(args.out)
        restored = restored_bvh.to_animation()
        skin = ""
    print(
        f"wrote {args.out}  ({rig}, {restored.n_frames} frames @ {restored.fps:g} fps, "
        f"{restored.n_joints} joints{skin})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
