"""One clip per rig at every stage of the round trip, as BVH and as FBX.

    <out>/<Rig>/1_source.{bvh,fbx}      the vendor's own files
    <out>/<Rig>/2_prepared.{bvh,fbx}    canonical, every joint
    <out>/<Rig>/3_model.{bvh,fbx}       what the model sees, reduced (FBX: expanded to drive the mesh)
    <out>/<Rig>/4_roundtrip.{bvh,fbx}   back on the user's rig

Every FBX is the rig's own skinned asset; 2_ and 3_ show it in canonical space.

The clip is the first one, other than the rest pose, that exists as a vendor
BVH, a vendor FBX and a prepared clip, so both ends have an independent
reference: open 1_ and 4_ side by side and they should agree up to the losses
the round trip makes by design (docs/guides/roundtrip.md). Every FBX is written
onto the asset the rig was calibrated on, the same character and mesh.
Needs the Autodesk FBX SDK:

    docker compose run --rm -T fbxsdk python scripts/build_stage_samples.py [--rigs Camel Flamingo]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from poseydon.build.index import action_slug, strip_skeleton_prefix
from poseydon.core.animation import RigidBodyAnimation
from poseydon.export import RigRecord
from poseydon.export.rig import same_clip_pairs
from poseydon.features.reduce import apply_reduction
from poseydon.io.bvh import BVH
from poseydon.io.fbx import FBX


def build(corpus: Path, raw: Path, rig: str, out: Path) -> str:
    record = RigRecord.load(corpus, raw, rig)
    calibration = record.calibration("source")
    pairs = same_clip_pairs(raw / "source" / rig, rig, None)
    pairs.sort(key=lambda pair: pair[0] == calibration.clip)  # rest pose last
    for source_bvh, source_fbx in pairs:
        action = strip_skeleton_prefix(action_slug(source_bvh.stem), rig)
        clip = corpus / "clips" / rig / f"{action}.npz"
        # Same name is not proof of same clip; frame counts are a cheap check
        # that catches a different take, which the comparison would misreport.
        if clip.is_file() and len(FBX.read(source_fbx).frames()) == BVH.read(source_bvh).rotations.shape[0]:
            break
    else:
        raise LookupError("no clip exists as vendor BVH, vendor FBX and prepared clip")
    prepared = RigidBodyAnimation.load(clip).as_rigid_body(joint_translation="drop")
    model = apply_reduction(prepared, record.reduction)
    restored = record.restore(model, facing_from=action)

    dest = out / rig
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_bvh, dest / "1_source.bvh")
    shutil.copy2(source_fbx, dest / "1_source.fbx")
    shutil.copy2(clip.with_suffix(".bvh"), dest / "2_prepared.bvh")
    in_prepared = record.calibration("prepared")
    FBX.from_animation(prepared, in_prepared).write(dest / "2_prepared.fbx", embed_textures=True)
    BVH.from_animation(model).write(dest / "3_model.bvh")
    FBX.from_animation(record.expand(model), in_prepared).write(dest / "3_model.fbx", embed_textures=True)
    BVH.from_animation(restored, channels=record.source_channels).write(dest / "4_roundtrip.bvh")
    FBX.from_animation(restored, calibration).write(dest / "4_roundtrip.fbx", embed_textures=True)
    return action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=Path("data/truebones"))
    parser.add_argument("--raw", type=Path, default=Path("data/_raw/truebones"))
    parser.add_argument("--out", type=Path, default=Path("tmp/truebones"))
    parser.add_argument("--rigs", nargs="*", default=None)
    args = parser.parse_args()
    rigs = args.rigs or sorted(p.parent.name for p in (args.corpus / "rigs").glob("*/preproc.npz"))
    failed = []
    for rig in rigs:
        try:
            print(f"{rig}: {build(args.corpus, args.raw, rig, args.out)}", flush=True)
        except Exception as error:  # noqa: BLE001 - one rig must not cost the others
            failed.append(rig)
            print(f"{rig}: FAILED {type(error).__name__}: {error}", flush=True)
    print(f"\n{len(rigs) - len(failed)}/{len(rigs)} rigs written to {args.out}")
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
