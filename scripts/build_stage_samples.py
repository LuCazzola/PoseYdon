"""One clip per rig at every stage of the round trip, as BVH and as FBX.

    <out>/<Rig>/1_source.{bvh,fbx}      the source clip as read (BVH), the vendor FBX it was read from
    <out>/<Rig>/2_prepared.{bvh,fbx}    canonical, every joint
    <out>/<Rig>/3_model.{bvh,fbx}       what the model sees, reduced (FBX: expanded to drive the mesh)
    <out>/<Rig>/4_roundtrip.{bvh,fbx}   back on the user's rig

Every FBX but 1_ is written onto the rig's rest asset (`RigRecord.asset`), the
same character and mesh, held in the rest pose; 2_ and 3_ show it in
canonical space (`RigRecord.placement`). The clip is the first of the rig's
training clips that has a vendor FBX in the dataset's clip index, so 1_source.fbx
is an independent reference: open it and 4_ side by side and they should agree
up to the losses the round trip makes by design
(docs/guides/concepts/data/roundtrip.md). Needs the Autodesk FBX SDK:

    docker compose run --rm -T fbxsdk python scripts/build_stage_samples.py [--data truebones] [--rigs Camel Flamingo]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from poseydon.build.pipeline import compose_build
from poseydon.build.sources import SOURCES, fbx_index
from poseydon.core.animation import RigidBodyAnimation
from poseydon.core.skeleton import SkeletonManifest
from poseydon.export import RigRecord
from poseydon.features.reduce import apply_reduction
from poseydon.io.bvh import BVH
from poseydon.io.fbx import FBX


def build(data, rig: str, out: Path) -> str:
    corpus = Path(data.paths.root)
    record = RigRecord.load(corpus, rig)
    files = {action: file for r, action, file in fbx_index(data.source.index) if r == rig}
    trained = [json.loads(line)["action"] for line in (corpus / "index.jsonl").read_text().splitlines()
               if line and json.loads(line)["skeleton"] == rig]
    action = next((a for a in sorted(trained) if a in files), None)
    if action is None:
        raise LookupError("no training clip has a vendor FBX in the clip index")
    clip = corpus / "clips" / rig / f"{action}.npz"
    prepared = RigidBodyAnimation.load(clip).as_rigid_body(joint_translation="drop")
    model = apply_reduction(prepared, record.reduction)
    restored = record.restore(model, facing_from=action)
    manifest = SkeletonManifest.load(corpus / "rigs" / rig / "manifest.yaml")
    source = SOURCES.build(data.source).clip(data, manifest, action).anim

    dest = out / rig
    dest.mkdir(parents=True, exist_ok=True)
    BVH.from_animation(source).write(dest / "1_source.bvh")
    shutil.copy2(Path(data.paths.raw) / "source" / files[action], dest / "1_source.fbx")
    shutil.copy2(clip.with_suffix(".bvh"), dest / "2_prepared.bvh")
    for name, anim in (("2_prepared", prepared), ("3_model", model)):
        placed, space = record.placement(anim, facing_from=action)
        FBX.from_animation(placed, record.asset, space, record.fixed).write(dest / f"{name}.fbx", embed_textures=True)
    BVH.from_animation(model).write(dest / "3_model.bvh")
    BVH.from_animation(restored, channels=record.source_channels).write(dest / "4_roundtrip.bvh")
    FBX.from_animation(restored, record.asset, fixed=record.fixed).write(dest / "4_roundtrip.fbx", embed_textures=True)
    return action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="truebones", help="a dataset with an `fbx_cache` source")
    parser.add_argument("--out", type=Path, default=None, help="default: tmp/<data>")
    parser.add_argument("--rigs", nargs="*", default=None)
    args = parser.parse_args()
    data = compose_build([f"data={args.data}"]).data
    args.out = args.out or Path("tmp") / args.data
    rigs = args.rigs or sorted(p.parent.name for p in (Path(data.paths.root) / "rigs").glob("*/preproc.npz"))
    failed = []
    for rig in rigs:
        try:
            print(f"{rig}: {build(data, rig, args.out)}", flush=True)
        except Exception as error:  # noqa: BLE001 - one rig must not cost the others
            failed.append(rig)
            print(f"{rig}: FAILED {type(error).__name__}: {error}", flush=True)
    print(f"\n{len(rigs) - len(failed)}/{len(rigs)} rigs written to {args.out}")
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
