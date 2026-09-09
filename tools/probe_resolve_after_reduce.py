"""Does resolve() survive reduction, for every rig?"""
from pathlib import Path

from scripts.process_dataset_truebones import rest_action

from poseydon.core.skeleton import ManifestError, SkeletonManifest, resolve
from poseydon.features.reduce import apply_reduction, build_reduction
from poseydon.io.bvh import BVH

CORPUS = Path("data/truebones")
bad = []
for m in sorted((CORPUS / "rigs").glob("*/manifest.yaml")):
    rig = m.parent.name
    manifest = SkeletonManifest.load(m)
    clip = CORPUS / "clips" / rig / f"{rest_action(manifest)}.bvh"
    if not clip.is_file():
        continue
    anim = BVH.read(clip).to_animation().as_rigid_body(joint_translation="drop")
    reduced = apply_reduction(anim, build_reduction(anim))
    try:
        resolve(manifest, reduced.names)
    except ManifestError as e:
        bad.append((rig, anim.n_joints, reduced.n_joints, str(e)[:80]))
print(f"rigs where resolve() FAILS after reduction: {len(bad)}")
for rig, full, red, msg in bad:
    print(f"  {rig:<18} {full}->{red}  {msg}")
