"""Does resolve() survive reduction, for every rig?"""
from pathlib import Path
from poseydon.core.skeleton import SkeletonManifest, resolve, ManifestError
from poseydon.features.reduce import build_reduction, apply_reduction
from poseydon.io.bvh import BVH
from scripts.process_dataset_truebones import rest_action

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
