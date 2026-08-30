"""Bind a fixture BVH to its shipped skeleton manifest."""

from pathlib import Path

from poseydon.core.skeleton import SkeletonManifest, resolve

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO_ROOT / "data" / "truebones" / "skeletons"

# Fixture stems encode the species differently from clip to clip, so map explicitly.
STEM_TO_SKELETON = {
    "BrownBear___RiseSwat_132": "BrownBear",
    "Coyote___Attack3_224": "Coyote",
    "Crab___Attack3_234": "Crab",
    "Flamingo_Flamingo_OneLEgBEnt_353": "Flamingo",
    "Goat___HeadButt_395": "Goat",
    "Scorpion___SlowForward_839": "Scorpion",
    "Skunk___Spray_891": "Skunk",
}


def resolved_for(bvh_path, anim):
    skeleton = STEM_TO_SKELETON[Path(bvh_path).stem]
    manifest = SkeletonManifest.load(MANIFEST_DIR / f"{skeleton}.yaml")
    return resolve(manifest, anim.names)
