"""Raw Truebones FBX -> +Z-facing clips with skin intact, plus one bind-pose OBJ per character.

The FBX counterpart to ``process_dataset_truebones.py``: same corpus, same
manifests, same +Z facing convention, but carrying the mesh and skinning
through, which BVH cannot represent at all. The FBX reading and writing
itself lives in :mod:`poseydon.io.fbx`, beside :mod:`poseydon.io.bvh`.

Two things the BVH path works for come free here:

* **T-pose-relative transforms.** FBX stores an explicit bind pose and
  Blender keeps pose bones relative to it, so animation is already
  rest-relative on import. Measured against the BVH pipeline's own
  reconstruction, the FBX bind pose IS the T-pose: 0.00 degrees median AND
  90th-percentile bone-direction difference over BrownBear's 37 comparable
  bones. There is no bind-pose removal step here; there is nothing to remove.
* **The skin following the rig.** Skinning is ``pose . rest^-1`` in world
  space, so turning the rig turns every deformed vertex with it.

Verified against the BVH pipeline on the same clips: joint positions agree
to 0.0000 at frame 0 and 0.2% of skeleton size across interpolated frames.
The two differ only in how they STORE a rotation -- each expresses a joint
in its own local bone frame, a constant per-joint offset -- so compare them
through world space, never by raw quaternion values.

Outputs, alongside the BVH pipeline's ``data/truebones/bvh/``:

    data/truebones/fbx/<Species>/<clip>.fbx   facing +Z, skin and curves intact
    data/truebones/mesh/<Species>.obj         the character in BIND pose, faced +Z

Run inside the fbx container:
    docker compose run --rm fbx blender --background \\
        --python scripts/process_dataset_truebones_fbx.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from poseydon.io.fbx import FBX

DEFAULT_RAW_ROOT = Path("data/truebones/Truebone_Z-OO")
DEFAULT_MANIFEST_DIR = Path("data/truebones/skeletons")
DEFAULT_FBX_DIR = Path("data/truebones/fbx")
DEFAULT_MESH_DIR = Path("data/truebones/mesh")
TARGET_AXIS = "+Z"


def facing_pairs(manifest: dict) -> list[tuple[str, str]]:
    facing = manifest["facing"]
    return [
        (facing["hips"]["right"], facing["hips"]["left"]),
        (facing["shoulders"]["right"], facing["shoulders"]["left"]),
    ]


def process_species(manifest: dict, raw_root: Path, fbx_dir: Path, mesh_dir: Path):
    """Process every FBX of one species. Returns (n_written, warnings)."""
    species = manifest["skeleton"]
    warnings: list[str] = []
    species_dir = raw_root / species
    if not species_dir.is_dir():
        return 0, [f"{species}: no raw directory at {species_dir}"]

    sources = sorted(species_dir.glob("*.fbx"))
    if not sources:
        return 0, [f"{species}: no .fbx files in {species_dir}"]

    pairs = facing_pairs(manifest)
    n_written = 0
    mesh_written = False

    for source in sources:
        # One bad file must not cost a species: Truebones ships rigs whose
        # bone sets differ between files, and a corrupt header or a missing
        # facing joint is a skip, not a failure of the other twenty.
        try:
            scene = FBX.read(source)

            # Frame 0, matching process_dataset_truebones.py: the facing
            # rotation is the one genuinely per-clip quantity.
            action = scene.action
            if action is not None:
                scene.set_frame(scene.frame_range()[0])
            scene.rotate(scene.facing_rotation(pairs, TARGET_AXIS))

            dot = scene.facing_dot(pairs)
            if dot < 0.99:
                warnings.append(f"{species}/{source.name}: facing is {dot:.4f}, not +Z")

            scene.write(fbx_dir / species / source.name)
            n_written += 1

            # One mesh per character, from the first file that carries one.
            # It has no frame to be faced from, so face the BIND pose: the
            # mesh is a per-character reference, not a clip.
            if not mesh_written and scene.meshes:
                scene.rotate(scene.facing_rotation(pairs, TARGET_AXIS, rest=True))
                scene.write_bind_pose_obj(mesh_dir / f"{species}.obj")
                mesh_written = True
        except Exception as error:  # noqa: BLE001 - collect, don't abort the corpus
            warnings.append(f"{species}/{source.name}: {type(error).__name__}: {error}")
            continue

    if not mesh_written:
        warnings.append(f"{species}: no mesh found in any FBX, no OBJ written")
    return n_written, warnings


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Process raw Truebones FBX files.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--fbx-dir", type=Path, default=DEFAULT_FBX_DIR)
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    parser.add_argument("--skeletons", nargs="*", default=None)
    args = parser.parse_args(argv)

    # `_`-prefixed files are shared fragments pulled in via `base:`, not
    # skeletons -- the same convention ingest and the BVH script use.
    manifests = sorted(
        p for p in args.manifest_dir.glob("*.yaml") if not p.stem.startswith("_")
    )
    if args.skeletons is not None:
        wanted = set(args.skeletons)
        manifests = [p for p in manifests if p.stem in wanted]

    total = 0
    all_warnings: list[str] = []
    for path in manifests:
        manifest = yaml.safe_load(path.read_text())
        written, warnings = process_species(manifest, args.raw_root, args.fbx_dir, args.mesh_dir)
        print(f"{manifest['skeleton']}: wrote {written} fbx", flush=True)
        total += written
        all_warnings.extend(warnings)

    print(f"\ntotal: {total} fbx across {len(manifests)} skeletons")
    print(f"meshes: {len(list(args.mesh_dir.glob('*.obj')))} obj -> {args.mesh_dir}")
    if all_warnings:
        print(f"\n{len(all_warnings)} warning(s):")
        for warning in all_warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
