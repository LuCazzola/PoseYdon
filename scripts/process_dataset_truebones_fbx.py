"""Raw Truebones FBX -> +Z-facing clips with skin intact, plus one mesh.npz per rig.

The FBX counterpart to ``process_dataset_truebones.py``: same corpus, same
manifests, same entity-first layout, same +Z facing convention, but carrying
the mesh and skinning through, which BVH cannot represent at all. The FBX
reading and writing itself lives in :mod:`poseydon.io.fbx`, beside
:mod:`poseydon.io.bvh`.

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
to 0.0000 at frame 0 and 0.2% of skeleton size across interpolated frames --
now a test, ``tests/build/test_bvh_fbx_agreement.py``, rather than only a
claim in a commit message. The two differ only in how they STORE a
rotation -- each expresses a joint in its own local bone frame, a constant
per-joint offset -- so compare them through world space, never by raw
quaternion values.

Outputs, alongside the BVH pipeline's ``data/truebones/clips/``:

    data/truebones/clips/<Rig>/<action>.fbx   facing +Z, skin and curves intact
    data/truebones/rigs/<Rig>/mesh.npz        mesh + skin weights + rest skeleton,
                                               faced +Z, bound to the same joints

``action`` is derived exactly as the BVH script derives it --
``strip_skeleton_prefix(action_slug(path.stem), rig)`` -- so the two
corpora share basenames; the agreement test and later consumers depend on
this. ``--write-obj`` additionally writes a bind-pose OBJ per rig, geometry
only, for quick inspection in a mesh viewer.

Run inside the fbx container:
    docker compose run --rm fbx blender --background \\
        --python scripts/process_dataset_truebones_fbx.py -- --rigs Goat Crab Flamingo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from poseydon.core.skeleton import SkeletonManifest
from poseydon.ingest.index import action_slug, strip_skeleton_prefix
from poseydon.ingest.pipeline import available_rigs
from poseydon.io.fbx import FBX

DEFAULT_RAW_ROOT = Path("data/truebones/source")
DEFAULT_OUT_ROOT = Path("data/truebones")
TARGET_AXIS = "+Z"


def facing_pairs(manifest: SkeletonManifest) -> list[tuple[str, str]]:
    return [(pair.right, pair.left) for pair in manifest.facing]


def process_species(
    manifest: SkeletonManifest, raw_root: Path, out_root: Path, write_obj: bool
) -> tuple[int, list[str]]:
    """Process every FBX of one rig. Returns (n_written, warnings)."""
    species = manifest.name
    warnings: list[str] = []
    species_dir = raw_root / species
    if not species_dir.is_dir():
        return 0, [f"{species}: no raw directory at {species_dir}"]

    # "ALL" files are Truebones' own all-takes compilations -- every
    # animation of the rig concatenated into one clip, e.g. `GoatAll.fbx`,
    # `scorpionALL.fbx`, `Flamingo-ALL.fbx` -- not a distinct action, and they
    # have no BVH counterpart to pair basenames against. They are always the
    # WHOLE stem ending in "all"; a hyphenated take that merely starts with
    # the rig name plus "ALL" (Anaconda ships `AnacondaALL-Twistrattle.fbx`
    # alongside the real `AnacondaALL.fbx`) is a real, separate clip and must
    # not be swept up by a bare substring match.
    sources = sorted(
        p for p in species_dir.glob("*.fbx") if not p.stem.lower().endswith("all")
    )
    if not sources:
        return 0, [f"{species}: no .fbx files in {species_dir}"]

    pairs = facing_pairs(manifest)
    clips_dir = out_root / "clips" / species
    rig_dir = out_root / "rigs" / species
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
            scene.scale_to_mean_bone_length()

            dot = scene.facing_dot(pairs)
            if dot < 0.99:
                warnings.append(f"{species}/{source.name}: facing is {dot:.4f}, not +Z")

            action_name = strip_skeleton_prefix(action_slug(source.stem), species)
            scene.write(clips_dir / f"{action_name}.fbx")
            n_written += 1

            # One mesh per character, from the first file that carries one.
            # It has no frame to be faced from, so face the BIND pose: the
            # mesh is a per-rig reference, not a clip.
            if not mesh_written and scene.meshes:
                scene.rotate(scene.facing_rotation(pairs, TARGET_AXIS, rest=True))
                rig_dir.mkdir(parents=True, exist_ok=True)
                scene.write_mesh_npz(rig_dir / "mesh.npz")
                if write_obj:
                    scene.write_bind_pose_obj(rig_dir / "mesh.obj")
                mesh_written = True
        except Exception as error:  # noqa: BLE001 - collect, don't abort the corpus
            warnings.append(f"{species}/{source.name}: {type(error).__name__}: {error}")
            continue

    if not mesh_written:
        warnings.append(f"{species}: no mesh found in any FBX, no mesh.npz written")
    return n_written, warnings


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Process raw Truebones FBX files.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--rigs",
        nargs="*",
        default=None,
        help="Rig names to process (default: every rig under --out-root/rigs)",
    )
    parser.add_argument(
        "--write-obj",
        action="store_true",
        help="Also write a geometry-only bind-pose OBJ per rig, for quick inspection.",
    )
    args = parser.parse_args(argv)

    rig_names = available_rigs(args.out_root / "rigs")
    if args.rigs is not None:
        wanted = set(args.rigs)
        rig_names = [name for name in rig_names if name in wanted]

    total = 0
    all_warnings: list[str] = []
    for rig_name in rig_names:
        manifest = SkeletonManifest.load(args.out_root / "rigs" / rig_name / "manifest.yaml")
        written, warnings = process_species(
            manifest, args.raw_root, args.out_root, args.write_obj
        )
        print(f"{manifest.name}: wrote {written} fbx", flush=True)
        total += written
        all_warnings.extend(warnings)

    print(f"\ntotal: {total} fbx across {len(rig_names)} rigs")
    if all_warnings:
        print(f"\n{len(all_warnings)} warning(s):")
        for warning in all_warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
