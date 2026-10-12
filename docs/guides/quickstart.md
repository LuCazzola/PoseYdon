# Quick start: run the tests, train, sample, export

The only host requirement is Docker. No Python, pip or uv installation.

```bash
docker compose build test
docker compose run --rm test pytest
docker compose run --rm test python -m poseydon.cli list
```

Two files need Blender, which the `test` image deliberately does not ship, so
they are excluded from the run above and must be run inside the `fbx`
container instead -- `tests/build/test_roundtrip_fbx.py` (FBX read/write and
`from_animation`/`apply_animation`) and `tests/io/test_render_fbx.py` (the
Blender renderer):

```bash
docker compose run --rm fbx blender -b --python-expr \
    "import sys, pytest; sys.exit(pytest.main(['tests/build/test_roundtrip_fbx.py','tests/io/test_render_fbx.py','-v','-rs']))"
```

Turn a generated clip back into an FBX on the rig you supplied, and look at it:

```bash
docker compose run --rm fbx blender -b --python scripts/anim_2_fbx.py -- \
    runs/<run>/retargets/step-75000/Flamingo_onelegbent__to__Scorpion.npz \
    --skin --out out/scorpion.fbx
docker compose run --rm fbx blender -b --python scripts/render_fbx.py -- \
    out/scorpion.fbx --out out/scorpion.mp4 --threads 4
```

Ingest a BVH corpus, train, and sample:

```bash
poseydon ingest <bvh_dir> --manifests data/truebones/rigs --out data/truebones
poseydon train  trainer=debug
poseydon sample --checkpoint runs/**/last.ckpt skeleton=Goat n_samples=4 --render
```

### Representation-aware export

Exports default to the strict prepared representation. Select `--representation prepared|presentation|original` and `--format bvh|fbx`:

    poseydon export samples/Rig__clip.bvh --rig Rig --representation prepared --format bvh --out prepared.bvh

BVH-only rigs expose BVH representations and report `source_surface_missing` for FBX requests. Paired or FBX-only rigs expose mesh-bearing prepared/presentation FBX assets when preparation produced them. Inspect each rig's `report.json` and index availability fields before requesting a representation.
