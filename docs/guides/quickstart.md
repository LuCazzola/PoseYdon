# Quick start: run the tests, train, sample, export

The only host requirement is Docker. No Python, pip or uv installation.

```bash
docker compose build test
docker compose run --rm test pytest
docker compose run --rm test python -m poseydon.cli list
```

Turn a generated clip back into an FBX on the rig you supplied, and look at it.
FBX needs the Autodesk FBX SDK, fetched once (see
[tooling](../index/tooling.md#dependencies)):

```bash
docker compose run --rm fbxsdk python -m poseydon.cli export \
    runs/<run>/retargets/step-75000/Flamingo_onelegbent__to__Scorpion.npz \
    --rig Scorpion --representation source --format fbx --out out/scorpion.fbx
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

Exports default to the strict prepared representation. Select `--representation prepared|source` and `--format bvh|fbx`:

    poseydon export samples/Rig__clip.anim.npz --rig Rig --representation prepared --format bvh --out prepared.bvh

Every FBX is the clip on the vendor's own skinned asset (`fbxsdk` container): a prepared one in canonical space, a source one in the source rig's own.
