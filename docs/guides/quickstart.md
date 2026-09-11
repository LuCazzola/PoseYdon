# Quick start

The only host requirement is Docker. No Python, pip or uv installation.

```bash
docker compose build test
docker compose run --rm test pytest
docker compose run --rm test python -m poseydon.cli list
```

`tests/build/test_roundtrip_fbx.py` needs Blender, which the `test` image
deliberately does not ship, so it is excluded from the run above and must be
run inside the `fbx` container instead:

```bash
docker compose run --rm fbx blender -b --python-expr \
    "import sys, pytest; sys.exit(pytest.main(['tests/build/test_roundtrip_fbx.py','-v','-rs']))"
```

Ingest a BVH corpus, train, and sample:

```bash
poseydon ingest <bvh_dir> --manifests data/truebones/rigs --out data/truebones
poseydon train  trainer=debug
poseydon sample --checkpoint runs/**/last.ckpt skeleton=Goat n_samples=4 --render
```

