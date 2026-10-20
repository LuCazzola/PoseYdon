"""Cache each rig's rest-pose mesh for `ground: mesh` (`poseydon.build.prepare.PutOnGround`).

The vendor FBX skin is posed on the first frame of the rig's `rest_pose:` clip
and written, in the raw BVH's space, as
`cache/rigs/<Rig>/rest_mesh.npz`: `rest_pose` (the clip it was made for), `names` and
`joints` (that frame's joint positions) and `vertices`. The build maps it onto
the rest pose as it stands at grounding, so it needs no FBX SDK; this does, so
it runs in the `fbxsdk` container (x86-64, emulated on ARM hosts: minutes per
rig). Rerun it for a rig whenever its `rest_pose:` changes: the build refuses a
cache made for another clip and grounds that rig on its joints, with a warning.

    docker compose run --rm -T --entrypoint python -e HOME=/tmp fbxsdk tools/build_rest_mesh.py [Rig ...]

The cache holds vendor geometry: `*.npz` under `data/` is gitignored, keep it so.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from poseydon.build.corpus import rest_source, rig_cache
from poseydon.build.prepare import REST_MESH
from poseydon.core.skeleton import SkeletonManifest
from poseydon.export.rig import source_calibration
from poseydon.ingest.pipeline import available_rigs
from poseydon.io.bvh import BVH
from poseydon.io.fbx import FBX

CORPUS, RAW = Path("data/truebones"), Path("data/_raw/truebones")


def rest_mesh(rig: str) -> dict[str, np.ndarray]:
    manifest = SkeletonManifest.load(CORPUS / "rigs" / rig / "manifest.yaml")
    rest = BVH.read(rest_source(manifest, sorted((RAW / "source" / rig).glob("*.bvh")))).to_animation().slice(0, 1)
    calibration = source_calibration(CORPUS, RAW, rig)
    vertices = calibration.to_source(FBX.from_animation(rest, calibration).skinned_vertices(0))
    return {"rest_pose": np.array(manifest.rest_pose), "names": np.array(rest.names),
            "joints": rest.global_positions()[0], "vertices": vertices}


def main() -> int:
    rigs = sys.argv[1:] or sorted(available_rigs(CORPUS / "rigs"))
    failed = []
    for rig in rigs:
        try:
            cache = rest_mesh(rig)
        except Exception as error:  # noqa: BLE001 - one rig without a usable FBX must not stop the rest
            failed.append(rig)
            print(f"{rig}: FAILED {type(error).__name__}: {error}", flush=True)
            continue
        out = rig_cache(CORPUS, rig)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / REST_MESH, **cache)
        below = cache["joints"][:, 1].min() - cache["vertices"][:, 1].min()
        print(f"{rig}: {len(cache['vertices'])} vertices, lowest {below:+.3f} below the lowest joint (raw units, along raw Y)", flush=True)
    print(f"{len(rigs) - len(failed)} cached, {len(failed)} failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
