"""Build each rig's rest asset: its vendor FBX held in the rig's rest pose.

    docker compose run --rm -T --entrypoint python -e HOME=/tmp fbxsdk tools/build_rest_fbx.py \
        [--data truebones|unimate] [Rig ...]

Writes, under `rig_cache` (`<paths.root>/cache/rigs/<Rig>/`):

- `rest.fbx`: the asset `poseydon export --format fbx` writes onto, by node name
  (`RigRecord.asset`), textures embedded. Mesh, skin and bind pose are the
  vendor's, so a posed mesh is exactly the vendor's; its one frame holds the rest
  pose (`FBX.pose`).
- `rest.npz`: its read (`scripts/build_fbx_cache.py`'s fields) plus `rest`, every
  node's world matrix in the rest pose, in float64, from which
  `FbxCache(rest_asset=True)` takes the rest pose without the SDK.
- `rest_mesh.npz` (truebones): the mesh at that pose for `ground: mesh`
  (`poseydon.build.prepare.PutOnGround`): `rest_pose` (the manifest value it was
  made for), `names`, `joints` (every node's position) and `vertices`.

truebones: the rest pose is the manifest's `rest_pose:`, a BVH, the format
`tools/tpose_author.py` authors in (15 rigs) or a vendor T-pose. It is carried onto
the FBX nodes through the vendor's BVH/FBX pair of one clip from
`data/truebones/fbx_clips.csv`: the two are one motion under a global similarity
(units, axes, placement) and a constant per-joint frame change, both solved on
that pair, which is checked on every frame (`CALIBRATION_TOL`). The pair with the
smallest residual among those whose FBX has a skin and at least
`MIN_PAIR_FRAMES` frames is used (a shorter one only if none has), and its FBX is the
asset. This is the one place the BVH-era files are still read; rerun a rig when
its `rest_pose:` changes (the build refuses a rest mesh made for another one).

unimate: the source FBX of the rig's rest clip (`FbxCache`), unchanged.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scripts.build_fbx_cache import read, repaired

from poseydon.build.corpus import cache_dir, rig_cache
from poseydon.build.prepare import REST_MESH, similarity
from poseydon.build.sources import REST_CACHE, REST_FBX, SOURCES
from poseydon.core.rotations import quat_to_matrix
from poseydon.core.skeleton import SkeletonManifest
from poseydon.io.bvh import BVH
from poseydon.io.fbx import FBX

ROOT = Path(__file__).resolve().parents[1]
#: Largest disagreement, as a fraction of the extent, between a BVH and an FBX
#: of one clip once calibrated. Above it they are not one motion.
CALIBRATION_TOL = 1e-4
#: Pairs this long are preferred for calibrating (`truebones_rest`).
MIN_PAIR_FRAMES = 10


def world_matrices(anim) -> np.ndarray:
    positions, rotations = anim.global_transforms()
    out = np.zeros((*positions.shape[:2], 4, 4))
    out[..., :3, :3] = quat_to_matrix(rotations)
    out[..., :3, 3], out[..., 3, 3] = positions, 1.0
    return out


def calibrate(bvh: np.ndarray, fbx: np.ndarray):
    """``(scale, rotation, translation, offsets, residual)`` taking ``(F, J, 4, 4)``
    BVH joint worlds onto FBX node worlds of the same joints and frames."""
    scale, rotation, translation = similarity(bvh[..., :3, 3].reshape(-1, 3), fbx[..., :3, 3].reshape(-1, 3))
    offsets = np.linalg.solve(scale * rotation @ bvh[0, :, :3, :3], fbx[0, :, :3, :3])
    mapped = scale * bvh[..., :3, 3] @ rotation.T + translation
    extent = np.ptp(fbx[0, :, :3, 3], axis=0).max()
    linear = np.abs(scale * rotation @ bvh[..., :3, :3] @ offsets - fbx[..., :3, :3]).max() / scale
    residual = max(float(np.linalg.norm(mapped - fbx[..., :3, 3], axis=-1).max() / extent), float(linear))
    return scale, rotation, translation, offsets, residual


def truebones_rest(data, rig: str) -> tuple[Path, dict[str, np.ndarray], str]:
    """The asset, the rest pose's world matrix per FBX node, and a note."""
    raw = ROOT / data.paths.raw / "source"
    manifest = SkeletonManifest.load(ROOT / data.paths.rigs / rig / "manifest.yaml")
    rest = BVH.read(raw / rig / manifest.rest_pose).to_animation().slice(0, 1)
    best = None
    with open(ROOT / data.source.index) as index:
        rows = [row for row in csv.DictReader(index) if row["rig"] == rig]
    for row in rows:
        with np.load(ROOT / data.source.cache / rig / f"{row['action']}.npz") as z:
            if not len(z["skin_mesh"]):
                continue
            names, world = [str(n) for n in z["names"]], z["world"]
        clip = BVH.read(raw / rig / row["bvh"]).to_animation()
        joints = [n for n in clip.names if n in names and n in rest.names]
        frames = min(len(world), clip.n_frames)
        fit = calibrate(world_matrices(clip)[:frames, [clip.names.index(n) for n in joints]],
                        world[:frames, [names.index(n) for n in joints]])
        # A one-frame pair (a T-pose) cannot show the frame change is constant.
        rank = (frames < MIN_PAIR_FRAMES, fit[-1])
        if best is None or rank < best[3]:
            best = (row, fit, joints, rank)
    if best is None:
        raise ValueError("no clip of the index has a skinned FBX")
    row, (scale, rotation, translation, offsets, residual), joints, _ = best
    if residual > CALIBRATION_TOL:
        raise ValueError(f"best pair {row['bvh']} / {row['file']} is off by {residual:.2e} of the extent")
    w = world_matrices(rest)[0, [rest.names.index(n) for n in joints]]
    out = np.zeros_like(w)
    out[:, :3, :3] = scale * rotation @ w[:, :3, :3] @ offsets
    out[:, :3, 3] = scale * w[:, :3, 3] @ rotation.T + translation
    out[:, 3, 3] = 1.0
    asset = repaired(raw / row["file"], cache_dir(ROOT / data.paths.root))
    return asset, dict(zip(joints, out)), f"{manifest.rest_pose} via {row['bvh']} / {row['file']} ({residual:.1e})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="truebones")
    parser.add_argument("rigs", nargs="*")
    args = parser.parse_args()
    data = OmegaConf.load(ROOT / "configs" / "data" / f"{args.data}.yaml")
    source = SOURCES.build(data.source)
    rigs = args.rigs or source.rigs(data)
    failed = []
    for rig in rigs:
        out = rig_cache(ROOT / data.paths.root, rig)
        out.mkdir(parents=True, exist_ok=True)
        try:
            if args.data == "truebones":
                asset, world, note = truebones_rest(data, rig)
                scene = FBX.read(asset)
                rest = {"rest": scene.pose(world, 30.0)}
                scene.write(out / REST_FBX, embed_textures=True)
                scene.close()
            else:
                _, first = source._rig(rig)
                with np.load(ROOT / data.source.cache / rig / f"{first}.npz") as z:
                    asset = repaired(Path(str(z["source"])), cache_dir(ROOT / data.paths.root))
                scene = FBX.read(asset)
                scene.write(out / REST_FBX, embed_textures=True)
                scene.close()
                note, rest = f"bind of {asset.name}", {}
            d = read(out / REST_FBX)
            np.savez_compressed(out / REST_CACHE, **d, **rest)
            if args.data == "truebones":
                index = {n: j for j, n in enumerate(str(x) for x in d["names"])}
                extent = np.ptp([m[:3, 3] for m in world.values()], axis=0).max()
                held = max(np.abs(d["world"][0, index[n]] - m).max() / extent for n, m in world.items())
                scene = FBX.read(out / REST_FBX)
                vertices = scene.skinned_vertices(0)
                scene.close()
                manifest = SkeletonManifest.load(ROOT / data.paths.rigs / rig / "manifest.yaml")
                np.savez_compressed(out / REST_MESH, rest_pose=np.array(manifest.rest_pose), names=d["names"],
                                    joints=d["world"][0, :, :3, 3], vertices=vertices)
                note += f"; the file holds the pose to {held:.1e}; {len(vertices)} vertices"
        except Exception as error:  # noqa: BLE001 - one rig must not stop the rest
            failed.append(rig)
            print(f"{rig}: FAILED {type(error).__name__}: {error}", flush=True)
            continue
        print(f"{rig}: {note}", flush=True)
    print(f"{len(rigs) - len(failed)} built, {len(failed)} failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
