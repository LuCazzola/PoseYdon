"""Read the reference's cond.npy with the real Motion package and export it flat.

cond.npy pickles Holden `Quaternions` objects, so it cannot be opened without
installing Motion from git. This converts it once into a plain .npz per skeleton
that PoseYdon reads with no such dependency -- and converts quaternions from
Holden's scalar-first (w, x, y, z) to the project-wide scalar-last layout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

COND = Path("external/neural_motion_blending/dataset/truebones/zoo/truebones_processed/cond.npy")
OUT = Path("data/truebones/reference")


def to_xyzw(quaternions) -> np.ndarray:
    """Holden stores (w, x, y, z); the project uses (x, y, z, w)."""
    qs = np.asarray(quaternions.qs if hasattr(quaternions, "qs") else quaternions)
    return np.concatenate([qs[..., 1:], qs[..., :1]], axis=-1)


def main() -> None:
    cond = np.load(COND, allow_pickle=True).item()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(cond)} skeletons in cond.npy")

    for name, entry in sorted(cond.items()):
        payload = {
            "joints_names": np.array(list(entry["joints_names"]), dtype=object),
            "parents": np.asarray(entry["parents"], dtype=np.int64),
            "offsets": np.asarray(entry["offsets"], dtype=np.float64),
            "tpos_first_frame": np.asarray(entry["tpos_first_frame"], dtype=np.float64),
            "mean": np.asarray(entry["mean"], dtype=np.float64),
            "std": np.asarray(entry["std"], dtype=np.float64),
            "joint_relations": np.asarray(entry["joint_relations"], dtype=np.int64),
            "joints_graph_dist": np.asarray(entry["joints_graph_dist"], dtype=np.int64),
            "foot_indices": np.asarray(list(entry["foot_indices"]), dtype=np.int64),
            "scale_factor": np.float64(entry["scale_factor"]),
            "ground_height": np.float64(entry["ground_height"]),
            "root_pose_init_xz": np.asarray(entry["root_pose_init_xz"], dtype=np.float64),
            "tpos_rots_xyzw": to_xyzw(entry["tpos_rots"]).astype(np.float64),
        }
        np.savez_compressed(OUT / f"{name}.npz", **payload)

    print(f"wrote {len(cond)} files to {OUT}")

    # Report the seven the test suite uses, as a sanity check.
    for name in ("BrownBear", "Coyote", "Crab", "Flamingo", "Goat", "Scorpion", "Skunk"):
        entry = cond[name]
        rots = to_xyzw(entry["tpos_rots"])
        norms = np.linalg.norm(rots, axis=-1)
        print(
            f"  {name:10s} joints={len(entry['parents']):3d} "
            f"tpose_frames={rots.shape[0]:4d} "
            f"unit_quats={bool(np.allclose(norms, 1.0, atol=1e-6))} "
            f"feat={entry['tpos_first_frame'].shape}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
