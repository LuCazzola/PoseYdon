"""One verifier case: the stage files of a built round trip, and the frames to judge."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Case:
    """``files`` are the files under test by name; ``clip_fbx`` is the clip's own vendor
    file (joint motion) and ``skin_fbx`` the rig's reference (mesh, skin, bind). Frames
    are positions in the file under test and in ``clip_fbx``."""

    data_name: str
    rig: str
    action: str
    files: dict[str, Path]
    clip_fbx: Path
    skin_fbx: Path
    frames: np.ndarray
    source_frames: np.ndarray


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def commit(root: str | Path = ".") -> str:
    """HEAD of the checkout, read from `.git` (the containers have no git binary)."""
    git = Path(root) / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        if (git / ref).is_file():
            return (git / ref).read_text().strip()
        packed = (git / "packed-refs").read_text().splitlines()
        return next(line.split()[0] for line in packed if line.endswith(" " + ref))
    except (OSError, StopIteration):
        return "unknown"


def sample_frames(world: np.ndarray, n: int) -> np.ndarray:
    """``n`` evenly spaced frames plus the frame of largest joint speed, sorted, unique.
    ``world`` is ``(F, N, 4, 4)``."""
    count = len(world)
    picked = set(np.linspace(0, count - 1, min(n, count)).round().astype(int))
    if count > 1:
        speed = np.linalg.norm(np.diff(world[..., :3, 3], axis=0), axis=-1).max(axis=1)
        picked.add(int(speed.argmax()) + 1)
    return np.array(sorted(picked), dtype=int)


def load_case(run_dir: str | Path) -> Case:
    meta = json.loads((Path(run_dir) / "case.json").read_text())
    return Case(meta["data"], meta["rig"], meta["action"],
                {k: Path(v) for k, v in meta["files"].items()}, Path(meta["clip_fbx"]), Path(meta["skin_fbx"]),
                np.array(meta["frames"], dtype=int), np.array(meta["source_frames"], dtype=int))
