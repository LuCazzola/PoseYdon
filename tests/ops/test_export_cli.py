from pathlib import Path

from poseydon.cli import main
from poseydon.core.animation import Animation
from poseydon.core.rotations import QUAT_IDENTITY
import numpy as np

def _input(path: Path):
    Animation(
        rotations=np.tile(QUAT_IDENTITY, (2, 2, 1)),
        translations=np.zeros((2, 2, 3)),
        offsets=np.array([[0, 0, 0], [0, 1, 0.]], dtype=float),
        parents=np.array([-1, 0], dtype=np.int32),
        names=("Root", "Leaf"), fps=30.0,
    ).save(path)

def test_export_cli_accepts_explicit_bare_skeleton(tmp_path):
    source, output = tmp_path / "input.anim.npz", tmp_path / "bare.bvh"
    _input(source)
    assert main(["export", str(source), "--rig", "Fixture", "--bare-skeleton", "--format", "bvh", "--out", str(output)]) == 0
    assert output.is_file()
