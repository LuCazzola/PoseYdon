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


def test_export_rejects_a_representation_with_no_such_format(capsys, tmp_path):
    """`presentation` is FBX-only; asking for it as BVH must say so, not crash."""
    import numpy as np

    from poseydon.cli import main
    from poseydon.core.animation import Animation
    from poseydon.io.bvh import BVH

    source = tmp_path / "clip.bvh"
    BVH.from_animation(
        Animation(
            rotations=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (1, 2, 1)),
            translations=np.tile(np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), (1, 1, 1)),
            offsets=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            parents=np.array([-1, 0], dtype=np.int32),
            names=("root", "leaf"),
            fps=30.0,
        )
    ).write(source)

    code = main(
        [
            "export",
            str(source),
            "--out", str(tmp_path / "out.bvh"),
            "--rig", "Nonexistent",
            "--representation", "presentation",
            "--format", "bvh",
        ]
    )
    assert code != 0
    assert "presentation" in capsys.readouterr().err
