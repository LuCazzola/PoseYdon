import numpy as np
import pytest

from poseydon.build.correspondence import (
    CorrespondenceError,
    SkeletonGeometry,
    build_correspondence,
)


MODEL = SkeletonGeometry(
    names=("Hips", "Spine", "Arm"), parents=(-1, 0, 1),
    bind_positions=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
)
PRESENTATION = SkeletonGeometry(
    names=("mixamorig:Hips", "Spine", "Arm"), parents=(-1, 0, 1),
    bind_positions=np.array([[2.0, 0.0, 3.0], [2.0, 2.0, 3.0], [4.0, 2.0, 3.0]]),
)


def test_correspondence_matches_alias_and_normalizes_bind_scale_translation():
    match = build_correspondence(
        MODEL, PRESENTATION, aliases={"mixamorig:Hips": "Hips"}
    )
    assert match.presentation_for("Hips") == "mixamorig:Hips"
    assert match.presentation_for("Spine") == "Spine"
    assert match.presentation_for("Arm") == "Arm"
    assert match.max_bind_residual < 1e-12


def test_duplicate_alias_target_is_rejected():
    presentation = SkeletonGeometry(
        names=("A", "B", "Arm"), parents=(-1, 0, 1),
        bind_positions=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
    )
    with pytest.raises(CorrespondenceError, match="ambiguous"):
        build_correspondence(MODEL, presentation, aliases={"A": "Hips", "B": "Hips"})


def test_bind_disagreement_over_threshold_is_rejected():
    bent = SkeletonGeometry(
        names=("Hips", "Spine", "Arm"), parents=(-1, 0, 1),
        bind_positions=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [3.0, 1.0, 0.0]]),
    )
    with pytest.raises(CorrespondenceError, match="bind residual"):
        build_correspondence(MODEL, bent, max_residual=0.01)
