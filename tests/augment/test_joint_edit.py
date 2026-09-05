import numpy as np

from poseydon.augment.joint_edit import JointEdit
from poseydon.core.spec import FeatureSpec
from poseydon.data.normalize import Normalizer


def test_identity_maps_every_joint_to_itself():
    edit = JointEdit.identity(4)
    assert edit.source_of == (0, 1, 2, 3)


def test_compose_reindexes_through_two_edits():
    # A 4-joint skeleton drops joint 1, yielding a 3-joint skeleton indexed
    # (0, 2, 3) into the original. That skeleton then drops its own joint 0
    # (original joint 0), yielding a 2-joint skeleton indexed (1, 2) into the
    # 3-joint intermediate.
    drop_middle = JointEdit(source_of=(0, 2, 3))
    then_drop_first = JointEdit(source_of=(1, 2))
    combined = then_drop_first.compose(drop_middle)
    assert combined.source_of == (2, 3)


def test_transport_gathers_normalizer_rows_by_source():
    spec = FeatureSpec((("x", 2),))
    normalizer = Normalizer(
        mean=np.arange(8, dtype=np.float64).reshape(4, 2),
        std=np.ones((4, 2)),
        spec=spec,
    )
    edit = JointEdit(source_of=(0, 2, 3))

    transported = edit.transport(normalizer)

    np.testing.assert_array_equal(transported.mean, normalizer.mean[[0, 2, 3]])
    np.testing.assert_array_equal(transported.std, normalizer.std[[0, 2, 3]])
    assert transported.spec is spec


def test_transport_row_gathers_a_single_array_by_source():
    edit = JointEdit(source_of=(2, 0))
    row = np.array([[1.0], [2.0], [3.0]])

    result = edit.transport_row(row)

    np.testing.assert_array_equal(result, np.array([[3.0], [1.0]]))
