import pytest
import torch

from poseydon.core.batch import Cond, Masks, MotionBatch, WindowInfo
from poseydon.core.spec import FeatureSpec

SPEC = FeatureSpec((("ric_pos", 3), ("rot6d", 6), ("local_vel", 3), ("foot_contact", 1)))


def make_batch(batch=2, joints=4, frames=5, real_frames=None, real_joints=None):
    frame_mask = torch.ones(batch, frames, dtype=torch.bool)
    joint_mask = torch.ones(batch, joints, dtype=torch.bool)
    if real_frames is not None:
        frame_mask[:, real_frames:] = False
    if real_joints is not None:
        joint_mask[:, real_joints:] = False
    return MotionBatch(
        x=torch.zeros(batch, joints, SPEC.dim, frames),
        spec=SPEC,
        masks=Masks(frames=frame_mask, joints=joint_mask),
        window=WindowInfo(
            start=torch.zeros(batch, dtype=torch.long),
            source_length=torch.full((batch,), frames, dtype=torch.long),
        ),
        cond=Cond({}),
    )


def test_block_selects_the_named_columns():
    batch = make_batch()
    assert batch.block("rot6d").shape == (2, 4, 6, 5)


def test_block_rejects_a_missing_name():
    with pytest.raises(KeyError, match="angular_vel"):
        make_batch().block("angular_vel")


def test_width_must_match_the_spec():
    with pytest.raises(ValueError, match="width"):
        MotionBatch(
            x=torch.zeros(1, 2, 99, 3),
            spec=SPEC,
            masks=Masks(
                frames=torch.ones(1, 3, dtype=torch.bool),
                joints=torch.ones(1, 2, dtype=torch.bool),
            ),
            window=WindowInfo(torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long)),
            cond=Cond({}),
        )


def test_mask_shapes_must_match_x():
    with pytest.raises(ValueError, match="frame mask"):
        MotionBatch(
            x=torch.zeros(1, 2, SPEC.dim, 3),
            spec=SPEC,
            masks=Masks(
                frames=torch.ones(1, 99, dtype=torch.bool),
                joints=torch.ones(1, 2, dtype=torch.bool),
            ),
            window=WindowInfo(torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long)),
            cond=Cond({}),
        )


def test_masks_must_be_boolean():
    with pytest.raises(TypeError, match="bool"):
        Masks(frames=torch.ones(1, 3), joints=torch.ones(1, 2, dtype=torch.bool))


def test_lengths_and_joint_counts_come_from_the_masks():
    batch = make_batch(frames=5, joints=4, real_frames=3, real_joints=2)
    assert batch.masks.lengths.tolist() == [3, 3]
    assert batch.masks.n_joints.tolist() == [2, 2]


def test_pairwise_masks_are_outer_products():
    batch = make_batch(frames=4, real_frames=2)
    temporal = batch.masks.temporal()
    assert temporal.shape == (2, 1, 4, 4)
    assert temporal[0, 0, 0, 0] and not temporal[0, 0, 0, 3]


def test_missing_conditioner_names_what_is_present():
    cond = Cond({"tpose": torch.zeros(1)})
    with pytest.raises(KeyError) as excinfo:
        cond["joint_names"]
    assert "joint_names" in str(excinfo.value)
    assert "tpose" in str(excinfo.value)


def test_to_moves_every_tensor():
    batch = make_batch()
    moved = batch.to("cpu")
    assert moved.x.device.type == "cpu"
    assert moved.masks.frames.device.type == "cpu"
    assert moved.window.start.device.type == "cpu"
