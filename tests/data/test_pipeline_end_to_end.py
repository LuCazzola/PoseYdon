"""Ingest a real corpus, read it, batch it, and take a training step.

This is the test that says the abstractions hold up against real data: seven
skeletons with joint counts from 31 to 63, different clip lengths, all flowing
through one code path into one loss.
"""

import numpy as np
import pytest
import torch

from poseydon.augment.topology import DropEndEffector
from poseydon.core.batch import MotionBatch
from poseydon.data.collate import collate
from poseydon.data.dataset import MotionDataset
from poseydon.data.window import FullClip, RandomCrop, SlidingWindow
from poseydon.ingest.pipeline import ingest_corpus
from poseydon.losses import LOSSES
from poseydon.process import FlowMatching, GaussianDiffusion
from poseydon.training.task import MotionTask
from tests.ingest.manifest_helper import MANIFEST_DIR
from tests.training.test_task import TinyDenoiser


@pytest.fixture(scope="module")
def corpus(truebones_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("corpus")
    result = ingest_corpus(sorted(truebones_dir.glob("*.bvh")), MANIFEST_DIR, out)
    assert len(result.index) == 7, result.skipped
    return result.index, out


def make_dataset(corpus, **kwargs):
    index, out = corpus
    kwargs.setdefault("conditioners", ("topology", "norm_stats"))
    return MotionDataset(index, out, MANIFEST_DIR, **kwargs)


def test_dataset_covers_every_clip(corpus):
    assert len(make_dataset(corpus, window=FullClip())) == 7


def test_sliding_window_yields_more_samples_than_clips(corpus):
    dataset = make_dataset(corpus, window=SlidingWindow(length=40, stride=20))
    assert len(dataset) > 7


def test_items_are_normalized_to_the_active_spec(corpus):
    dataset = make_dataset(corpus, features=("rot6d", "local_vel"))
    item = dataset[0]
    assert item.spec.names == ("rot6d", "local_vel")
    assert item.features.shape[-1] == 9


def test_collate_pads_to_the_batch_maximum_not_a_global_constant(corpus):
    # Goat has 31 joints and Scorpion 63; a batch of both must be 63 wide, and
    # never the reference's global 143.
    dataset = make_dataset(corpus, window=FullClip())
    items = [dataset[i] for i in range(len(dataset))]
    batch = collate(items)

    joint_counts = sorted({int(n) for n in batch.masks.n_joints})
    assert batch.n_joints == max(joint_counts) == 63
    assert min(joint_counts) == 31


def test_padded_joints_and_frames_are_masked_out(corpus):
    dataset = make_dataset(corpus, window=FullClip())
    batch = collate([dataset[i] for i in range(len(dataset))])

    for i in range(len(batch)):
        joints = int(batch.masks.n_joints[i])
        frames = int(batch.masks.lengths[i])
        assert torch.all(batch.x[i, joints:] == 0)
        assert torch.all(batch.x[i, :, :, frames:] == 0)


def test_window_records_its_offset_into_the_real_animation(corpus):
    dataset = make_dataset(corpus, window=RandomCrop(length=40))
    batch = collate([dataset[i] for i in range(len(dataset))])
    assert torch.all(batch.window.start >= 0)
    assert torch.all(batch.window.start + batch.masks.lengths <= batch.window.source_length)


def test_conditioners_supply_only_what_was_declared(corpus):
    dataset = make_dataset(corpus, conditioners=("topology",), window=FullClip())
    batch = collate([dataset[i] for i in range(3)])
    assert set(batch.cond.payloads) == {"topology"}
    assert "norm_stats" not in batch.cond


def test_topology_padding_uses_a_sentinel_parent(corpus):
    dataset = make_dataset(corpus, conditioners=("topology",), window=FullClip())
    batch = collate([dataset[i] for i in range(len(dataset))])
    parents = batch.cond["topology"]["parents"]
    for i in range(len(batch)):
        joints = int(batch.masks.n_joints[i])
        assert torch.all(parents[i, joints:] == -1)
        assert parents[i, 0] == -1  # the real root


@pytest.mark.parametrize("process", [GaussianDiffusion(), FlowMatching()])
def test_a_real_batch_trains(corpus, process):
    torch.manual_seed(0)
    dataset = make_dataset(corpus, window=RandomCrop(length=32))
    batch = collate([dataset[i] for i in range(4)])
    assert isinstance(batch, MotionBatch)

    task = MotionTask(
        model=TinyDenoiser(dim=batch.spec.dim),
        process=process,
        losses=[(n, 1.0, LOSSES.get(n)()) for n in ("simple", "geodesic", "footskate")],
    )
    task.setup_checks(batch)

    losses = task.compute_losses(batch)
    assert all(v.isfinite() for v in losses.values())
    losses["total"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in task.parameters())


def test_empty_augmentation_list_matches_unaugmented_output(corpus):
    plain = make_dataset(corpus, window=FullClip())
    explicit_empty = make_dataset(corpus, window=FullClip(), augmentations=())

    np.testing.assert_array_equal(plain[0].features, explicit_empty[0].features)


def test_structural_augmentation_keeps_conditioning_consistent_with_features(corpus):
    dataset = make_dataset(
        corpus,
        window=FullClip(),
        augmentations=(DropEndEffector(p=1.0),),
        conditioners=("topology", "tpose", "norm_stats"),
    )
    batch = collate([dataset[i] for i in range(len(dataset))])

    for i in range(len(batch)):
        joints = int(batch.masks.n_joints[i])
        parents = batch.cond["topology"]["parents"][i]
        mean = batch.cond["norm_stats"]["mean"][i]
        tpose = batch.cond["tpose"][i]

        assert torch.all(parents[joints:] == -1)
        assert torch.all(mean[joints:] == 0)
        assert torch.all(tpose[joints:] == 0)
        # the surviving joint count must match across the feature tensor and
        # every declared conditioner -- this is the property the whole design
        # exists to guarantee.
        assert batch.x[i, joints:].abs().sum() == 0


def test_changing_features_needs_no_reingest(corpus):
    # The point of storing canonical animations: two representations from the
    # same files, no preprocessing in between.
    full = make_dataset(corpus, window=FullClip())
    rotations_only = make_dataset(corpus, features=("rot6d",), window=FullClip())

    assert full[0].features.shape[-1] == 13
    assert rotations_only[0].features.shape[-1] == 6
    assert len(full) == len(rotations_only)


def test_normalized_features_are_finite(corpus):
    dataset = make_dataset(corpus, window=FullClip())
    for i in range(len(dataset)):
        assert np.isfinite(dataset[i].features).all()
