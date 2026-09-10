"""`run_retarget`: the one path the CLI and the training callback both take.

Real corpus, stub model. What is under test is the plumbing -- resolve a clip,
condition on ANOTHER rig, sample at the content's length, reconstruct the way
the recipe says, write three loadable files -- not the quality of the motion,
which no untrained model has anyway.

Skips without the stage-2 corpus, like the rest of the read-path suite: these
assertions are about real rigs with real joint counts, and a synthetic pair
would be a second, divergent definition of what a rig is.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from poseydon.build.index import CorpusIndex
from poseydon.core.animation import RigidBodyAnimation
from poseydon.data.dataset import MotionDataset
from poseydon.data.window import RandomCrop
from poseydon.io.bvh import BVH
from poseydon.models.base import Denoiser, Prediction
from poseydon.ops.retarget_run import RetargetResult, run_retarget
from poseydon.process.gaussian import GaussianDiffusion
from poseydon.sampling.diffusion import DDPM
from poseydon.training import recipe

CORPUS = Path("data/truebones")
FEATURES = ["ric_pos", "rot6d", "local_vel", "foot_contact"]

# A short content clip on purpose: every assertion here is about topology and
# length, and a 25-frame clip costs seconds where a 200-frame one costs minutes
# of IK. The pair is still cross-topology in the direction that matters -- a
# 60-odd joint milliped decoded onto a bird.
CONTENT_RIG, CONTENT_ACTION, TARGET_RIG = "Scorpion", "agitated", "Flamingo"

# Deliberately shorter than any clip: the output length must follow the CONTENT
# CLIP, not whatever window the dataset was built for training with.
WINDOW = 8

pytestmark = pytest.mark.skipif(
    not (CORPUS / "index.jsonl").is_file(),
    reason="stage-2 corpus absent; run scripts/build_features.py",
)


class _Pooled(Denoiser):
    """MoDiffAE's two shape contracts, without its cost.

    `encode` pools over joints, so the latent is joint-independent -- that is
    the whole mechanism retargeting rests on. `forward` predicts x0 and is a
    near-identity, which keeps the sampled trajectory finite without pretending
    to be a trained model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.ones(1))

    def encode(self, clean, cond, masks=None, temporal_valid=None):
        pooled = clean.mean(dim=(1, 2))  # (B, T): joints and channels pooled away
        return pooled.permute(1, 0)[:, :, None, None].repeat(1, 1, 1, 8), {}

    def forward(self, z_t, t, cond, masks=None):
        return Prediction(out=z_t * self.gain, aux={})


@pytest.fixture(scope="module")
def dataset() -> MotionDataset:
    return MotionDataset(
        index=CorpusIndex.load(CORPUS / "index.jsonl"),
        root=CORPUS,
        manifest_dir=CORPUS / "rigs",
        features=FEATURES,
        window=RandomCrop(length=WINDOW),
        conditioners=["topology", "tpose", "norm_stats"],
        split="train",
        seed=0,
    )


@pytest.fixture(scope="module")
def content(dataset: MotionDataset):
    matching = [
        r
        for r in dataset.records
        if r.skeleton == CONTENT_RIG and r.action == CONTENT_ACTION
    ]
    if not matching:
        pytest.skip(f"no clip {CONTENT_RIG}/{CONTENT_ACTION} in the index")
    return matching[0]


def _run(dataset, content, out_dir, **kwargs) -> RetargetResult:
    return run_retarget(
        model=_Pooled().eval(),
        process=GaussianDiffusion(num_steps=2),
        sampler=DDPM(),
        dataset=dataset,
        content=content,
        target_rig=TARGET_RIG,
        out_dir=out_dir,
        generator=torch.Generator().manual_seed(0),
        **kwargs,
    )


@pytest.fixture(scope="module")
def result(dataset, content, tmp_path_factory) -> RetargetResult:
    """One run, shared: the IK solve is the expensive part of this file."""
    return _run(dataset, content, tmp_path_factory.mktemp("retarget"))


def test_it_writes_all_three_files_named_for_the_pair(result):
    """Nothing downstream parses these names, but a human reading a validation
    directory does, and the callback writes nine of them per firing.
    """
    stem = f"{CONTENT_RIG}_{CONTENT_ACTION}__to__{TARGET_RIG}"
    for path, suffix in ((result.npz, ".npz"), (result.bvh, ".bvh"), (result.mp4, ".mp4")):
        assert path.name == stem + suffix
        assert path.is_file(), f"{path} was reported but not written"
        assert path.stat().st_size > 0


def test_the_bvh_parses_back_on_the_TARGET_rig(result, dataset):
    """The assertion the feature exists for. A BVH written on the CONTENT's
    joints would still load; it would simply not be a retarget.
    """
    written = BVH.read(result.bvh)
    assert written.names == result.target_anim.names
    np.testing.assert_array_equal(written.parents, result.target_anim.parents)
    assert written.names != result.content_anim.names, "that is the content rig"
    assert len(written.names) != len(result.content_anim.names)


def test_the_output_is_the_content_clips_own_length(result, dataset, content):
    """Not the training window's, and not a configured `n_frames`."""
    expected = content.n_frames - dataset._frame_cost
    assert expected > WINDOW, "pick a longer content clip; this proves nothing"
    assert result.solved_positions.shape[0] == expected
    assert result.predicted_positions.shape[0] == expected
    assert BVH.read(result.bvh).rotations.shape[0] == expected


def test_the_raw_and_the_solved_positions_are_both_kept(result):
    """Task 4 measures bone-length drift on the PRE-IK positions: on solved
    output it is identically zero by construction, which measures the solver's
    parameterization rather than the model.
    """
    assert result.predicted_positions.shape == result.solved_positions.shape
    assert not np.array_equal(result.predicted_positions, result.solved_positions), (
        "pre-IK and post-IK positions are identical, so one of them is the other"
    )


def test_it_carries_the_arrays_the_metrics_need(result, dataset):
    assert result.latent is not None and result.latent.ndim == 4
    # The latent's frame axis follows the CONTENT clip, which is why the output
    # is the content's length: the two have to line up token for token. Real
    # `MoDiffAE.encode` prepends the rig's rest frame, so against that model the
    # axis is one longer than the motion -- never the TARGET clip's length.
    frames = result.solved_positions.shape[0]
    assert result.latent.shape[0] in (frames, frames + 1)
    assert isinstance(result.target_anim, RigidBodyAnimation)
    assert isinstance(result.content_anim, RigidBodyAnimation)
    assert result.content_anim.names == dataset._anim(
        next(r for r in dataset.records if r.clip_id == f"{CONTENT_RIG}__{CONTENT_ACTION}")
    ).names


def test_the_npz_holds_what_a_later_step_would_be_diffed_against(result):
    with np.load(result.npz, allow_pickle=False) as data:
        np.testing.assert_allclose(data["solved_positions"], result.solved_positions)
        np.testing.assert_allclose(data["predicted_positions"], result.predicted_positions)
        np.testing.assert_array_equal(data["parents"], result.target_anim.parents)
        assert list(data["names"]) == list(result.target_anim.names)


def test_the_reconstruction_is_the_one_the_recipe_records(result):
    assert recipe.DEFAULT_RECONSTRUCT == "positions_ik"
    assert result.reconstruct == recipe.DEFAULT_RECONSTRUCT


def test_changing_the_recipe_changes_what_is_run(dataset, content, tmp_path, monkeypatch):
    """The method is read from the recipe, not hardcoded beside it -- otherwise
    `sample`, `retarget` and the callback are free to disagree silently.
    """
    monkeypatch.setattr(recipe, "DEFAULT_RECONSTRUCT", "fk")
    changed = _run(dataset, content, tmp_path)
    assert changed.reconstruct == "fk"
    # `fk` is forward kinematics from the rotation channels, so its positions
    # are rigid by construction and cannot be the IK solve's.
    assert not np.array_equal(changed.solved_positions, changed.predicted_positions)


def test_a_reconstruction_that_cannot_write_a_bvh_is_refused(dataset, content, tmp_path):
    """`positions` yields no rotations. Writing two of the three files and
    calling it a validation artifact is worse than not running.
    """
    with pytest.raises(ValueError, match="positions"):
        _run(dataset, content, tmp_path, reconstruct="positions")
