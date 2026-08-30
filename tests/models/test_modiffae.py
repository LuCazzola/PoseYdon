"""MoDiffAE: the semantic autoencoder, and blending without model changes."""

import pytest
import torch

from poseydon.core.batch import Cond
from poseydon.data.collate import collate
from poseydon.data.dataset import MotionDataset
from poseydon.data.window import RandomCrop
from poseydon.ingest.pipeline import ingest_corpus
from poseydon.losses import LOSSES
from poseydon.models import CLEAN_MOTION, MODELS, MoDiffAE
from poseydon.models.modiffae import Z_SEM
from poseydon.ops import Blend
from poseydon.process import GaussianDiffusion
from poseydon.training.task import MotionTask
from tests.ingest.manifest_helper import MANIFEST_DIR


@pytest.fixture(scope="module")
def batch(truebones_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("modiffae_corpus")
    result = ingest_corpus(sorted(truebones_dir.glob("*.bvh")), MANIFEST_DIR, out)
    assert len(result.index) == 7, result.skipped
    dataset = MotionDataset(
        result.index,
        out,
        MANIFEST_DIR,
        window=RandomCrop(length=16),
        conditioners=("topology", "tpose", "norm_stats"),
    )
    return collate([dataset[i] for i in range(4)])


def small_model(batch, **kwargs):
    return MoDiffAE(
        feature_dim=batch.spec.dim,
        d_model=32,
        n_layers_semantic=1,
        n_layers_stochastic=1,
        n_heads=4,
        ff_size=64,
        **kwargs,
    )


def with_clean(batch):
    return Cond({**batch.cond.payloads, CLEAN_MOTION: batch.x})


def test_registered_alongside_anytop():
    assert MODELS.names() == ["anytop", "modiffae"]


def test_declares_it_needs_the_clean_motion():
    assert CLEAN_MOTION in MoDiffAE.requires


def test_output_matches_input_shape(batch):
    torch.manual_seed(0)
    model = small_model(batch)
    out = model(batch.x, torch.zeros(len(batch), dtype=torch.long), with_clean(batch), batch.masks)
    assert out.out.shape == batch.x.shape


def test_semantic_latent_is_per_frame_and_skeleton_agnostic(batch):
    # Fixed width regardless of joint count: that is what the virtual joints buy.
    torch.manual_seed(0)
    model = small_model(batch)
    z_sem, _ = model.encode(batch.x, with_clean(batch), batch.masks)
    assert z_sem.shape == (len(batch), batch.n_frames, model.d_model)


def test_latent_depends_on_the_motion(batch):
    torch.manual_seed(0)
    model = small_model(batch).eval()
    with torch.no_grad():
        a, _ = model.encode(batch.x, with_clean(batch), batch.masks)
        b, _ = model.encode(torch.zeros_like(batch.x), with_clean(batch), batch.masks)
    assert not torch.allclose(a, b)


def test_supplied_latent_bypasses_the_encoder(batch):
    # This is what makes blending possible without touching the model: hand it a
    # z_sem and the encoder is never consulted.
    torch.manual_seed(0)
    model = small_model(batch).eval()
    t = torch.zeros(len(batch), dtype=torch.long)

    chosen = torch.zeros(len(batch), batch.n_frames, model.d_model)
    cond = Cond({**with_clean(batch).payloads, Z_SEM: chosen})
    with torch.no_grad():
        out = model(batch.x, t, cond, batch.masks)
    torch.testing.assert_close(out.aux[Z_SEM], chosen)


def test_blend_control_drives_the_model_without_model_changes(batch):
    torch.manual_seed(0)
    model = small_model(batch).eval()
    frames, dim = batch.n_frames, model.d_model

    reference = torch.zeros(len(batch), frames, dim)
    target = torch.ones(len(batch), frames, dim)
    control = Blend(reference, target, schedule="linear", length=frames).build_controls(
        with_clean(batch)
    )[0]

    _, blended = control.before_step(batch.x, torch.zeros(1), with_clean(batch))
    with torch.no_grad():
        out = model(batch.x, torch.zeros(len(batch), dtype=torch.long), blended, batch.masks)

    mixed = out.aux[Z_SEM]
    torch.testing.assert_close(mixed[:, 0], reference[:, 0])
    torch.testing.assert_close(mixed[:, -1], target[:, -1])


def test_kl_bottleneck_exposes_its_posterior(batch):
    torch.manual_seed(0)
    model = small_model(batch, kl_bottleneck=True)
    out = model(batch.x, torch.zeros(len(batch), dtype=torch.long), with_clean(batch), batch.masks)
    assert {"mu", "logvar"} <= set(out.aux)


def test_without_the_bottleneck_kl_is_unavailable(batch):
    torch.manual_seed(0)
    model = small_model(batch, kl_bottleneck=False)
    out = model(batch.x, torch.zeros(len(batch), dtype=torch.long), with_clean(batch), batch.masks)
    assert "mu" not in out.aux


def test_task_supplies_the_clean_motion(batch):
    # The dataset cannot produce this conditioner; only the task can.
    torch.manual_seed(0)
    task = MotionTask(
        model=small_model(batch, kl_bottleneck=True),
        process=GaussianDiffusion(),
        losses=[(n, w, LOSSES.get(n)()) for n, w in (("simple", 1.0), ("kl", 0.001))],
    )
    task.setup_checks(batch)
    losses = task.compute_losses(batch)
    assert all(v.isfinite() for v in losses.values())


def test_trains_and_improves_on_one_batch(batch):
    torch.manual_seed(0)
    task = MotionTask(
        model=small_model(batch),
        process=GaussianDiffusion(),
        losses=[("simple", 1.0, LOSSES.get("simple")())],
    )
    optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)

    torch.manual_seed(1)
    noise = torch.randn_like(batch.x)
    first = task.compute_losses(batch, noise=noise)["total"].item()
    for _ in range(30):
        optimizer.zero_grad()
        task.compute_losses(batch, noise=noise)["total"].backward()
        optimizer.step()
    last = task.compute_losses(batch, noise=noise)["total"].item()

    assert last < first, f"loss did not fall: {first:.4f} -> {last:.4f}"
