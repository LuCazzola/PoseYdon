"""MoDiffAE: the semantic autoencoder, and blending without model changes."""

from pathlib import Path

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
from poseydon.process import GaussianDiffusion
from poseydon.training.task import MotionTask
from tests.ingest.manifest_helper import MANIFEST_DIR

REFERENCE_SAVE = Path(__file__).resolve().parents[2] / "external/neural_motion_blending/save"


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


def test_semantic_latent_is_one_vector_per_frame(batch):
    # Fixed width regardless of joint count: the point of pooling. The leading
    # entry is the rest-pose frame the input process prepends.
    torch.manual_seed(0)
    model = small_model(batch)
    z_sem, _ = model.encode(batch.x, with_clean(batch), batch.masks)
    assert z_sem.shape == (batch.n_frames + 1, len(batch), 1, model.d_model)


def test_latent_depends_on_the_motion(batch):
    torch.manual_seed(0)
    model = small_model(batch).eval()
    with torch.no_grad():
        a, _ = model.encode(batch.x, with_clean(batch), batch.masks)
        b, _ = model.encode(torch.zeros_like(batch.x), with_clean(batch), batch.masks)
    assert not torch.allclose(a, b)


def test_supplied_latent_bypasses_the_encoder(batch):
    # What makes blending and cross-skeleton transfer possible without touching
    # the model: hand it a z_sem and the encoder is never consulted.
    torch.manual_seed(0)
    model = small_model(batch).eval()
    chosen = torch.zeros(batch.n_frames + 1, len(batch), 1, model.d_model)
    cond = Cond({**with_clean(batch).payloads, Z_SEM: chosen})
    with torch.no_grad():
        out = model(batch.x, torch.zeros(len(batch), dtype=torch.long), cond, batch.masks)
    torch.testing.assert_close(out.aux[Z_SEM], chosen)


def test_latent_can_come_from_a_different_skeleton(batch):
    # Cross-skeleton transfer in miniature: encode one item, decode onto another.
    torch.manual_seed(0)
    model = small_model(batch).eval()
    with torch.no_grad():
        source, _ = model.encode(batch.x, with_clean(batch), batch.masks)
        borrowed = source[:, :1].expand(-1, len(batch), -1, -1).contiguous()
        out = model(
            batch.x,
            torch.zeros(len(batch), dtype=torch.long),
            Cond({**with_clean(batch).payloads, Z_SEM: borrowed}),
            batch.masks,
        )
    assert torch.isfinite(out.out).all()


def test_kl_bottleneck_exposes_its_posterior(batch):
    torch.manual_seed(0)
    model = small_model(batch, kl_bottleneck=True)
    out = model(batch.x, torch.zeros(len(batch), dtype=torch.long), with_clean(batch), batch.masks)
    assert {"mu", "logvar"} <= set(out.aux)


def test_without_the_bottleneck_kl_is_unavailable(batch):
    torch.manual_seed(0)
    model = small_model(batch)
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


def test_published_checkpoint_loads_without_remapping():
    # The port matches the reference's parameter names exactly, so its published
    # weights load with strict=True and no conversion step.
    checkpoint = (
        REFERENCE_SAVE / "truebones_globpool" / "model000449998.pt"
    )
    if not checkpoint.is_file():
        pytest.skip("reference checkpoints not present")

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    model = MoDiffAE(feature_dim=13, d_model=128, ff_size=1024, projection_head_depth=2)
    model.load_state_dict(state, strict=True)
    assert len(state) == 292
