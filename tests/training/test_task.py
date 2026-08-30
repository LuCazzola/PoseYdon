"""The task must work identically for a feature-space and a latent model."""

import pytest
import torch
from torch import nn

from poseydon.core.batch import Cond, Masks, MotionBatch, WindowInfo
from poseydon.core.spec import FeatureSpec
from poseydon.losses import LOSSES, NORM_STATS
from poseydon.models.base import Denoiser, LatentDenoiser, Prediction
from poseydon.process import FlowMatching, GaussianDiffusion
from poseydon.training.task import MotionTask

SPEC = FeatureSpec((("ric_pos", 3), ("rot6d", 6), ("local_vel", 3), ("foot_contact", 1)))


def make_batch(batch=2, joints=3, frames=5, cond=None):
    x = torch.zeros(batch, joints, SPEC.dim, frames)
    x[:, :, 3:9, :] = torch.tensor([1.0, 0, 0, 0, 1.0, 0])[None, None, :, None]
    payloads = {
        NORM_STATS: {
            "mean": torch.zeros(batch, joints, SPEC.dim),
            "std": torch.ones(batch, joints, SPEC.dim),
        }
    }
    payloads.update(cond or {})
    return MotionBatch(
        x=x,
        spec=SPEC,
        masks=Masks(
            frames=torch.ones(batch, frames, dtype=torch.bool),
            joints=torch.ones(batch, joints, dtype=torch.bool),
        ),
        window=WindowInfo(
            torch.zeros(batch, dtype=torch.long), torch.full((batch,), frames, dtype=torch.long)
        ),
        cond=Cond(payloads),
    )


class TinyDenoiser(Denoiser):
    """A one-layer stand-in, enough to exercise the wiring."""

    requires = ()

    def __init__(self, dim=SPEC.dim):
        super().__init__()
        self.net = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, z_t, t, cond):
        return Prediction(out=self.net(z_t.transpose(1, 2)).transpose(1, 2))


class TinyLatentDenoiser(LatentDenoiser):
    """Halves the feature width, so the latent space is genuinely different."""

    def __init__(self, dim=SPEC.dim, latent=4):
        super().__init__()
        self.encoder = nn.Conv2d(dim, latent, kernel_size=1)
        self.decoder = nn.Conv2d(latent, dim, kernel_size=1)
        self.net = nn.Conv2d(latent, latent, kernel_size=1)
        self.latent = latent

    def encode(self, x0, cond):
        return self.encoder(x0.transpose(1, 2)).transpose(1, 2)

    def decode(self, z0, cond):
        return self.decoder(z0.transpose(1, 2)).transpose(1, 2)

    def forward(self, z_t, t, cond):
        out = self.net(z_t.transpose(1, 2)).transpose(1, 2)
        return Prediction(out=out, aux={"mu": out.mean(dim=(1, 3)), "logvar": torch.zeros_like(out.mean(dim=(1, 3)))})


def make_task(model, process, names=("simple",)):
    losses = [(name, 1.0, LOSSES.get(name)()) for name in names]
    return MotionTask(model=model, process=process, losses=losses)


@pytest.mark.parametrize(
    "process", [GaussianDiffusion(), GaussianDiffusion(parameterization="eps"), FlowMatching()]
)
def test_feature_space_model_trains_under_every_process(process):
    torch.manual_seed(0)
    task = make_task(TinyDenoiser(), process)
    batch = make_batch()
    task.setup_checks(batch)

    losses = task.compute_losses(batch)
    assert losses["total"].isfinite()
    losses["total"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in task.parameters())


def test_latent_model_uses_the_identical_task_body():
    # No branch on model kind anywhere in the task: prepare/restore do the work.
    torch.manual_seed(0)
    task = make_task(TinyLatentDenoiser(), GaussianDiffusion())
    batch = make_batch()
    task.setup_checks(batch)
    assert task.compute_losses(batch)["total"].isfinite()


def test_latent_model_corrupts_in_latent_space_not_feature_space():
    model = TinyLatentDenoiser()
    batch = make_batch()
    assert model.prepare(batch).shape[2] == model.latent
    assert model.prepare(batch).shape[2] != batch.x.shape[2]


def test_feature_space_model_prepares_the_batch_unchanged():
    batch = make_batch()
    torch.testing.assert_close(TinyDenoiser().prepare(batch), batch.x)


def test_structural_losses_combine_with_the_simple_term():
    torch.manual_seed(0)
    task = make_task(TinyDenoiser(), GaussianDiffusion(), ("simple", "geodesic", "footskate"))
    batch = make_batch()
    task.setup_checks(batch)

    losses = task.compute_losses(batch)
    assert set(losses) == {"simple", "geodesic", "footskate", "total"}
    assert all(value.isfinite() for value in losses.values())


def test_setup_rejects_a_loss_whose_block_is_absent():
    spare = FeatureSpec((("ric_pos", 3),))
    batch = MotionBatch(
        x=torch.zeros(1, 2, 3, 4),
        spec=spare,
        masks=Masks(
            frames=torch.ones(1, 4, dtype=torch.bool), joints=torch.ones(1, 2, dtype=torch.bool)
        ),
        window=WindowInfo(torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long)),
        cond=Cond({}),
    )
    task = make_task(TinyDenoiser(dim=3), GaussianDiffusion(), ("geodesic",))
    with pytest.raises(ValueError, match="rot6d"):
        task.setup_checks(batch)


def test_setup_rejects_a_model_missing_a_required_conditioner():
    class NeedsTPose(TinyDenoiser):
        requires = ("tpose", "topology")

    task = make_task(NeedsTPose(), GaussianDiffusion())
    with pytest.raises(ValueError, match="topology"):
        task.setup_checks(make_batch())


def test_setup_accepts_a_model_whose_conditioners_are_present():
    class NeedsTPose(TinyDenoiser):
        requires = ("tpose",)

    task = make_task(NeedsTPose(), GaussianDiffusion())
    task.setup_checks(make_batch(cond={"tpose": torch.zeros(1)}))


def test_kl_reaches_the_models_auxiliary_outputs():
    torch.manual_seed(0)
    task = make_task(TinyLatentDenoiser(), GaussianDiffusion(), ("simple", "kl"))
    batch = make_batch()
    task.setup_checks(batch)
    assert task.compute_losses(batch)["kl"].isfinite()


def test_kl_refuses_a_model_without_a_bottleneck():
    task = make_task(TinyDenoiser(), GaussianDiffusion(), ("kl",))
    with pytest.raises(ValueError, match="bottleneck"):
        task.compute_losses(make_batch())


def test_loss_is_deterministic_given_the_noise():
    torch.manual_seed(0)
    task = make_task(TinyDenoiser(), GaussianDiffusion(), ("simple",))
    batch = make_batch()
    noise = torch.randn_like(batch.x)

    torch.manual_seed(1)
    first = task.compute_losses(batch, noise=noise)["total"]
    torch.manual_seed(1)
    second = task.compute_losses(batch, noise=noise)["total"]
    torch.testing.assert_close(first, second)
