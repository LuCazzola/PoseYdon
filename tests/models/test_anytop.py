"""AnyTop against real, ragged batches."""

import pytest
import torch

from poseydon.core.batch import Cond
from poseydon.core.topology import EdgeType
from poseydon.data.collate import collate
from poseydon.data.dataset import MotionDataset
from poseydon.data.window import RandomCrop
from poseydon.ingest.pipeline import ingest_corpus
from poseydon.losses import LOSSES
from poseydon.models import MODELS, AnyTop
from poseydon.models.modules.graph_attention import GraphAttention
from poseydon.process import GaussianDiffusion
from poseydon.training.task import MotionTask
from tests.ingest.manifest_helper import MANIFEST_DIR

CONDITIONERS = ("topology", "tpose", "norm_stats")


@pytest.fixture(scope="module")
def corpus(truebones_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("anytop_corpus")
    result = ingest_corpus(sorted(truebones_dir.glob("*.bvh")), MANIFEST_DIR, out)
    assert len(result.index) == 7, result.skipped
    return result.index, out


@pytest.fixture(scope="module")
def batch(corpus):
    index, out = corpus
    dataset = MotionDataset(
        index, out, MANIFEST_DIR, window=RandomCrop(length=16), conditioners=CONDITIONERS
    )
    return collate([dataset[i] for i in range(len(dataset))])


def small_model(batch):
    return AnyTop(feature_dim=batch.spec.dim, d_model=32, n_layers=2, n_heads=4, ff_size=64)


def test_registered():
    assert MODELS.names() == ["anytop"]


def test_declares_the_conditioners_it_needs():
    assert set(AnyTop.requires) == {"topology", "tpose"}


def test_missing_conditioner_fails_before_training(batch):
    model = small_model(batch)
    with pytest.raises(ValueError, match="tpose"):
        model.check_conditioners({"topology"})


def test_output_matches_input_shape_on_a_real_ragged_batch(batch):
    torch.manual_seed(0)
    model = small_model(batch)
    t = torch.randint(0, 100, (len(batch),))
    out = model(batch.x, t, batch.cond, batch.masks).out
    assert out.shape == batch.x.shape


def test_spans_skeletons_of_different_sizes_in_one_batch(batch):
    # 31 to 63 joints in a single forward pass; the topology conditioner is
    # what makes that possible rather than a per-skeleton model.
    counts = {int(n) for n in batch.masks.n_joints}
    assert len(counts) > 1
    assert min(counts) == 31 and max(counts) == 63


def test_padded_joints_do_not_influence_real_ones(batch):
    # Perturbing a padded joint must leave every real joint's output untouched,
    # or the spatial mask is not doing its job.
    torch.manual_seed(0)
    model = small_model(batch).eval()
    t = torch.zeros(len(batch), dtype=torch.long)

    with torch.no_grad():
        before = model(batch.x, t, batch.cond, batch.masks).out
        perturbed = batch.x.clone()
        for i in range(len(batch)):
            perturbed[i, int(batch.masks.n_joints[i]) :] = 7.0
        after = model(perturbed, t, batch.cond, batch.masks).out

    for i in range(len(batch)):
        real = int(batch.masks.n_joints[i])
        torch.testing.assert_close(after[i, :real], before[i, :real], rtol=1e-4, atol=1e-5)


def test_timestep_changes_the_prediction(batch):
    torch.manual_seed(0)
    model = small_model(batch).eval()
    with torch.no_grad():
        early = model(batch.x, torch.zeros(len(batch), dtype=torch.long), batch.cond, batch.masks)
        late = model(batch.x, torch.full((len(batch),), 99), batch.cond, batch.masks)
    assert not torch.allclose(early.out, late.out)


def test_topology_changes_the_prediction(batch):
    # If rewiring the skeleton graph changed nothing, the model would not be
    # topology aware at all.
    torch.manual_seed(0)
    model = small_model(batch).eval()
    t = torch.zeros(len(batch), dtype=torch.long)

    scrambled = {
        "topology": {
            "hops": torch.zeros_like(batch.cond["topology"]["hops"]),
            "relations": torch.full_like(
                batch.cond["topology"]["relations"], int(EdgeType.NO_RELATION)
            ),
        },
        "tpose": batch.cond["tpose"],
    }
    with torch.no_grad():
        original = model(batch.x, t, batch.cond, batch.masks).out
        rewired = model(batch.x, t, Cond(scrambled), batch.masks).out
    assert not torch.allclose(original, rewired)


def test_rest_pose_changes_the_prediction(batch):
    torch.manual_seed(0)
    model = small_model(batch).eval()
    t = torch.zeros(len(batch), dtype=torch.long)
    other = Cond({**batch.cond.payloads, "tpose": torch.zeros_like(batch.cond["tpose"])})
    with torch.no_grad():
        a = model(batch.x, t, batch.cond, batch.masks).out
        b = model(batch.x, t, other, batch.masks).out
    assert not torch.allclose(a, b)


def test_trains_on_a_real_batch(batch):
    torch.manual_seed(0)
    task = MotionTask(
        model=small_model(batch),
        process=GaussianDiffusion(),
        losses=[(n, 1.0, LOSSES.get(n)()) for n in ("simple", "geodesic", "footskate")],
    )
    task.setup_checks(batch)
    losses = task.compute_losses(batch)

    assert all(v.isfinite() for v in losses.values())
    losses["total"].backward()
    grads = [p.grad for p in task.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0


def test_loss_decreases_when_overfitting_one_batch(batch):
    # The strongest cheap signal that the wiring is right: it can learn.
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


def test_graph_attention_rejects_indivisible_head_count():
    with pytest.raises(ValueError, match="divisible"):
        GraphAttention(d_model=10, n_heads=4)
