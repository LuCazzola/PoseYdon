"""Joint-name conditioning: the cache resolves, and the model actually uses it.

The reference trains with `skip_t5: False` and `cond_mask_prob: 0.0`, so these
embeddings ride every forward pass. `MoDiffAE` accepts them as OPTIONAL and
skips the term in silence when they are absent -- which is why their absence
went unnoticed through a whole training run. The second test below is the one
that would have caught it: it asserts the key changes the output at all.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from poseydon.conditioners import CONDITIONERS
from poseydon.conditioners.skeleton import DEFAULT_JOINT_NAME_CACHE, JointNames
from poseydon.models.modiffae import JOINT_NAMES

RIGS = Path("data/truebones/rigs")


def _cache() -> JointNames:
    if not DEFAULT_JOINT_NAME_CACHE.exists():
        pytest.skip("joint-name cache not built (tools/build_joint_name_cache.py)")
    return CONDITIONERS.get(JOINT_NAMES)()


def _reduced_names(rig: Path) -> list[str]:
    with np.load(rig / "skeleton.npz", allow_pickle=True) as data:
        full = [str(n) for n in data["names"]]
        return [full[int(i)] for i in data["reduction_source_of"]]


def _view(names: list[str], parents: list[int]) -> SimpleNamespace:
    anim = SimpleNamespace(names=tuple(names), parents=np.array(parents, dtype="int64"))
    return SimpleNamespace(anim=anim)


def test_every_joint_of_every_rig_resolves():
    """A miss here is a silent training failure, not a crash.

    A name the cache does not carry raises at read time, mid-epoch, on whichever
    rig happens to come up -- so the coverage is asserted up front, over the
    whole corpus, rather than discovered by a dataloader worker.
    """
    conditioner = _cache()
    rigs = sorted(p for p in RIGS.iterdir() if p.is_dir())
    if not rigs:
        pytest.skip("no prepared rigs")

    missing: list[str] = []
    for rig in rigs:
        for name in _reduced_names(rig):
            try:
                conditioner._lookup(name)
            except KeyError:
                missing.append(f"{rig.name}/{name}")
    assert not missing, f"{len(missing)} joint names have no embedding: {missing[:10]}"


def test_the_payload_is_one_row_per_joint():
    conditioner = _cache()
    names = _reduced_names(min(p for p in RIGS.iterdir() if p.is_dir()))
    payload = conditioner.extract(_view(names, [-1] + [0] * (len(names) - 1)))
    assert payload.shape == (len(names), 768)
    assert payload.dtype == torch.float32


def test_a_duplicated_joint_takes_the_midpoint_of_its_source_and_parent():
    """Mirrors the reference's `add_joint_augmentation`, which averages the two.

    `DuplicateJoint` inserts `<source>__mid` with the midpoint of the source's
    and the parent's FEATURES; the name embedding follows the same rule, so the
    inserted joint reads as what it geometrically is.
    """
    conditioner = _cache()
    table = conditioner._table
    parent_name, source_name = sorted(table)[:2]

    view = _view([parent_name, f"{source_name}__mid"], [-1, 0])
    payload = conditioner.extract(view).numpy()

    expected = (table[source_name] + table[parent_name]) / 2.0
    np.testing.assert_allclose(payload[1], expected, rtol=1e-6)


def test_an_unknown_name_names_itself_and_the_fix():
    conditioner = _cache()
    with pytest.raises(KeyError, match="no cached embedding"):
        conditioner.extract(_view(["not_a_real_joint_zzz"], [-1]))


def test_padded_joints_get_a_zero_row():
    conditioner = _cache()
    name = min(conditioner._table)
    batched = conditioner.collate([conditioner.extract(_view([name], [-1]))], max_joints=4)
    assert batched.shape == (1, 4, 768)
    assert torch.count_nonzero(batched[0, 1:]) == 0


def _forward(model, cond_extra: dict) -> torch.Tensor:
    """One deterministic MoDiffAE forward, differing only by `cond_extra`."""
    from poseydon.core.batch import Cond
    from poseydon.models.base import CLEAN_MOTION

    joints, frames, dim = 3, 8, 13
    torch.manual_seed(0)
    clean = torch.randn(1, joints, dim, frames)
    z_t = torch.randn(1, joints, dim, frames)
    cond = Cond({
        "topology": {
            "hops": torch.zeros(1, joints, joints, dtype=torch.long),
            "relations": torch.zeros(1, joints, joints, dtype=torch.long),
        },
        "tpose": torch.zeros(1, joints, dim),
        CLEAN_MOTION: clean,
        **cond_extra,
    })
    with torch.no_grad():
        return model(z_t, torch.zeros(1, dtype=torch.long), cond).out


def test_the_conditioning_actually_reaches_the_output():
    """The test whose absence let a whole training run go without joint names.

    `MoDiffAE` reads this key with `cond.get(JOINT_NAMES)` and, finding nothing,
    skips the embedding term without a word. Every other test in the tree passed
    identically with and without it. Asserting only that a forward pass runs
    cannot tell a model that used the conditioning from one that ignored it.
    """
    from poseydon.models.modiffae import MoDiffAE

    model = MoDiffAE(feature_dim=13, d_model=32, n_heads=2, temporal_window=31).eval()
    torch.manual_seed(1)
    names = torch.randn(1, 3, 768)

    without = _forward(model, {})
    with_names = _forward(model, {JOINT_NAMES: names})

    assert not torch.allclose(without, with_names, atol=1e-6), (
        "joint-name conditioning left the output unchanged -- the model is "
        "ignoring the key"
    )
