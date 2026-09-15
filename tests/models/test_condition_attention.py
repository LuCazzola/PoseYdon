"""Semantic modulation: one gated transform, or several averaged."""

from __future__ import annotations

import pytest
import torch

from poseydon.models.modules.graph_backbone import ConditionAttention

D_MODEL = 8


def _inputs():
    torch.manual_seed(0)
    return torch.randn(3, 2, 4, D_MODEL), torch.randn(3, 2, 1, D_MODEL)


def test_one_head_keeps_the_published_parameter_names():
    """The reference's checkpoints load into this module by name. One head must
    stay exactly the module it has always been."""
    keys = set(ConditionAttention(D_MODEL).state_dict())
    assert keys == {
        "query_projector.0.weight", "query_projector.0.bias",
        "query_projector.1.weight", "query_projector.1.bias",
        "query_projector.3.weight", "query_projector.3.bias",
        "value_map.weight", "value_map.bias",
        "out_proj.weight", "out_proj.bias",
    }


def test_one_head_is_the_gated_transform_it_has_always_been():
    """Not just the same parameter names -- the same arithmetic. Guards the
    `n_heads == 1` path against a refactor that changes what it computes."""
    x, memory = _inputs()
    module = ConditionAttention(D_MODEL, dropout=0.0).eval()

    expanded = memory.expand(-1, -1, x.shape[2], -1)
    gate = torch.sigmoid(module.query_projector(torch.cat([x, expanded], dim=-1)))
    expected = module.out_proj(module.value_map(expanded) * gate)
    assert torch.allclose(module(x, memory), expected, atol=1e-6)


def test_mean_over_identical_heads_equals_one_head():
    x, memory = _inputs()
    many = ConditionAttention(D_MODEL, dropout=0.0, n_heads=3, aggregate="mean").eval()
    for head in list(many.heads)[1:]:
        head.load_state_dict(many.heads[0].state_dict())

    one = ConditionAttention(D_MODEL, dropout=0.0).eval()
    one.query_projector.load_state_dict(many.heads[0]["query_projector"].state_dict())
    one.value_map.load_state_dict(many.heads[0]["value_map"].state_dict())
    one.out_proj.load_state_dict(many.out_proj.state_dict())

    assert torch.allclose(many(x, memory), one(x, memory), atol=1e-6)


def test_sum_over_identical_heads_is_n_times_one_head():
    x, memory = _inputs()
    many = ConditionAttention(D_MODEL, dropout=0.0, n_heads=3, aggregate="sum").eval()
    for head in list(many.heads)[1:]:
        head.load_state_dict(many.heads[0].state_dict())

    one = ConditionAttention(D_MODEL, dropout=0.0).eval()
    one.query_projector.load_state_dict(many.heads[0]["query_projector"].state_dict())
    one.value_map.load_state_dict(many.heads[0]["value_map"].state_dict())
    # `out_proj` is linear but has a bias, so compare before it: 3x the summed
    # pre-projection signal is not 3x the projected one.
    one.out_proj.load_state_dict(many.out_proj.state_dict())
    with torch.no_grad():
        one.out_proj.bias.zero_()
        many.out_proj.bias.zero_()

    assert torch.allclose(many(x, memory), 3.0 * one(x, memory), atol=1e-6)


def test_heads_are_independent_by_default():
    many = ConditionAttention(D_MODEL, n_heads=2)
    first = many.heads[0]["value_map"].weight
    second = many.heads[1]["value_map"].weight
    assert not torch.allclose(first, second)


@pytest.mark.parametrize("bad", [0, -1])
def test_a_head_count_below_one_is_rejected(bad):
    with pytest.raises(ValueError, match="at least one"):
        ConditionAttention(D_MODEL, n_heads=bad)


def test_an_unknown_aggregation_is_rejected():
    with pytest.raises(ValueError, match="aggregate"):
        ConditionAttention(D_MODEL, n_heads=2, aggregate="concat")
