"""A recipe written before a config key existed cannot disagree about it."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from poseydon.training.recipe import check_recipe


def _recipe(model: dict) -> OmegaConf:
    return OmegaConf.create(
        {
            "features": ["ric_pos"],
            "conditioners": ["topology"],
            "model": model,
            "process": {"_target_": "p"},
            "dataset": {"reduce_tolerance": 1e-8, "normalize": "per_channel"},
        }
    )


def test_a_key_the_recipe_predates_is_not_checked():
    """Adding a model config key must not invalidate every checkpoint on disk."""
    recorded = _recipe({"_target_": "m", "d_model": 128})
    active = _recipe({"_target_": "m", "d_model": 128, "cond_drop": 0.0})
    check_recipe(active, recorded)


def test_a_key_the_active_config_lacks_still_raises():
    recorded = _recipe({"_target_": "m", "d_model": 128, "cond_drop": 0.1})
    active = _recipe({"_target_": "m", "d_model": 128})
    with pytest.raises(ValueError, match="cond_drop"):
        check_recipe(active, recorded)


def test_a_real_disagreement_still_raises():
    recorded = _recipe({"_target_": "m", "d_model": 128, "cond_drop": 0.1})
    active = _recipe({"_target_": "m", "d_model": 128, "cond_drop": 0.0})
    with pytest.raises(ValueError, match="cond_drop"):
        check_recipe(active, recorded)


def test_a_whole_missing_path_is_not_checked_either():
    recorded = _recipe({"_target_": "m"})
    recorded.pop("process")
    active = _recipe({"_target_": "m"})
    check_recipe(active, recorded)


def test_an_absent_key_and_a_real_conflict_in_one_block_names_only_the_conflict():
    """`_rows` is row-scoped, not block-scoped: one checked block can carry both
    a key that postdates the recipe (ignored) and a key that genuinely
    disagrees (raised), and the escape must not swallow the real conflict.
    """
    recorded = _recipe({"_target_": "m", "d_model": 128, "cond_drop": 0.1})
    active = _recipe(
        {"_target_": "m", "d_model": 128, "cond_drop": 0.0, "dropout": 0.1}
    )
    with pytest.raises(ValueError, match="model.cond_drop") as excinfo:
        check_recipe(active, recorded)
    assert "model.dropout" not in str(excinfo.value)
