"""Stage contract and chain composition."""

from __future__ import annotations

import numpy as np
import pytest

from poseydon.build.prepare import PrepareChain, PrepareStage
from poseydon.core.animation import Animation


def _anim(n_frames: int = 3, n_joints: int = 2) -> Animation:
    rotations = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n_frames, n_joints, 1))
    offsets = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    translations = np.broadcast_to(offsets, (n_frames, n_joints, 3)).copy()
    # A non-zero root trajectory, so `Shift` is not a no-op. With a zero root
    # the order tests below cannot tell reverse-order inversion from
    # forward-order inversion: adding zero commutes with everything.
    translations[:, 0] = np.array([5.0, 2.0, -3.0])
    return Animation(
        rotations=rotations,
        translations=translations,
        offsets=offsets,
        parents=np.array([-1, 0], dtype=np.int32),
        names=("root", "child"),
        fps=30.0,
    )


class Shift(PrepareStage):
    """Adds a fitted constant to the root trajectory. Rig-scoped."""

    name = "shift"
    scope = "rig"

    def fit(self, anim, resolved):
        return {"amount": anim.translations[0, 0].copy()}

    def apply(self, anim, params):
        translations = anim.translations.copy()
        translations[:, 0] = translations[:, 0] - params["amount"]
        return Animation(
            rotations=anim.rotations, translations=translations, offsets=anim.offsets,
            parents=anim.parents, names=anim.names, fps=anim.fps,
        )

    def invert(self, anim, params):
        translations = anim.translations.copy()
        translations[:, 0] = translations[:, 0] + params["amount"]
        return Animation(
            rotations=anim.rotations, translations=translations, offsets=anim.offsets,
            parents=anim.parents, names=anim.names, fps=anim.fps,
        )


class Double(PrepareStage):
    """Scales every offset. Clip-scoped, so it refits per animation."""

    name = "double"
    scope = "clip"

    def fit(self, anim, resolved):
        return {"factor": np.float64(2.0)}

    def apply(self, anim, params):
        return Animation(
            rotations=anim.rotations, translations=anim.translations * params["factor"],
            offsets=anim.offsets * params["factor"], parents=anim.parents,
            names=anim.names, fps=anim.fps,
        )

    def invert(self, anim, params):
        return Animation(
            rotations=anim.rotations, translations=anim.translations / params["factor"],
            offsets=anim.offsets / params["factor"], parents=anim.parents,
            names=anim.names, fps=anim.fps,
        )


def test_fit_rig_keeps_only_rig_scoped_parameters():
    chain = PrepareChain((Shift(), Double()))
    rig = chain.fit_rig(_anim(), resolved=None)
    assert set(rig) == {"shift"}


def test_apply_then_invert_is_the_identity():
    chain = PrepareChain((Shift(), Double()))
    source = _anim()
    rig = chain.fit_rig(source, resolved=None)

    prepared, params = chain.apply(source, resolved=None, rig_params=rig)
    restored = chain.invert(prepared, params)

    np.testing.assert_allclose(restored.translations, source.translations, atol=1e-12)
    np.testing.assert_allclose(restored.offsets, source.offsets, atol=1e-12)


def test_invert_runs_stages_in_reverse_order():
    """Shift-then-Double does not commute. Unwinding in the wrong order divides
    the restored root trajectory by two, so the correct order recovers the
    source and the swapped order must not."""
    chain = PrepareChain((Shift(), Double()))
    source = _anim()
    rig = chain.fit_rig(source, resolved=None)
    prepared, params = chain.apply(source, resolved=None, rig_params=rig)

    restored = chain.invert(prepared, params)
    np.testing.assert_allclose(
        restored.translations[:, 0], source.translations[:, 0], atol=1e-12
    )

    # The same stages unwound forward instead of reversed must NOT recover it.
    wrong = prepared
    for stage in chain.stages:
        wrong = stage.invert(wrong, params[stage.name])
    assert not np.allclose(
        wrong.translations[:, 0], source.translations[:, 0], atol=1e-9
    )


def test_a_clip_scoped_stage_refits_on_each_animation():
    chain = PrepareChain((Double(),))
    rig = chain.fit_rig(_anim(), resolved=None)
    assert rig == {}
    _prepared, params = chain.apply(_anim(), resolved=None, rig_params=rig)
    assert "double" in params


def test_apply_rejects_a_missing_rig_parameter():
    chain = PrepareChain((Shift(),))
    with pytest.raises(KeyError, match="shift"):
        chain.apply(_anim(), resolved=None, rig_params={})
