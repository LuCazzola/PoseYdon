"""Humanized joint names (parity spec §7).

RAW names remain the canonical identity -- manifests and `resolve()` match on
them, and that is what makes a re-exported rig fail loudly instead of silently
mirroring the character. Humanized names are only what a text encoder sees.
"""

from __future__ import annotations

import pytest

from poseydon.build.names import Humanize


def test_the_common_prefix_is_computed_not_declared():
    """Bip01_ is stripped because every joint carries it, not because a
    manifest said so -- which is the whole point of replacing
    strip_joint_prefix."""
    out = Humanize()(["Bip01_Pelvis", "Bip01_Spine", "Bip01_R_Thigh"])
    assert out == ("pelvis", "spine", "right thigh")


def test_a_prefix_shared_by_only_some_joints_is_kept():
    out = Humanize()(["Bip01_Pelvis", "Spine", "Bip01_Head"])
    assert out == ("bip01 pelvis", "spine", "bip01 head")


def test_camel_case_is_split():
    assert Humanize()(["LeftForeArm"]) == ("left fore arm",)


def test_isolated_side_letters_expand_and_embedded_ones_do_not():
    """`R` alone is a side; the R in `Arm` is not."""
    out = Humanize()(["R_Thigh", "L_Foot", "Arm"])
    assert out == ("right thigh", "left foot", "arm")


def test_expand_sides_can_be_disabled():
    assert Humanize(expand_sides=False)(["R_Thigh"]) == ("r thigh",)


def test_lowercase_can_be_disabled():
    """Side expansion substitutes a lowercase word; the rest keeps its case."""
    out = Humanize(lowercase=False)(["Bip01_R_Thigh", "Bip01_L_Foot"])
    assert out == ("right Thigh", "left Foot")


def test_a_single_name_has_no_shared_prefix():
    """One name has no sibling to share a prefix with, so nothing is stripped.
    Guessing at boilerplate here would turn `Leg_01` into `01`."""
    assert Humanize()(["Bip01_R_Thigh"]) == ("bip01 right thigh",)
    assert Humanize()(["Leg_01"]) == ("leg 01",)


def test_real_truebones_names():
    """Names taken verbatim from the corpus, not invented."""
    out = Humanize()(["BN_Leg_R_11", "BN_Leg_L_11", "BN_Arm_R_02"])
    assert out == ("leg right 11", "leg left 11", "arm right 02")


def test_output_length_always_matches_input():
    names = ["Hips", "Bip01_Pelvis", "jt_Cog_C", "_00", "Sabrecat__pelv_"]
    assert len(Humanize()(names)) == len(names)


def test_an_empty_name_list_is_refused():
    with pytest.raises(ValueError, match="no joint names"):
        Humanize()([])
