import pytest

from poseydon.core.spec import Block, FeatureSpec

REFERENCE = FeatureSpec(
    (("ric_pos", 3), ("rot6d", 6), ("local_vel", 3), ("foot_contact", 1))
)


def test_dim_is_the_sum_of_widths():
    assert REFERENCE.dim == 13


def test_slices_are_contiguous_and_ordered():
    assert REFERENCE.slice("ric_pos") == slice(0, 3)
    assert REFERENCE.slice("rot6d") == slice(3, 9)
    assert REFERENCE.slice("local_vel") == slice(9, 12)
    assert REFERENCE.slice("foot_contact") == slice(12, 13)


def test_missing_block_fails_loudly_and_names_what_is_present():
    # This is the whole point: a magic [..., 3:9] would silently read whatever
    # sits there instead.
    with pytest.raises(KeyError) as excinfo:
        FeatureSpec((("ric_pos", 3),)).slice("rot6d")
    assert "rot6d" in str(excinfo.value)
    assert "ric_pos" in str(excinfo.value)


def test_subset_spec_reindexes_from_zero():
    subset = FeatureSpec((("rot6d", 6), ("local_vel", 3)))
    assert subset.slice("rot6d") == slice(0, 6)
    assert subset.dim == 9


def test_rejects_duplicate_blocks():
    with pytest.raises(ValueError, match="duplicate"):
        FeatureSpec((("rot6d", 6), ("rot6d", 6)))


def test_rejects_non_positive_width():
    with pytest.raises(ValueError, match="width"):
        FeatureSpec((("rot6d", 0),))


def test_contains_and_names():
    assert "rot6d" in REFERENCE
    assert "angular_vel" not in REFERENCE
    assert REFERENCE.names == ("ric_pos", "rot6d", "local_vel", "foot_contact")


def test_block_defaults_to_normalized_space():
    assert Block("rot6d").space == "normalized"
    assert Block("rot6d", space="raw").space == "raw"
