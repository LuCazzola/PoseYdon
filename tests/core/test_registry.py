import pytest

from poseydon.core.registry import Registry


class Thing:
    pass


def test_register_and_get():
    registry: Registry[Thing] = Registry("thing")

    @registry.register("alpha")
    class Alpha(Thing):
        pass

    assert registry.get("alpha") is Alpha
    assert "alpha" in registry
    assert len(registry) == 1


def test_unknown_name_suggests_a_close_match():
    registry: Registry[Thing] = Registry("thing")
    registry.register("rot6d")(Thing)
    with pytest.raises(KeyError) as excinfo:
        registry.get("rot6")
    assert "rot6d" in str(excinfo.value)


def test_duplicate_registration_is_rejected():
    registry: Registry[Thing] = Registry("thing")
    registry.register("alpha")(Thing)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("alpha")(Thing)


def test_names_are_sorted():
    registry: Registry[Thing] = Registry("thing")
    for name in ("zeta", "alpha", "mu"):
        registry.register(name)(type(name, (Thing,), {}))
    assert registry.names() == ["alpha", "mu", "zeta"]
