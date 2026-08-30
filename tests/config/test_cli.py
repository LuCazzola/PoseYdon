"""The command line surface."""

import pytest

from poseydon.cli import main


def test_list_enumerates_every_registry(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    for kind in ("features", "conditioners", "losses", "models", "processes", "samplers"):
        assert f"{kind}:" in out
    assert "anytop" in out
    assert "foot_contact" in out


def test_list_can_be_narrowed(capsys):
    assert main(["list", "models"]) == 0
    out = capsys.readouterr().out
    assert "models:" in out
    assert "features:" not in out


def test_print_config_shows_the_composed_result(capsys):
    assert main(["train", "--print-config", "trainer=debug"]) == 0
    out = capsys.readouterr().out
    assert "features:" in out
    assert "max_steps: 5" in out


def test_print_config_reflects_overrides(capsys):
    assert main(["train", "--print-config", "process=flow"]) == 0
    assert "FlowMatching" in capsys.readouterr().out


def test_unknown_subcommand_fails():
    with pytest.raises(SystemExit):
        main(["nonsense"])
