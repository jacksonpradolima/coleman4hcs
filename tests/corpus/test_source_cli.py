"""Source CLI validation is presentation-only and requires no source cache or execution runtime."""

import pytest

from coleman.transports.source_cli import source_arguments


def test_empty_preset_contract_rejects_before_parsing():
    with pytest.raises(ValueError, match="at least one declared preset"):
        source_arguments("owned", (), [])


def test_mutually_exclusive_recipe_choices_reject():
    with pytest.raises(SystemExit) as error:
        source_arguments("owned", ("owned-v1",), ["--preset", "owned-v1", "--recipe-stdin"])
    assert error.value.code == 2


def test_explicit_fetch_and_paths_remain_cli_choices(tmp_path):
    args = source_arguments(
        "owned",
        ("owned-v1",),
        ["--cache", str(tmp_path / "cache"), "--output", str(tmp_path / "output"), "--trial", "owned", "--fetch"],
    )
    assert args.preset == "owned-v1"
    assert args.fetch is True
    assert args.cache == tmp_path / "cache"
    assert args.output == tmp_path / "output"
    assert not tmp_path.joinpath("cache").exists()
    assert not tmp_path.joinpath("output").exists()
