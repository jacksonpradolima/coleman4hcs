"""Explicit local/fetch/offline transport behavior around actual source materialization and conversion."""

import io
import json
from contextlib import contextmanager

import pytest

from coleman.transports.westermo_etl import _recipe, main
from tests.corpus.westermo_support import owned_recipe


def test_local_source_and_offline_cli_are_identical(tmp_path, capsys):
    recipe, payload = owned_recipe()
    recipe_path, source = tmp_path / "recipe.json", tmp_path / "source.tsv"
    recipe_path.write_text(recipe.to_json())
    source.write_bytes(payload)
    args = ["--recipe", str(recipe_path), "--cache", str(tmp_path / "cache"), "--output", str(tmp_path / "out")]
    assert main([*args, "--source", str(source)]) == 0
    first = json.loads(capsys.readouterr().out)
    source.unlink()
    assert main(args) == 0
    second = json.loads(capsys.readouterr().out)
    assert first == second
    assert first["counts"]["selected_rows"] == 5
    assert first["counts"]["invalid"] == 1
    assert first["admission"] == "blocked_source_evidence"


def test_explicit_fetch_uses_only_transport_boundary_and_cache_retry(tmp_path, monkeypatch, capsys):
    recipe, payload = owned_recipe()
    path = tmp_path / "recipe.json"
    path.write_text(recipe.to_json())
    calls = []

    @contextmanager
    def http(url, size):
        calls.append((url, size))
        with io.BytesIO(payload) as stream:
            yield stream

    monkeypatch.setattr("coleman.transports.westermo_etl.public_https_source", http)
    args = ["--recipe", str(path), "--cache", str(tmp_path / "cache"), "--output", str(tmp_path / "out")]
    assert main([*args, "--fetch"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert calls == [(recipe.source.source.uri, len(payload))]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out) == first
    assert len(calls) == 1


def test_no_implicit_network_bad_metadata_and_preflight_fail_before_acquisition(tmp_path):
    assert len(_recipe(None).sessions) == 16
    recipe, _ = owned_recipe()
    path = tmp_path / "recipe.json"
    path.write_text(recipe.to_json())
    args = ["--recipe", str(path), "--cache", str(tmp_path / "cache"), "--output", str(tmp_path / "out")]
    with pytest.raises(ValueError, match="not cached"):
        main(args)
    with pytest.raises(SystemExit) as error:
        main([*args, "--source", "source.zip", "--fetch"])
    assert error.value.code == 2
    assert not list(tmp_path.rglob("complete.json"))
    path.write_bytes(b" " * (1024**2 + 1))
    with pytest.raises(ValueError, match="recipe exceeds"):
        main(args)
    path.write_text(recipe.to_json())
    with pytest.raises(ValueError, match="remain separate"):
        main(["--recipe", str(path), "--cache", str(tmp_path / "equal"), "--output", str(tmp_path / "equal")])
