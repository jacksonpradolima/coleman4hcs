"""Explicit local/fetch/offline transport behavior around actual source materialization and conversion."""

import io
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from coleman.transports.westermo_etl import _recipe, main
from tests.corpus.westermo_support import owned_recipe


def test_local_source_and_offline_cli_are_identical(tmp_path, capsys, monkeypatch):
    recipe, payload = owned_recipe()
    source = tmp_path / "source.tsv"
    source.write_bytes(payload)
    args = ["--recipe-stdin", "--cache", str(tmp_path / "cache"), "--output", str(tmp_path / "out")]
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(recipe.to_json().encode())))
    assert main([*args, "--source", str(source)]) == 0
    first = json.loads(capsys.readouterr().out)
    source.unlink()
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(recipe.to_json().encode())))
    assert main(args) == 0
    second = json.loads(capsys.readouterr().out)
    assert first == second
    assert first["counts"]["selected_rows"] == 5
    assert first["counts"]["invalid"] == 1
    assert first["admission"] == "blocked_source_evidence"


def test_explicit_fetch_uses_only_transport_boundary_and_cache_retry(tmp_path, monkeypatch, capsys):
    recipe, payload = owned_recipe()
    calls = []

    @contextmanager
    def http(url, size):
        calls.append((url, size))
        with io.BytesIO(payload) as stream:
            yield stream

    monkeypatch.setattr("coleman.transports.westermo_etl.public_https_source", http)
    args = ["--recipe-stdin", "--cache", str(tmp_path / "cache"), "--output", str(tmp_path / "out")]
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(recipe.to_json().encode())))
    assert main([*args, "--fetch"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert calls == [(recipe.source.source.uri, len(payload))]
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(recipe.to_json().encode())))
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out) == first
    assert len(calls) == 1


def test_no_implicit_network_bad_metadata_and_preflight_fail_before_acquisition(tmp_path, monkeypatch):
    assert len(_recipe(False).sessions) == 16
    recipe, _ = owned_recipe()
    args = ["--recipe-stdin", "--cache", str(tmp_path / "cache"), "--output", str(tmp_path / "out")]
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(recipe.to_json().encode())))
    with pytest.raises(ValueError, match="not cached"):
        main(args)
    with pytest.raises(SystemExit) as error:
        main([*args, "--source", "source.zip", "--fetch"])
    assert error.value.code == 2
    assert not list(tmp_path.rglob("complete.json"))
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(b" " * (1024**2 + 1))))
    with pytest.raises(ValueError, match="recipe exceeds"):
        main(args)
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(recipe.to_json().encode())))
    overlapping = ["--recipe-stdin", "--cache", str(tmp_path / "equal"), "--output", str(tmp_path / "equal")]
    with pytest.raises(ValueError, match="remain separate"):
        main(overlapping)


def test_recipe_bytes_cannot_be_interpreted_as_a_filename(tmp_path, monkeypatch):
    """Metadata is a bounded data stream; neither bare nor JSON-quoted filenames can trigger file reads."""
    path = tmp_path / "not-a-recipe.txt"
    path.write_text("Owned marker, not versioned recipe metadata")
    args = ["--recipe-stdin", "--cache", str(tmp_path / "cache"), "--output", str(tmp_path / "out")]

    def forbid_file_access(*_args, **_kwargs):
        """Reject every file open at the boundary; malformed metadata must fail before filesystem access."""
        raise AssertionError("Recipe metadata attempted to open a caller-named file")

    with monkeypatch.context() as boundary:
        boundary.setattr(Path, "open", forbid_file_access)
        for payload in (str(path).encode(), json.dumps(str(path)).encode()):
            boundary.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(payload)))
            with pytest.raises(ValueError):
                main(args)
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "out").exists()
    assert path.read_text() == "Owned marker, not versioned recipe metadata"
