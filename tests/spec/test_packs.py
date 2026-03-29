"""Tests for config pack resolution."""

import contextlib
import os
import tempfile

import pytest
import yaml

from coleman4hcs.spec.packs import _deep_merge, resolve_packs


class TestDeepMerge:
    def test_simple_merge(self):
        result = _deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_override_scalar(self):
        result = _deep_merge({"a": 1}, {"a": 2})
        assert result == {"a": 2}

    def test_nested_merge(self):
        base = {"a": {"b": 1, "c": 2}}
        over = {"a": {"c": 3, "d": 4}}
        result = _deep_merge(base, over)
        assert result == {"a": {"b": 1, "c": 3, "d": 4}}

    def test_does_not_mutate_inputs(self):
        base = {"a": {"b": 1}}
        over = {"a": {"c": 2}}
        _deep_merge(base, over)
        assert base == {"a": {"b": 1}}


class TestResolvePacks:
    def test_no_packs(self):
        raw = {"execution": {"verbose": True}}
        result = resolve_packs(raw, packs_dir="/nonexistent")
        assert result == {"execution": {"verbose": True}}

    def test_pack_file_not_found(self):
        raw = {"packs": ["missing/pack"]}
        with pytest.raises(FileNotFoundError, match="Pack file not found"):
            resolve_packs(raw, packs_dir="/tmp/no_such_dir")

    def test_single_pack(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "policy"))
            pack_path = os.path.join(tmpdir, "policy", "ucb.yaml")
            with open(pack_path, "w") as fh:
                yaml.dump({"experiment": {"policies": ["UCB"]}}, fh)

            raw = {"packs": ["policy/ucb"], "experiment": {"rewards": ["RNFail"]}}
            result = resolve_packs(raw, packs_dir=tmpdir)

            assert result["experiment"]["policies"] == ["UCB"]
            assert result["experiment"]["rewards"] == ["RNFail"]
            assert "packs" not in result

    def test_multiple_packs_merge_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "policy"))
            os.makedirs(os.path.join(tmpdir, "results"))

            with open(os.path.join(tmpdir, "policy", "greedy.yaml"), "w") as fh:
                yaml.dump({"experiment": {"policies": ["Greedy"]}}, fh)
            with open(os.path.join(tmpdir, "results", "parquet.yaml"), "w") as fh:
                yaml.dump({"results": {"sink": "parquet", "enabled": True}}, fh)

            raw = {"packs": ["policy/greedy", "results/parquet"]}
            result = resolve_packs(raw, packs_dir=tmpdir)

            assert result["experiment"]["policies"] == ["Greedy"]
            assert result["results"]["sink"] == "parquet"

    def test_inline_overrides_win(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "results"))
            with open(os.path.join(tmpdir, "results", "parquet.yaml"), "w") as fh:
                yaml.dump({"results": {"sink": "parquet", "batch_size": 500}}, fh)

            raw = {"packs": ["results/parquet"], "results": {"batch_size": 2000}}
            result = resolve_packs(raw, packs_dir=tmpdir)

            # Pack sets sink, but inline overrides batch_size
            assert result["results"]["sink"] == "parquet"
            assert result["results"]["batch_size"] == 2000

    def test_does_not_mutate_input(self):
        raw = {"packs": ["missing"], "execution": {"verbose": True}}
        original = dict(raw)
        # Even though the pack doesn't exist, the input should not be mutated
        # by the time the error is raised.
        with contextlib.suppress(FileNotFoundError):
            resolve_packs(raw, packs_dir="/tmp/no_such_dir")
        assert raw == original
