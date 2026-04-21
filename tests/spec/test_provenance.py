"""Tests for provenance tracking."""

import json
import os
import tempfile

from coleman.spec.provenance import build_provenance, save_provenance


class TestBuildProvenance:
    def test_contains_required_keys(self):
        prov = build_provenance()
        assert "python_version" in prov
        assert "platform" in prov
        assert "cwd" in prov
        assert "git" in prov
        assert "uv_lock_hash" in prov

    def test_git_info_structure(self):
        prov = build_provenance()
        git = prov["git"]
        assert "commit" in git
        assert "dirty" in git


class TestSaveProvenance:
    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = save_provenance(tmpdir)
            assert out.exists()
            assert out.name == "provenance.json"
            with open(out) as fh:
                data = json.load(fh)
            assert "python_version" in data

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "a", "b", "c")
            out = save_provenance(nested)
            assert out.exists()
