"""Tests for provenance tracking."""

import json
import os
import subprocess
import tempfile
from unittest.mock import patch

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

    def test_git_info_failure_returns_none_fields(self):
        """When git is unavailable lines 46-48 are covered."""
        with patch("subprocess.check_output", side_effect=FileNotFoundError("git not found")):
            prov = build_provenance()
        assert prov["git"]["commit"] is None
        assert prov["git"]["dirty"] is None

    def test_git_info_called_process_error_returns_none_fields(self):
        """CalledProcessError is also handled gracefully."""
        with patch(
            "subprocess.check_output",
            side_effect=subprocess.CalledProcessError(128, "git"),
        ):
            prov = build_provenance()
        assert prov["git"]["commit"] is None

    def test_lock_hash_returns_none_when_uv_lock_absent(self):
        """Line 65 in provenance.py: _lock_hash returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = os.getcwd()
            try:
                os.chdir(tmpdir)  # no uv.lock here
                prov = build_provenance()
                assert prov["uv_lock_hash"] is None
            finally:
                os.chdir(original_dir)


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
