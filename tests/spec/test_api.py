"""Tests for the public API module."""

import os
import tempfile

import yaml

from coleman4hcs.api import RunResult, load_spec, run, run_many, save_resolved, sweep
from coleman4hcs.spec.models import RunSpec
from coleman4hcs.spec.run_id import compute_run_id
from coleman4hcs.spec.sweep import SweepAxis, SweepSpec


class TestRunResult:
    def test_repr(self):
        spec = RunSpec()
        r = RunResult(run_id="abc123", spec=spec, metrics={"napfd": 0.9})
        assert "abc123" in repr(r)
        assert "napfd" in repr(r)

    def test_defaults(self):
        spec = RunSpec()
        r = RunResult(run_id="x", spec=spec)
        assert r.metrics == {}
        assert r.artifacts_dir is None


class TestRun:
    def test_produces_run_id(self):
        spec = RunSpec(results={"out_dir": tempfile.mkdtemp()})
        result = run(spec)
        assert len(result.run_id) == 12
        assert result.run_id == compute_run_id(spec)

    def test_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = RunSpec(results={"out_dir": tmpdir})
            result = run(spec)
            assert result.artifacts_dir is not None
            assert os.path.exists(os.path.join(result.artifacts_dir, "spec.resolved.json"))
            assert os.path.exists(os.path.join(result.artifacts_dir, "provenance.json"))

    def test_deterministic_run_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec1 = RunSpec(results={"out_dir": tmpdir})
            spec2 = RunSpec(results={"out_dir": tmpdir})
            r1 = run(spec1)
            r2 = run(spec2)
            assert r1.run_id == r2.run_id


class TestRunMany:
    def test_sequential(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            specs = [
                RunSpec(
                    execution={"parallel_pool_size": i},
                    results={"out_dir": tmpdir},
                )
                for i in range(3)
            ]
            results = run_many(specs, max_workers=1)
            assert len(results) == 3
            ids = [r.run_id for r in results]
            assert len(set(ids)) == 3  # All unique


class TestSweep:
    def test_returns_expanded_specs(self):
        base = RunSpec()
        sw = SweepSpec(axes=[SweepAxis(mode="grid", params={"execution.parallel_pool_size": [1, 2, 4]})])
        specs = sweep(base, sw)
        assert len(specs) == 3

    def test_empty_sweep(self):
        base = RunSpec()
        sw = SweepSpec()
        specs = sweep(base, sw)
        assert len(specs) == 1


class TestApiLoadSave:
    def test_load_spec_via_api(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
            yaml.dump({"execution": {"verbose": True}}, fh)
            path = fh.name
        try:
            spec = load_spec(path)
            assert spec.execution.verbose is True
        finally:
            os.unlink(path)

    def test_save_resolved_via_api(self):
        spec = RunSpec()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = save_resolved(spec, os.path.join(tmpdir, "spec.json"))
            assert out.exists()
