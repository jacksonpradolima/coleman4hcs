"""Tests for deterministic run_id computation."""

from coleman4hcs.spec.models import RunSpec
from coleman4hcs.spec.run_id import _canonical_json, compute_run_id


class TestCanonicalJson:
    def test_deterministic(self):
        spec = RunSpec()
        assert _canonical_json(spec) == _canonical_json(spec)

    def test_sorted_keys(self):
        spec = RunSpec()
        cj = _canonical_json(spec)
        assert '"algorithm"' in cj
        # Keys should appear in sorted order
        idx_algo = cj.index('"algorithm"')
        idx_exec = cj.index('"execution"')
        assert idx_algo < idx_exec


class TestComputeRunId:
    def test_length(self):
        spec = RunSpec()
        rid = compute_run_id(spec)
        assert len(rid) == 12

    def test_hex_chars(self):
        spec = RunSpec()
        rid = compute_run_id(spec)
        assert all(c in "0123456789abcdef" for c in rid)

    def test_same_spec_same_id(self):
        spec1 = RunSpec()
        spec2 = RunSpec()
        assert compute_run_id(spec1) == compute_run_id(spec2)

    def test_different_spec_different_id(self):
        spec1 = RunSpec()
        spec2 = RunSpec(execution={"parallel_pool_size": 999})
        assert compute_run_id(spec1) != compute_run_id(spec2)

    def test_algorithm_params_affect_id(self):
        spec1 = RunSpec(algorithm={"ucb": {"timerank": {"c": 0.5}}})
        spec2 = RunSpec(algorithm={"ucb": {"timerank": {"c": 0.3}}})
        assert compute_run_id(spec1) != compute_run_id(spec2)

    def test_golden_determinism(self):
        """Same spec must always produce the same run_id (golden test).

        If this fails, the determinism contract is broken.
        """
        spec = RunSpec(
            execution={"parallel_pool_size": 1, "independent_executions": 1, "verbose": False},
            experiment={"datasets": ["test@proj"], "rewards": ["RNFail"], "policies": ["UCB"]},
            algorithm={"ucb": {"rnfail": {"c": 0.3}}},
            results={"enabled": False},
            checkpoint={"enabled": False},
            telemetry={"enabled": False},
        )
        # This pre-computed value must never change.
        golden_rid = "ddd8bbefa143"
        assert compute_run_id(spec) == golden_rid
        # Two independent constructions must match.
        spec_copy = RunSpec.model_validate(spec.model_dump())
        assert compute_run_id(spec_copy) == golden_rid
