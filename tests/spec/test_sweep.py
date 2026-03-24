"""Tests for the sweep engine."""

from coleman4hcs.spec.models import RunSpec
from coleman4hcs.spec.run_id import compute_run_id
from coleman4hcs.spec.sweep import SweepAxis, SweepSpec, _expand_axis, _set_nested, expand_sweep


class TestSetNested:
    def test_simple_key(self):
        d: dict = {}
        _set_nested(d, "key", 42)
        assert d == {"key": 42}

    def test_nested_key(self):
        d: dict = {}
        _set_nested(d, "a.b.c", "val")
        assert d == {"a": {"b": {"c": "val"}}}

    def test_overwrite_existing(self):
        d = {"a": {"b": 1}}
        _set_nested(d, "a.b", 2)
        assert d["a"]["b"] == 2


class TestExpandAxis:
    def test_grid_single_param(self):
        axis = SweepAxis(mode="grid", params={"alpha": [0.1, 0.3, 0.5]})
        result = _expand_axis(axis)
        assert len(result) == 3
        assert result[0] == {"alpha": 0.1}
        assert result[2] == {"alpha": 0.5}

    def test_grid_two_params(self):
        axis = SweepAxis(mode="grid", params={"alpha": [0.1, 0.3], "seed": [0, 1, 2]})
        result = _expand_axis(axis)
        assert len(result) == 6  # 2 x 3

    def test_zip_mode(self):
        axis = SweepAxis(mode="zip", params={"window": [10, 50, 100], "decay": [0.9, 0.95, 0.99]})
        result = _expand_axis(axis)
        assert len(result) == 3
        assert result[0] == {"decay": 0.9, "window": 10}
        assert result[2] == {"decay": 0.99, "window": 100}

    def test_grid_stable_order(self):
        axis = SweepAxis(mode="grid", params={"b": [1, 2], "a": [10, 20]})
        r1 = _expand_axis(axis)
        r2 = _expand_axis(axis)
        assert r1 == r2
        # Sorted by key name: 'a' before 'b'
        assert r1[0] == {"a": 10, "b": 1}


class TestExpandSweep:
    def test_no_axes_returns_copy(self):
        base = RunSpec()
        sweep = SweepSpec(axes=[])
        result = expand_sweep(base, sweep)
        assert len(result) == 1
        assert compute_run_id(result[0]) == compute_run_id(base)

    def test_grid_expansion(self):
        base = RunSpec()
        sweep = SweepSpec(axes=[SweepAxis(mode="grid", params={"algorithm.ucb.timerank.c": [0.1, 0.3, 0.5]})])
        result = expand_sweep(base, sweep)
        assert len(result) == 3
        assert result[0].algorithm["ucb"]["timerank"]["c"] == 0.1
        assert result[2].algorithm["ucb"]["timerank"]["c"] == 0.5

    def test_deterministic_order(self):
        base = RunSpec()
        sweep = SweepSpec(axes=[SweepAxis(mode="grid", params={"execution.parallel_pool_size": [1, 2, 4]})])
        r1 = expand_sweep(base, sweep)
        r2 = expand_sweep(base, sweep)
        ids1 = [compute_run_id(s) for s in r1]
        ids2 = [compute_run_id(s) for s in r2]
        assert ids1 == ids2

    def test_seed_replication(self):
        base = RunSpec()
        sweep = SweepSpec(
            axes=[SweepAxis(mode="grid", params={"execution.parallel_pool_size": [1, 2]})],
            seeds=[0, 1, 2],
        )
        result = expand_sweep(base, sweep)
        # 2 specs x 3 seeds = 6
        assert len(result) == 6

    def test_multi_axis_cartesian(self):
        base = RunSpec()
        sweep = SweepSpec(
            axes=[
                SweepAxis(mode="grid", params={"execution.parallel_pool_size": [1, 2]}),
                SweepAxis(mode="grid", params={"results.batch_size": [100, 200]}),
            ]
        )
        result = expand_sweep(base, sweep)
        # 2 x 2 = 4
        assert len(result) == 4

    def test_each_spec_unique_run_id(self):
        base = RunSpec()
        sweep = SweepSpec(axes=[SweepAxis(mode="grid", params={"execution.parallel_pool_size": [1, 2, 4]})])
        result = expand_sweep(base, sweep)
        ids = [compute_run_id(s) for s in result]
        assert len(set(ids)) == 3
