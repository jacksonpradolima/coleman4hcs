"""Tests for the CLI module."""

import os
import tempfile

import yaml

from coleman.cli import _parse_kv, main


class TestParseKv:
    def test_simple_values(self):
        key, vals = _parse_kv("alpha=0.1,0.3,0.5")
        assert key == "alpha"
        assert vals == [0.1, 0.3, 0.5]

    def test_integer_values(self):
        key, vals = _parse_kv("seed=1,2,3")
        assert key == "seed"
        assert vals == [1, 2, 3]

    def test_range_expression(self):
        key, vals = _parse_kv("seed=range(0,5)")
        assert key == "seed"
        assert vals == [0, 1, 2, 3, 4]

    def test_range_with_step(self):
        key, vals = _parse_kv("seed=range(0,10,2)")
        assert key == "seed"
        assert vals == [0, 2, 4, 6, 8]

    def test_string_values(self):
        key, vals = _parse_kv("policy=Random,Greedy")
        assert key == "policy"
        assert vals == ["Random", "Greedy"]


class TestCLIValidate:
    def test_validate_valid_config(self, capsys):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
            yaml.dump({"execution": {"verbose": True}}, fh)
            path = fh.name
        try:
            main(["validate", "--config", path])
            captured = capsys.readouterr()
            assert "VALID" in captured.out
            assert "run_id=" in captured.out
        finally:
            os.unlink(path)

    def test_validate_with_resolve(self, capsys):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
            yaml.dump({"execution": {"parallel_pool_size": 3}}, fh)
            path = fh.name
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                out_path = os.path.join(tmpdir, "resolved.json")
                main(["validate", "--config", path, "--resolve", out_path])
                assert os.path.exists(out_path)
        finally:
            os.unlink(path)


class TestCLIRun:
    def test_run_command(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "run.yaml")
            with open(cfg_path, "w") as fh:
                yaml.dump(
                    {
                        "execution": {
                            "parallel_pool_size": 1,
                            "independent_executions": 1,
                            "verbose": False,
                        },
                        "experiment": {
                            "scheduled_time_ratio": [0.1],
                            "datasets_dir": "examples",
                            "datasets": ["fakedata"],
                            "rewards": ["RNFail"],
                            "policies": ["Random"],
                        },
                        "results": {"out_dir": tmpdir},
                    },
                    fh,
                )
            main(["run", "--config", cfg_path])
            captured = capsys.readouterr()
            assert "run_id:" in captured.out


class TestCLISweep:
    def test_sweep_dry_run(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "base.yaml")
            with open(cfg_path, "w") as fh:
                yaml.dump(
                    {
                        "execution": {
                            "parallel_pool_size": 1,
                            "independent_executions": 1,
                            "verbose": False,
                        },
                        "experiment": {
                            "scheduled_time_ratio": [0.1],
                            "datasets_dir": "examples",
                            "datasets": ["fakedata"],
                            "rewards": ["RNFail"],
                            "policies": ["Random"],
                        },
                        "results": {"out_dir": tmpdir},
                    },
                    fh,
                )
            main(
                [
                    "sweep",
                    "--config",
                    cfg_path,
                    "--grid",
                    "execution.parallel_pool_size=1,2,4",
                    "--dry-run",
                ]
            )
            captured = capsys.readouterr()
            assert "Generated 3 specs" in captured.out
            assert "run_id=" in captured.out

    def test_sweep_multiple_grid_flags(self, capsys):
        """Multiple --grid flags accumulate all sweep dimensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "base.yaml")
            with open(cfg_path, "w") as fh:
                yaml.dump(
                    {
                        "execution": {
                            "parallel_pool_size": 1,
                            "independent_executions": 1,
                        },
                        "experiment": {
                            "scheduled_time_ratio": [0.1],
                            "datasets_dir": "examples",
                            "datasets": ["fakedata"],
                            "rewards": ["RNFail"],
                            "policies": ["Random"],
                        },
                        "results": {"out_dir": tmpdir},
                    },
                    fh,
                )
            main(
                [
                    "sweep",
                    "--config",
                    cfg_path,
                    "--grid",
                    "execution.seed=1,2",
                    "--grid",
                    "execution.parallel_pool_size=1,2",
                    "--dry-run",
                ]
            )
            captured = capsys.readouterr()
            # 2 seeds × 2 pool sizes = 4 specs
            assert "Generated 4 specs" in captured.out
