"""Noneditable base-wheel benchmarks and real TCPCI execution outside the source checkout."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from coleman.benchmarks import BenchmarkReport

ROOT = Path(__file__).parents[2]


def command(args, cwd, env):
    return subprocess.run(args, cwd=cwd, env=env, check=True, capture_output=True, text=True, timeout=240)


def test_installed_base_wheel_repeats_mathematics_and_native_ci_without_component_mocks(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    env = {**os.environ, "UV_LINK_MODE": "copy", "UV_NO_PROGRESS": "1"}
    distributions = tmp_path / "dist"
    command([uv, "build", "--out-dir", str(distributions)], ROOT, env)
    virtual = tmp_path / "installed"
    command([uv, "venv", str(virtual), "--python", sys.executable], tmp_path, env)
    python = virtual / "bin/python"
    wheel = next(distributions.glob("*.whl"))
    command([uv, "pip", "install", "--python", str(python), str(wheel)], tmp_path, env)
    outside = tmp_path / "outside"
    outside.mkdir()
    identity = command(
        [
            str(python),
            "-c",
            "from pathlib import Path; import coleman; print(Path(coleman.__file__).resolve())",
        ],
        outside,
        env,
    )
    assert Path(identity.stdout.strip()).is_relative_to(virtual)
    assert not outside.is_relative_to(ROOT)
    artifacts = []
    for _ in range(2):
        output = command(
            [
                str(python),
                "-m",
                "coleman.benchmarks",
                "--profile",
                "correctness",
                "--output",
                "benchmark-output",
            ],
            outside,
            env,
        )
        payload = json.loads(output.stdout)
        assert payload["status"] == "complete"
        assert payload["reproducibility"] == "verified_repeated_labels"
        artifacts.append(BenchmarkReport.from_json(Path(payload["artifact"]).read_bytes()))
    assert artifacts[0].mathematical_identity == artifacts[1].mathematical_identity
    assert artifacts[0].environment.source_commit is None
    assert all(row.summary.absolute_regret == 0 for row in artifacts[0].observations if row.summary.evidence == "exact")
    script = outside / "ci.py"
    shutil.copyfile(ROOT / "examples/finite-coverage/ci.py", script)
    outputs = []
    for _ in range(2):
        result = command([str(python), str(script), "native-ci"], outside, env)
        outputs.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert outputs[0] == outputs[1]
    assert outputs[0]["native_scheduled"] == ["A", "C"]
    assert outputs[0]["objective"] == "1"
    reports = list((outside / "native-ci").rglob("benchmark-reports/*.json"))
    assert len(reports) == 4
    grouped = {}
    for path in reports:
        report = BenchmarkReport.from_json(path.read_bytes())
        assert report.status == "complete"
        assert len(report.observations) == 3
        assert all(row.summary.selected_ids == ("A", "C") for row in report.observations)
        method = report.observations[0].method
        grouped.setdefault(method, set()).add(report.mathematical_identity)
    assert set(grouped) == {"greedy_density_v1", "branch_bound_v1"}
    assert all(len(values) == 1 for values in grouped.values())
    no_transport = command(
        [
            str(python),
            "-c",
            "import sys; import coleman.benchmarks; assert 'coleman.cli' not in sys.modules; "
            "assert not any(name=='mcp' or name.startswith('mcp.') for name in sys.modules)",
        ],
        outside,
        env,
    )
    assert no_transport.returncode == 0
