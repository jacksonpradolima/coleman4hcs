"""Noneditable base-wheel corpus commands and a real native CI run outside the checkout."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from coleman.corpus import CorpusFixtureReport

ROOT = Path(__file__).parents[2]


def command(args, cwd, env, *, check=True):
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, check=check, timeout=240)


def test_installed_registry_correctness_fixtures_and_real_native_ci(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update({"UV_LINK_MODE": "copy", "UV_NO_PROGRESS": "1"})
    distributions = tmp_path / "dist"
    command([uv, "build", "--out-dir", str(distributions)], ROOT, env)
    virtual = tmp_path / "installed"
    command([uv, "venv", str(virtual), "--python", sys.executable], tmp_path, env)
    python = virtual / "bin/python"
    wheel = next(distributions.glob("*.whl"))
    command([uv, "pip", "install", "--offline", "--python", str(python), str(wheel)], tmp_path, env)
    outside = tmp_path / "outside"
    outside.mkdir()
    origin = command([str(python), "-c", "import coleman; print(coleman.__file__)"], outside, env)
    assert Path(origin.stdout.strip()).resolve().is_relative_to(virtual.resolve())
    command([str(python), "-c", "import importlib.util; assert importlib.util.find_spec('mcp') is None"], outside, env)
    registered = command([str(python), "-m", "coleman.corpus", "list"], outside, env)
    initial = json.loads(registered.stdout)
    assert len(initial["registered_sources"]) == 12
    assert len(initial["blocked_sources"]) == 10
    assert initial["evidence_status"] == "insufficient_diversity"
    assert len(initial["uncovered_required_regimes"]) == 7
    shown = command([str(python), "-m", "coleman.corpus", "show", "ci-serial-boundary-v1"], outside, env)
    assert json.loads(shown.stdout)["manifest"]["benchmark_id"] == "ci-serial-boundary-v1"
    schema = command([str(python), "-m", "coleman.corpus", "schema"], outside, env)
    assert json.loads(schema.stdout)["properties"]["schema_version"]["maximum"] == 1
    for source in ("ci-serial-boundary-v1", "ci-parallel-occupancy-v1"):
        reports = []
        for _ in range(2):
            result = command(
                [str(python), "-m", "coleman.corpus", "validate", source, "--output", "verified"], outside, env
            )
            payload = json.loads(result.stdout)
            assert payload["fixture_status"] == "verified"
            assert payload["generalization"] == "insufficient_evidence"
            reports.append(CorpusFixtureReport.from_json(Path(payload["artifact"]).read_bytes()))
        assert reports[0] == reports[1]
    rejected = command([str(python), "-m", "coleman.corpus", "validate", "../../arbitrary"], outside, env, check=False)
    assert rejected.returncode == 2
    escaped = command(
        [str(python), "-m", "coleman.corpus", "validate", "ci-serial-boundary-v1", "--output", "../escape"],
        outside,
        env,
        check=False,
    )
    assert escaped.returncode == 2
    assert not (tmp_path / "escape").exists()
    example = outside / "ci.py"
    shutil.copyfile(ROOT / "examples/corpus-foundation/ci.py", example)
    native = command([str(python), str(example), "native-output"], outside, env)
    final = json.loads(native.stdout.strip().splitlines()[-1])
    assert final["native_rows"] == 2
    assert final["fixture_sources_verified"] == 2
    assert final["evidence_status"] == "insufficient_diversity"
    assert final["uncovered_regimes"] == 7
    run_root = Path(final["artifacts_dir"])
    manifest = json.loads((run_root / "manifest.json").read_bytes())
    assert len([entry for entry in manifest["artifacts"] if entry["type"] == "corpus_fixture_report"]) == 2

    documentation = (ROOT / "docs/benchmark-corpus.md").read_text()
    snippet = re.search(r"```python\n(.*?)```", documentation, flags=re.S)
    assert snippet is not None
    command([str(python), "-c", snippet.group(1)], outside, env)
