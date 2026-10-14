"""Execute the local release-note selector without publishing or credentials."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "version,has_notes,success",
    [
        ("3.6.1", True, True),
        ("4.0.0", False, True),
        ("../../outside", False, False),
        ("3.6.1; touch injected", False, False),
    ],
)
def test_release_notes_require_a_safe_version_and_preserve_generated_fallback(tmp_path, version, has_notes, success):
    with (ROOT / ".github/workflows/release.yml").open() as stream:
        workflow = yaml.safe_load(stream)
    step = next(step for step in workflow["jobs"]["release"]["steps"] if step.get("id") == "release_notes")
    notes = tmp_path / "docs/releases/3.6.1.md"
    notes.parent.mkdir(parents=True)
    if has_notes:
        notes.write_text("Reviewed public release description.\n")
    output = tmp_path / "outputs"
    output.write_text("")
    completed = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        cwd=tmp_path,
        check=False,
        env={**os.environ, "RELEASE_VERSION": version, "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
    )
    assert (completed.returncode == 0) == success
    assert output.read_text() == ("path=docs/releases/3.6.1.md\n" if has_notes else "")
    assert not (tmp_path / "injected").exists()
