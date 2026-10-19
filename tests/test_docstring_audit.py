"""Keep parameter coverage distinct from the mere presence of a docstring."""

import json
import subprocess
import sys
from pathlib import Path

from scripts.audit_docstrings import audit_paths


def test_one_line_docstring_does_not_certify_parameters(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text('def run(spec, *, limit=1):\n    """Execute a run."""\n    return spec\n')
    report = audit_paths([source])
    assert report["signatures_checked"] == 1
    assert report["findings"][0]["missing_parameters"] == ["limit", "spec"]


def test_numpy_parameters_cover_grouped_positional_keyword_and_variadic_names(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text('''def collect(first, /, second, *values, limit=1, **options):
    """Collect observations.

    Parameters
    ----------
    first, second : str
        Source identities.
    *values : float
        Observations.
    limit : int
        Bound.
    **options : object
        Explicit options.
    """
    return first
''')
    report = audit_paths([source, tmp_path])
    assert report["files"] == 1
    assert report["signatures_checked"] == 1
    assert report["findings"] == []


def test_class_parameters_cover_constructor_without_executing_foreign_module(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text('''raise RuntimeError("inspection must not execute me")
class Capture:
    """Capture observations.

    Parameters
    ----------
    source : str
        Pinned source.
    """
    def __init__(self, source):
        """Construct a capture."""
    def consume(self, *, packet):
        """Consume a packet."""
''')
    report = audit_paths([source])
    assert report["signatures_checked"] == 2
    assert [item["name"] for item in report["findings"]] == ["Capture.consume"]
    assert report["findings"][0]["missing_parameters"] == ["packet"]


def test_function_local_and_private_definitions_are_not_reported_as_public_api(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text('''def outer():
    def helper(secret):
        return secret
class _Internal:
    def expose(value):
        pass
async def receive(packet):
    """Receive a packet."""
def _private(value):
    pass
''')
    report = audit_paths([source])
    assert report["signatures_checked"] == 1
    assert [item["name"] for item in report["findings"]] == ["receive"]


def test_only_implicit_method_receiver_is_excluded(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text('''def bind(self):
    """A module function still requires documentation for this parameter."""
class Adapter:
    @staticmethod
    def emit(cls):
        """A static method has no implicit receiver."""
    def receive(receiver, packet):
        """Only the first positional method parameter is implicit."""
''')
    report = audit_paths([source])
    assert [(item["name"], item["missing_parameters"]) for item in report["findings"]] == [
        ("bind", ["self"]),
        ("Adapter.emit", ["cls"]),
        ("Adapter.receive", ["packet"]),
    ]


def test_cli_report_is_read_only_and_check_mode_is_explicit(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text('def run(spec):\n    """Run a specification."""\n')
    before = source.read_bytes()
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/audit_docstrings.py"), str(source)]
    report = subprocess.run(command, capture_output=True, text=True, check=False)
    checked = subprocess.run([*command, "--check"], capture_output=True, text=True, check=False)
    assert report.returncode == 0
    assert checked.returncode == 1
    assert json.loads(report.stdout) == json.loads(checked.stdout)
    assert source.read_bytes() == before
