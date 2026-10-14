"""Security-audit inventory covers all locked platforms and rejects unauditable sources."""

import tomllib
from pathlib import Path

import pytest

from scripts.audit_lock import audit_requirements

ROOT = Path(__file__).parents[1]


def test_actual_lock_inventory_includes_all_registry_packages_and_no_root():
    with (ROOT / "uv.lock").open("rb") as stream:
        locked = tomllib.load(stream)
    lines = audit_requirements(ROOT / "uv.lock").splitlines()
    expected = {f"{row['name']}=={row['version']}" for row in locked["package"] if row["name"] != "coleman"}
    assert set(lines) == expected
    assert len(lines) == len(expected)
    assert not any(line.startswith("coleman==") for line in lines)


@pytest.mark.parametrize(
    "source", ['{ git = "https://example.invalid/pkg" }', '{ registry = "https://example.invalid" }']
)
def test_unsupported_sources_fail_instead_of_disappearing(tmp_path, source):
    lock = tmp_path / "uv.lock"
    lock.write_text('version = 1\n[[package]]\nname = "example"\nversion = "1.0"\nsource = ' + source)
    with pytest.raises(ValueError, match="unsupported package source"):
        audit_requirements(lock)


def test_optional_and_platform_only_entries_are_not_filtered(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text("""version = 1
[[package]]
name = "coleman"
version = "3.6.0"
source = { editable = "." }
[[package]]
name = "windows-only"
version = "1.0"
source = { registry = "https://pypi.org/simple" }
resolution-markers = ["sys_platform == 'win32'"]
""")
    assert audit_requirements(lock) == "windows-only==1.0\n"


@pytest.mark.parametrize("version", ["true", "2", '"1"'])
def test_unknown_lock_versions_fail_closed(tmp_path, version):
    lock = tmp_path / "uv.lock"
    lock.write_text(f"version = {version}\n")
    with pytest.raises(ValueError, match="Unsupported uv lock schema"):
        audit_requirements(lock)


def test_empty_inventory_cannot_report_a_clean_audit(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text("""version = 1
package = []
""")
    with pytest.raises(ValueError, match="lacks root or registry dependencies"):
        audit_requirements(lock)
