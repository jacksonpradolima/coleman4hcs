"""Owned synthetic wheels exercise real bounded inspection; no foreign code is imported or installed."""

import io
import zipfile

import pytest

from coleman.corpus.materialization_contracts import PackageMember
from coleman.transports.python_contracts import PythonWheel
from coleman.transports.python_wheels import validate_python_wheel
from tests.corpus.test_source_views import changed, digest, source_spec


def owned_wheel(*, extras=(), metadata=None, declaration=None):
    """Create authored wheel-shaped bytes; this is a parser fixture, not an installable distribution claim."""
    entries = {
        "owned_pytest-1.dist-info/METADATA": metadata or b"Metadata-Version: 2.1\nName: pytest\nVersion: 1\n\n",
        "owned_pytest-1.dist-info/WHEEL": declaration
        or b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n",
        "owned_pytest-1.dist-info/LICENSE": b"Owned schema-only notice, not foreign legal evidence.\n",
        **dict(extras),
    }
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(entries.items()):
            archive.writestr(name, payload)
    payload = target.getvalue()
    package = source_spec("owned_pytest-1-py3-none-any.whl", payload)
    package = changed(package, source=changed(package.source, version="1", uri="urn:owned:pytest-1-py3-none-any.whl"))
    pins = tuple(
        PackageMember(name=name, sha256=digest(data), size_bytes=len(data))
        for name, data in entries.items()
        if ".dist-info/" in name
    )
    wheel = PythonWheel.model_validate(
        {
            "distribution": {"name": "pytest", "version": "1"},
            "package": package,
            "members": pins,
            "entry_count": len(entries),
            "decoded_bytes": sum(len(data) for data in entries.values()),
            "components": [
                {
                    "component": "owned-synthetic-notice",
                    "scope": "All owned fixture bytes",
                    "license": changed(
                        package.license,
                        redistribution="recipe_only",
                        terms_sha256=digest(entries["owned_pytest-1.dist-info/LICENSE"]),
                    ),
                    "notice_members": ["owned_pytest-1.dist-info/LICENSE"],
                    "additional_notices": [],
                    "attribution": "declared_upstream",
                    "citation": "Owned parser fixture, not foreign licensing evidence",
                }
            ],
        }
    )
    return wheel, payload


def test_whole_opaque_wheel_uses_existing_selection_without_expanding_complete_package_limit(tmp_path):
    wheel, payload = owned_wheel(extras=tuple((f"owned/file{i}.py", b"# Owned bytes\n") for i in range(300)))
    path = tmp_path / "owned.whl"
    path.write_bytes(payload)
    assert wheel.entry_count == 303
    assert len(wheel.package.members) == 1
    assert len(wheel.members) == 3
    validate_python_wheel(wheel, path)
    assert path.read_bytes() == payload
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "metadata",
    [
        b"Name: other\nVersion: 1\n\n",
        b"Name: pytest\nVersion: 2\n\n",
        b"Name: pytest\nName: pytest\nVersion: 1\n\n",
        b"Name: pytest\nVersion: 1\nVersion: 1\n\n",
    ],
)
def test_actual_distribution_metadata_must_match_declared_identity(tmp_path, metadata):
    wheel, payload = owned_wheel(metadata=metadata)
    path = tmp_path / "owned.whl"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="name/version|requires one"):
        validate_python_wheel(wheel, path)


@pytest.mark.parametrize(
    "declaration",
    [
        b"Wheel-Version: 2.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n",
        b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: py3-none-any\n\n",
        b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: cp38-linux-aarch64\n\n",
        b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\n\n",
    ],
)
def test_future_native_or_ambiguous_wheel_formats_reject(tmp_path, declaration):
    wheel, payload = owned_wheel(declaration=declaration)
    path = tmp_path / "owned.whl"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="purelib|pure-Python"):
        validate_python_wheel(wheel, path)


@pytest.mark.parametrize("name", ["../escape.py", "owned.pth", "owned.data/scripts/cmd", "owned/native.so"])
def test_unsupported_installation_paths_reject_before_host_extraction(tmp_path, name):
    wheel, payload = owned_wheel(extras=((name, b"Owned bytes\n"),))
    path = tmp_path / "owned.whl"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="member name|installation hooks"):
        validate_python_wheel(wheel, path)
    assert list(tmp_path.iterdir()) == [path]


def test_raw_tamper_and_whole_archive_accounting_reject(tmp_path):
    wheel, payload = owned_wheel()
    path = tmp_path / "owned.whl"
    path.write_bytes(payload + b"x")
    with pytest.raises(ValueError, match="frozen size"):
        validate_python_wheel(wheel, path)
    path.write_bytes(payload)
    wrong_size = changed(wheel, decoded_bytes=wheel.decoded_bytes + 1)
    with pytest.raises(ValueError, match="accounting"):
        validate_python_wheel(wrong_size, path)
    wrong_count = changed(wheel, entry_count=wheel.entry_count + 1)
    with pytest.raises(ValueError, match="accounting"):
        validate_python_wheel(wrong_count, path)
    metadata = next(member for member in wheel.members if member.name.endswith("/METADATA"))
    changed_member = changed(metadata, sha256=digest(b"different metadata"))
    wrong_metadata = changed(
        wheel, members=tuple(changed_member if item == metadata else item for item in wheel.members)
    )
    with pytest.raises(ValueError, match="metadata disagrees"):
        validate_python_wheel(wrong_metadata, path)


def test_unknown_or_unpinned_component_notices_cannot_be_claimed():
    wheel, _ = owned_wheel()
    component = wheel.components[0]
    unpinned = changed(component, notice_members=("missing/LICENSE",))
    with pytest.raises(ValueError, match="independently pinned"):
        changed(wheel, components=(unpinned,))
    pending = changed(component.license, status="pending")
    with pytest.raises(ValueError, match="recipe-only"):
        changed(component, license=pending)


def test_original_wheel_script_requires_an_explicit_independent_pin(tmp_path):
    name = "owned_pytest-1.data/scripts/owned.py"
    script = b"# Owned script; never execute this parser fixture\n"
    wheel, payload = owned_wheel(extras=((name, script),))
    path = tmp_path / "owned.whl"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="installation hooks"):
        validate_python_wheel(wheel, path)
    pin = PackageMember(name=name, sha256=digest(script), size_bytes=len(script))
    declared = changed(wheel, members=(*wheel.members, pin), installation_scripts=(name,))
    validate_python_wheel(declared, path)
    assert path.read_bytes() == payload
