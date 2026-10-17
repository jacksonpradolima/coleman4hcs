"""Read-only wheel preparation through existing bounded ZIP selection, never host installation."""

from __future__ import annotations

import hashlib
import re
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path

from coleman.corpus.archives import consume_selected_zip_members
from coleman.corpus.materialization_contracts import ByteReader, PackageMember
from coleman.transports.python_contracts import PythonWheel


def _field(payload: bytes, name: str) -> str:
    """Require one unambiguous metadata field; malformed/duplicate declarations raise ValueError."""
    message = BytesParser(policy=policy.default).parsebytes(payload)
    values = message.get_all(name, [])
    if message.defects or len(values) != 1:
        raise ValueError("Python wheel metadata requires one " + name)
    return str(values[0])


def validate_python_wheel(wheel: PythonWheel, path: Path) -> None:
    """Verify whole raw wheel and selected metadata before staging an offline container install.

    The whole source SHA-256 and byte count bind all original wheel bytes. The
    existing selection reader checks bounded central/local records, safe unique
    names and types for every entry, and CRC/deflate/content pins for selected
    notices/metadata. Unselected member bodies are not independently certified.
    No member is extracted or executed on the host. Original .data/scripts Python
    members require explicit pins; pip may install them but never executes them.
    Other .data/.pth hooks, compiled wheels, ZIP64 and ambiguous metadata reject.
    Read errors propagate. This is integrity validation, not a code safety claim.
    """
    wheel = PythonWheel.model_validate(wheel)
    expected = wheel.package.source
    sha = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        while chunk := stream.read(65536):
            total += len(chunk)
            if total > (expected.size_bytes or 0):
                raise ValueError("Python wheel source exceeds its frozen size")
            sha.update(chunk)
    if total != expected.size_bytes or "sha256:" + sha.hexdigest() != expected.sha256:
        raise ValueError("Python wheel source disagrees with frozen bytes")
    selected = {}

    def consume(member: PackageMember, reader: ByteReader) -> None:
        """Bound and checksum selected metadata/notices; never allocate an unbounded member."""
        payload = reader.read(member.size_bytes + 1)
        if len(payload) != member.size_bytes or "sha256:" + hashlib.sha256(payload).hexdigest() != member.sha256:
            raise ValueError("Python wheel selected metadata disagrees with its pin")
        selected[member.name] = payload

    consume_selected_zip_members(path, wheel.members, consume, maximum_entries=1024, maximum_metadata_bytes=1024**2)
    _archive_accounting(wheel, path)
    _distribution_metadata(wheel, selected)


def _archive_accounting(wheel: PythonWheel, path: Path) -> None:
    """After bounded structural inspection, conserve whole-wheel sizes and reviewed installation paths."""
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) != wheel.entry_count or sum(info.file_size for info in entries) != wheel.decoded_bytes:
            raise ValueError("Python wheel whole-archive accounting disagrees with the recipe")
        if any(
            info.filename.endswith((".pth", ".so", ".pyd", ".dll"))
            or (
                any(part.endswith(".data") for part in info.filename.split("/"))
                and info.filename not in wheel.installation_scripts
            )
            for info in entries
        ):
            raise ValueError("Python wheel contains unsupported native/data/path installation hooks")


def _distribution_metadata(wheel: PythonWheel, selected: dict[str, bytes]) -> None:
    """Require actual same-directory name/version and pure-Python wheel tags; no dependency code executes."""
    metadata_name = next(name for name in selected if name.endswith(".dist-info/METADATA"))
    wheel_name = next(name for name in selected if name.endswith(".dist-info/WHEEL"))
    if metadata_name.rsplit("/", 1)[0] != wheel_name.rsplit("/", 1)[0]:
        raise ValueError("Python wheel metadata directories disagree")
    script_prefix = metadata_name.removesuffix(".dist-info/METADATA") + ".data/scripts/"
    if any(
        not name.startswith(script_prefix) or "/" in name.removeprefix(script_prefix)
        for name in wheel.installation_scripts
    ):
        raise ValueError("Python wheel script paths disagree with their distribution directory")
    metadata = selected[metadata_name]
    normalized_name = re.sub(r"[-_.]+", "-", _field(metadata, "Name")).lower()
    if (normalized_name, _field(metadata, "Version")) != (wheel.distribution.name, wheel.distribution.version):
        raise ValueError("Python wheel actual name/version disagrees with the frozen distribution")
    declaration = selected[wheel_name]
    if _field(declaration, "Wheel-Version") != "1.0" or _field(declaration, "Root-Is-Purelib").lower() != "true":
        raise ValueError("Python wheel requires purelib wheel format 1.0")
    tags = BytesParser(policy=policy.default).parsebytes(declaration).get_all("Tag", [])
    if not tags or any(re.fullmatch(r"py[23](?:\.py[23])?-none-any", str(tag)) is None for tag in tags):
        raise ValueError("Python wheel requires only portable pure-Python tags")
