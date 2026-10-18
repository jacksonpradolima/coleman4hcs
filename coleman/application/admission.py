"""Explicit original-package raw-capture verification for corpus admission.

The corpus domain depends only on a typed verifier protocol. This application
adapter invokes a caller-selected, already installed Python in isolated mode
and reuses that package's existing offline capture validators. It never installs
packages, fetches sources, pulls images or executes a subject test program.
Original captures keep their package/version identity rather than being retagged
to the current release's conservative whole-SDK producer fingerprint.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from email.parser import BytesParser
from importlib.resources import files
from pathlib import Path
from typing import Literal

from coleman.corpus.admission_contracts import ProducerVerification
from coleman.corpus.controlled_contracts import ControlledDataset
from coleman.corpus.materialization import SourceCache, validate_source_storage

_SHA256_PREFIX = "sha256:"

DRIVER = r"""
import hashlib,importlib.metadata,json,platform,sys
from importlib.resources import files
from pathlib import Path
from coleman.corpus.controlled_contracts import ControlledDataset
from coleman.corpus.materialization import SourceCache
from coleman.transports.producer_identity import installed_producer_identity,package_pins
config=json.loads(Path(sys.argv[1]).read_text())
sdk=Path(str(files('coleman'))).resolve()
if not sdk.is_relative_to(Path(sys.prefix).resolve()):
 raise ValueError('Original verifier must use an installed wheel outside the source checkout')
if (importlib.metadata.version('coleman')!=config['package_version']
 or package_pins(files('coleman'))!=config['sdk_pins']):
 raise ValueError('Original installed SDK differs from its pinned wheel bytes/version')
dataset=ControlledDataset.from_json(Path(config['dataset']).read_bytes())
cache=SourceCache(config['cache'])
producer=installed_producer_identity()
if config['family']=='junit_matrix':
 from coleman.transports.junit_matrix import validate_junit_matrix
 from coleman.transports.junit_matrix_contracts import JunitMatrixReceipt
 receipt=JunitMatrixReceipt.from_json(Path(config['receipt']).read_bytes())
 validate_junit_matrix(receipt,cache,Path(config['output']))
 if dataset.fingerprint not in {cell.dataset_fingerprint for cell in receipt.cells}:
  raise ValueError('Controlled source is not a member of the verified original matrix')
 actual=Path(config['output'])/'corpus-controlled'/(dataset.fingerprint.removeprefix('sha256:')+'.json')
 if ControlledDataset.from_json(actual.read_bytes())!=dataset:
  raise ValueError('Controlled dataset differs from verified original matrix evidence')
 plan_fingerprint=receipt.plan.fingerprint
 if receipt.producer_sha256!=producer:
  raise ValueError('Original matrix producer identity does not match installed SDK')
elif config['family']=='python':
 from coleman.transports.python_capture import validate_python_capture
 from coleman.transports.python_contracts import PythonCapturePlan,PythonCaptureReceipt
 receipt=PythonCaptureReceipt.from_json(Path(config['receipt']).read_bytes())
 plan=PythonCapturePlan.from_json(Path(config['plan']).read_bytes())
 validate_python_capture(receipt,plan,dataset,cache)
 plan_fingerprint=plan.fingerprint
else:
 from coleman.transports.c_capture import validate_c_capture
 from coleman.transports.c_contracts import CCapturePlan,CCaptureReceipt
 receipt=CCaptureReceipt.from_json(Path(config['receipt']).read_bytes())
 plan=CCapturePlan.from_json(Path(config['plan']).read_bytes())
 validate_c_capture(receipt,plan,dataset,cache)
 plan_fingerprint=plan.fingerprint
members={m.name:m.sha256 for m in dataset.spec.report_materialization.spec.members}
packets=[members[role+'-capture.json'] for role in ('buggy','fixed')]
deps=[{'name':d.metadata['Name'],'version':d.version} for d in importlib.metadata.distributions()]
deps.sort(key=lambda x:x['name'])
print(json.dumps(dict(payload_fingerprint=dataset.fingerprint,producer_sha256=producer,
 capture_receipt_fingerprint=receipt.fingerprint,capture_plan_fingerprint=plan_fingerprint,
 raw_packets=packets,python_version=platform.python_version(),dependencies=deps),sort_keys=True))
"""


def _wheel_metadata(archive: zipfile.ZipFile) -> tuple[str, tuple[str, ...]]:
    """Validate bounded unambiguous wheel metadata without extracting or importing archive content."""
    names = tuple(archive.namelist())
    if len(set(names)) != len(names):
        raise ValueError("Original verifier wheel has duplicate members")
    metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
    if len(metadata_names) != 1 or sum(item.file_size for item in archive.infolist()) > 64 * 1024**2:
        raise ValueError("Original verifier wheel metadata/decoded bytes exceed supported bounds")
    metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
    if metadata["Name"] != "coleman" or not metadata["Version"]:
        raise ValueError("Original verifier wheel must identify a specific COLEMAN version")
    return str(metadata["Version"]), names


def _wheel_sdk_pins(archive: zipfile.ZipFile, names: tuple[str, ...]) -> dict[str, str]:
    """Hash only installed authored package resources; a missing package cannot become a valid original SDK."""
    pins = {
        name: _SHA256_PREFIX + hashlib.sha256(archive.read(name)).hexdigest()
        for name in names
        if name.startswith("coleman/") and not name.endswith("/")
    }
    if not pins:
        raise ValueError("Original verifier wheel contains no authored SDK files")
    return pins


@dataclass(frozen=True)
class OriginalCaptureVerifier:
    """Bind an explicit original installed interpreter/wheel and source-specific capture receipt.

    Wheel SHA-256 is an operator-reviewed pin, not a trust decision inferred from
    a filename. Package URI is descriptive and never opened. Python/receipt/plan
    paths are local verification bindings and never enter portable proof identity.
    Java matrix output contains its original controlled datasets; Python/C plans
    remain separate original versioned artifacts. No fallback to another package
    or silent legacy migration is permitted. Offline validators may republish
    identical owned cache metadata through their existing materializers.
    """

    family: Literal["junit_matrix", "python", "c"]
    python: Path
    wheel: Path
    wheel_sha256: str
    package_uri: str
    receipt: Path
    plan: Path | None = None
    output: Path | None = None
    timeout_seconds: int = 300

    def verify(self, payload: ControlledDataset, cache: SourceCache) -> ProducerVerification:
        """Recheck original SDK bytes/version and original raw-wire/source/context correspondence offline.

        Uses a fixed script with argument-list subprocess invocation, isolated
        interpreter imports and no caller-provided code. Source data remains in
        temporary/cache locations outside Git. Wrong/missing package pins,
        families, receipt/plan/packets, normalized outcomes or unknown original
        producers reject with ValueError; filesystem/subprocess timeout errors
        propagate. No Docker/network/subject execution or result rewrite occurs.
        """
        payload = ControlledDataset.model_validate(payload)
        self._validate_bindings()
        version, pins, digest, size = self._wheel_identity()
        with tempfile.TemporaryDirectory(prefix="coleman-original-capture-verifier-") as temporary:
            root = Path(temporary)
            validate_source_storage(root)
            dataset = root / "dataset.json"
            dataset.write_text(payload.to_json() + "\n")
            config = root / "verification.json"
            config.write_text(
                json.dumps(
                    {
                        "family": self.family,
                        "package_version": version,
                        "sdk_pins": pins,
                        "dataset": str(dataset),
                        "cache": str(cache.root),
                        "receipt": str(self.receipt.resolve()),
                        "plan": str(self.plan.resolve()) if self.plan is not None else None,
                        "output": str(self.output.resolve()) if self.output is not None else None,
                    },
                    sort_keys=True,
                )
            )
            environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
            result = subprocess.run(
                [str(self.python.absolute()), "-I", "-c", DRIVER, str(config)],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        if result.returncode != 0:
            raise ValueError("Original capture verification failed: " + result.stderr[-4096:])
        if len(result.stdout.encode("utf-8")) > 1024**2:
            raise ValueError("Original capture verifier output exceeds 1 MiB")
        proof = json.loads(result.stdout.strip().splitlines()[-1])
        return ProducerVerification.model_validate(
            {
                **proof,
                "schema_version": 1,
                "package": {
                    "uri": self.package_uri,
                    "version": version,
                    "citation": "Pinned original COLEMAN wheel and installed SDK for offline raw-capture verification",
                    "sha256": digest,
                    "size_bytes": size,
                    "availability": "verified_local",
                },
                "verifier_sha256": _SHA256_PREFIX
                + hashlib.sha256(files("coleman.application").joinpath("admission.py").read_bytes()).hexdigest(),
            }
        )

    def _validate_bindings(self) -> None:
        """Reject unsupported families and ambiguous optional path/timeout semantics before starting a process."""
        if self.family not in {"junit_matrix", "python", "c"}:
            raise ValueError("Unsupported original capture verifier family")
        if type(self.timeout_seconds) is not int or not 1 <= self.timeout_seconds <= 900:
            raise ValueError("Original capture verification timeout must be integral seconds in [1, 900]")
        if self.family == "junit_matrix":
            if self.output is None or self.plan is not None:
                raise ValueError("Original Java matrix verification requires its output tree and embedded plan")
        elif self.plan is None or self.output is not None:
            raise ValueError("Original Python/C verification requires its separate original plan")

    def _wheel_identity(self) -> tuple[str, dict[str, str], str, int]:
        """Pin wheel bytes and every authored SDK resource; code is never loaded from the wheel in-process."""
        size = self.wheel.stat().st_size
        if not 1 <= size <= 32 * 1024**2:
            raise ValueError("Original verifier wheel exceeds its 32-MiB bound")
        encoded = self.wheel.read_bytes()
        digest = _SHA256_PREFIX + hashlib.sha256(encoded).hexdigest()
        if digest != self.wheel_sha256:
            raise ValueError("Original verifier wheel SHA-256 disagrees with its reviewed pin")
        with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
            version, names = _wheel_metadata(archive)
            pins = _wheel_sdk_pins(archive, names)
            return version, pins, digest, size
