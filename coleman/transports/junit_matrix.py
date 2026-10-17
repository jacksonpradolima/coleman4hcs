"""Optional bounded Java matrix transport, reusing existing source capture and controlled observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from typing import NamedTuple

from coleman.artifacts import ArtifactWriter
from coleman.corpus.controlled import validate_controlled_dataset
from coleman.corpus.controlled_contracts import ControlledDataset, ControlledElapsed, ControlledObservation
from coleman.corpus.materialization import SourceCache, validate_cache_separation, validate_source_storage
from coleman.results.domain import canonical_json
from coleman.transports.junit_capture import capture_junit_source
from coleman.transports.junit_contracts import JunitRoleCapture
from coleman.transports.junit_matrix_contracts import (
    JunitMatrixCell,
    JunitMatrixCellResult,
    JunitMatrixReceipt,
    JunitMatrixRecipe,
    JunitMatrixRole,
    JunitPairCounts,
)
from coleman.transports.junit_protocol import validate_role_bytes
from coleman.transports.junit_source import prepare_junit_inputs, validate_junit_report_budget

_COMPLETION = "complete.json"
_DATASET_BYTES = 16 * 1024**2
_HASH_PREFIX = "sha256:"
_JSON_SUFFIX = ".json"
_TRANSPORT_PACKAGE = "coleman.transports"
_CORPUS_PACKAGE = "coleman.corpus"
_PLAN_LIMIT_ERROR = "Java matrix plan exceeds its byte limit"


def _digest(payload: bytes) -> str:
    """Return the explicit SHA-256 identity of exact bytes, with no filesystem access."""
    return _HASH_PREFIX + hashlib.sha256(payload).hexdigest()


def junit_matrix_producer() -> str:
    """Bind matrix implementation/serialization bytes and package version; old capture provenance stays separate.

    A changed producer requires a new output generation. Unknown producers reject
    on validation rather than silently relabeling existing observations.
    """
    modules = (
        (_TRANSPORT_PACKAGE, "junit_matrix.py"),
        (_TRANSPORT_PACKAGE, "junit_matrix_contracts.py"),
        (_TRANSPORT_PACKAGE, "junit_capture.py"),
        (_TRANSPORT_PACKAGE, "junit_contracts.py"),
        (_TRANSPORT_PACKAGE, "junit_process.py"),
        (_TRANSPORT_PACKAGE, "junit_protocol.py"),
        (_TRANSPORT_PACKAGE, "junit_source.py"),
        (_CORPUS_PACKAGE, "controlled.py"),
        (_CORPUS_PACKAGE, "controlled_contracts.py"),
        (_CORPUS_PACKAGE, "source_views.py"),
        (_CORPUS_PACKAGE, "materialization.py"),
        (_CORPUS_PACKAGE, "materialization_contracts.py"),
        (_CORPUS_PACKAGE, "archives.py"),
        ("coleman.results", "versioned.py"),
        ("coleman.results", "domain.py"),
        ("coleman.selection", "contracts.py"),
        ("coleman", "artifacts.py"),
    )
    pins = {package + "/" + name: _digest(files(package).joinpath(name).read_bytes()) for package, name in modules}
    return _digest(canonical_json({"package_version": version("coleman"), "modules": pins}).encode())


def _read(path: Path, maximum: int) -> bytes:
    """Bound one metadata read; oversized input raises ValueError and I/O failures propagate."""
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        raise ValueError("Java matrix metadata exceeds its byte limit")
    return payload


def _contained(root: Path, path: Path) -> Path:
    """Reject escaped/symlink artifacts and nested Git storage before accessing source observations."""
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Java matrix artifact must remain within its output tree")
    validate_source_storage(path.parent)
    return path


def _generation(plan: JunitMatrixRecipe, root: Path, producer: str) -> Path:
    """Return a contained hash-only generation path; logical labels never become filesystem paths."""
    return _contained(
        root,
        root
        / "corpus-junit-matrix"
        / plan.fingerprint.removeprefix(_HASH_PREFIX)
        / producer.removeprefix(_HASH_PREFIX),
    )


def _cell_result(cell: JunitMatrixCell, dataset: ControlledDataset, cache: SourceCache) -> JunitMatrixCellResult:
    """Revalidate source/context and actual raw packets before deriving pair counts.

    This operates offline and starts no process. The existing source preparer may
    republish identical owned cache metadata while rebuilding the exact context.
    Normalized observations must correspond to every validated raw case, including
    errors and zero-duration cases. Forged/missing/conflicting evidence rejects.
    """
    validate_controlled_dataset(dataset, cache)
    environments = [item for item in dataset.spec.context.inputs if item.name == "execution-environment"]
    if len(environments) != 1:
        raise ValueError("Java matrix dataset lacks its execution environment")
    prepared = prepare_junit_inputs(cell.recipe, cache, environments[0].materialization)
    if prepared.context != dataset.spec.context:
        raise ValueError("Java matrix dataset context disagrees with its frozen source recipe")
    packet = dataset.spec.report_materialization
    roles = []
    observations = []
    for role in ("buggy", "fixed"):
        name = role + "-capture.json"
        members = [member for member in packet.spec.members if member.name == name]
        if len(members) != 1:
            raise ValueError("Java matrix dataset lacks a distinct raw role packet")
        payload = _read(cache.member_path(packet, name), JunitRoleCapture.MAXIMUM_BYTES)
        capture = JunitRoleCapture.from_json(payload)
        validate_role_bytes(capture, cell.recipe)
        if (capture.context_fingerprint, capture.trial_id, capture.role) != (
            prepared.context.fingerprint,
            cell.trial_id,
            role,
        ):
            raise ValueError("Java matrix raw role identity disagrees with its planned cell")
        pin = _digest(payload)
        if pin != members[0].sha256:
            raise ValueError("Java matrix raw role bytes disagree with their source pin")
        roles.append(
            JunitMatrixRole(
                role=capture.role,
                packet_sha256=pin,
                status=capture.status,
                producer_elapsed=capture.producer_elapsed,
            )
        )
        observations.extend(
            ControlledObservation(
                trial_id=capture.trial_id,
                role=capture.role,
                test_id=case.test_id,
                outcome=case.outcome,
                elapsed=ControlledElapsed(value=case.elapsed_ns, status="observed"),
                capture_started_at=case.capture_started_at,
                capture_finished_at=case.capture_finished_at,
                diagnostics_sha256=pin,
            )
            for case in capture.cases
        )
    expected = tuple(sorted(observations, key=lambda item: item.logical_key))
    if dataset.observations != expected or dataset.input_rows != len(expected) or dataset.duplicates_excluded:
        raise ValueError("Java matrix normalized observations disagree with actual raw cases")
    counts = Counter(pair.classification for pair in dataset.pairs)
    return JunitMatrixCellResult(
        cell_id=cell.cell_id,
        dataset_fingerprint=dataset.fingerprint,
        context_fingerprint=prepared.context.fingerprint,
        planned_pairs=len(cell.recipe.test_ids),
        unobserved_pairs=len(cell.recipe.test_ids) - len(dataset.pairs),
        pairs=JunitPairCounts(**counts),
        roles=tuple(roles),
    )


def validate_junit_matrix(receipt: JunitMatrixReceipt, cache: SourceCache, output: Path) -> None:
    """Verify an index against actual cached sources, raw packets and local controlled datasets offline.

    Unknown producer/schema, mutation or missing evidence raises ValueError or an
    I/O error. No Docker, network, source execution or native experiment is used.
    Identical source-view cache metadata may be republished by context validation.
    This proves correspondence, never authentic execution or scientific admission.
    """
    receipt = JunitMatrixReceipt.model_validate(receipt)
    if receipt.producer_sha256 != junit_matrix_producer():
        raise ValueError("Unknown Java matrix producer; explicit generation/migration is required")
    if len((receipt.plan.to_json() + "\n").encode()) > receipt.plan.MAXIMUM_BYTES:
        raise ValueError(_PLAN_LIMIT_ERROR)
    if len((receipt.to_json() + "\n").encode()) > receipt.MAXIMUM_BYTES:
        raise ValueError("Java matrix receipt exceeds its byte limit")
    validate_source_storage(output)
    validate_cache_separation(cache.root, output)
    actual = []
    references = {item.cell_id: item for item in receipt.cells}
    for cell in receipt.plan.cells:
        reference = references[cell.cell_id]
        path = _contained(
            output,
            output / "corpus-controlled" / (reference.dataset_fingerprint.removeprefix(_HASH_PREFIX) + _JSON_SUFFIX),
        )
        dataset = ControlledDataset.from_json(_read(path, _DATASET_BYTES))
        if dataset.fingerprint != reference.dataset_fingerprint:
            raise ValueError("Java matrix dataset identity disagrees with its index")
        actual.append(_cell_result(cell, dataset, cache))
    if tuple(actual) != receipt.cells:
        raise ValueError("Java matrix index disagrees with actual source evidence")


class JunitMatrixOutcome(NamedTuple):
    """Typed local result; an incomplete receipt has no completion path or claimed benchmark success."""

    receipt: JunitMatrixReceipt
    receipt_path: Path
    completion_path: Path | None


def capture_junit_matrix(
    plan: JunitMatrixRecipe, cache: SourceCache, writer: ArtifactWriter, *, fetch: bool = False
) -> JunitMatrixOutcome:
    """Collect predeclared source trials sequentially through the existing durable producer.

    Fetch is explicit/off by default; no image is pulled. Existing bounded Docker
    collection retains started/raw role acknowledgments, negative outcomes and
    same-trial interruption semantics. Concurrent ownership conflicts reject
    without duplicate execution; retry after the owner finishes reuses captures.
    A fresh collection needs a new trial ID. Infrastructure/preflight
    exceptions propagate with earlier source artifacts retained; no completion is
    written. Incomplete captures publish an honest index without a completion
    marker. Immutable complete publication occurs last, after raw/source validation.
    """
    plan = JunitMatrixRecipe.model_validate(plan)
    if type(fetch) is not bool:
        raise ValueError("Java matrix fetch flag must be boolean")
    plan_payload = plan.to_json() + "\n"
    if len(plan_payload.encode()) > plan.MAXIMUM_BYTES:
        raise ValueError(_PLAN_LIMIT_ERROR)
    validate_source_storage(writer.run_root)
    validate_cache_separation(cache.root, writer.run_root)
    producer = junit_matrix_producer()
    generation = _generation(plan, writer.run_root, producer)
    complete = _contained(writer.run_root, generation / _COMPLETION)
    plan_path = _contained(writer.run_root, generation / "plan.json")
    for cell in plan.cells:
        validate_junit_report_budget(cell.recipe, cell.trial_id)
    writer.write_immutable(plan_path, plan_payload)
    if complete.exists():
        saved = JunitMatrixReceipt.from_json(_read(complete, JunitMatrixReceipt.MAXIMUM_BYTES))
        if saved.plan != plan or saved.status != "collected":
            raise ValueError("Java matrix completion disagrees with the requested complete plan")
        validate_junit_matrix(saved, cache, writer.run_root)
        path = _contained(writer.run_root, generation / (saved.fingerprint.removeprefix(_HASH_PREFIX) + _JSON_SUFFIX))
        writer.write_immutable(path, saved.to_json() + "\n")
        return JunitMatrixOutcome(saved, path, complete)
    results = []
    for cell in plan.cells:
        result = capture_junit_source(cell.recipe, cache, writer, cell.trial_id, fetch=fetch)
        results.append(_cell_result(cell, result.dataset, cache))
    receipt = JunitMatrixReceipt(
        schema_version=1,
        plan=plan,
        producer_sha256=producer,
        cells=tuple(results),
        status="collected" if all(item.collected for item in results) else "incomplete",
    )
    payload = receipt.to_json() + "\n"
    if len(payload.encode()) > receipt.MAXIMUM_BYTES:
        raise ValueError("Java matrix receipt exceeds its byte limit")
    validate_junit_matrix(receipt, cache, writer.run_root)
    path = _contained(writer.run_root, generation / (receipt.fingerprint.removeprefix(_HASH_PREFIX) + _JSON_SUFFIX))
    writer.write_immutable(path, payload)
    if receipt.status == "collected":
        writer.write_immutable(complete, payload)
        return JunitMatrixOutcome(receipt, path, complete)
    return JunitMatrixOutcome(receipt, path, None)


def load_junit_matrix(path: Path | None = None) -> JunitMatrixRecipe:
    """Read a bounded plan or the bundled three-fix arm64 preset; never acquire or execute sources."""
    resource = path or files("coleman.corpus.resources").joinpath("java-real-fault-matrix-arm64-v1.json")
    with resource.open("rb") as stream:
        payload = stream.read(JunitMatrixRecipe.MAXIMUM_BYTES + 1)
    if len(payload) > JunitMatrixRecipe.MAXIMUM_BYTES:
        raise ValueError(_PLAN_LIMIT_ERROR)
    return JunitMatrixRecipe.from_json(payload)


def main(argv: list[str] | None = None) -> int:
    """Collect or offline-validate source evidence and emit JSON; incomplete collection returns 2.

    This optional transport does not run policies, alter admission, infer CI
    chronology or interpret source errors as assertion-based fault detections.
    Validation consumes bounded JSON bytes from stdin, never a receipt path;
    malformed/oversized input fails before source/data correspondence checks.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--recipe", type=Path)
    choice.add_argument("--validate", action="store_true", help="Validate bounded receipt JSON from stdin")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args(argv)
    if args.validate and args.fetch:
        parser.error("Offline validation cannot fetch sources")
    cache = SourceCache(args.cache)
    if args.validate:
        payload = sys.stdin.buffer.read(JunitMatrixReceipt.MAXIMUM_BYTES + 1)
        if len(payload) > JunitMatrixReceipt.MAXIMUM_BYTES:
            raise ValueError("Java matrix validation stdin exceeds its byte limit")
        receipt = JunitMatrixReceipt.from_json(payload)
        validate_junit_matrix(receipt, cache, args.output)
        print(json.dumps({"integrity": "verified", "receipt": receipt.fingerprint, "status": receipt.status}))
        return 0
    outcome = capture_junit_matrix(load_junit_matrix(args.recipe), cache, ArtifactWriter(args.output), fetch=args.fetch)
    print(
        json.dumps(
            {
                "receipt": str(outcome.receipt_path),
                "completion": str(outcome.completion_path) if outcome.completion_path else None,
                "status": outcome.receipt.status,
                "declared_projects": outcome.receipt.declared_projects,
                "curated_focal_fixes": outcome.receipt.curated_focal_fixes,
                "admission": outcome.receipt.admission,
                "generalization": outcome.receipt.generalization,
            },
            sort_keys=True,
        )
    )
    return 0 if outcome.receipt.status == "collected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
