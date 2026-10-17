"""Durable optional C source capture through existing generic controlled-result artifacts."""

from __future__ import annotations

import io
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

from pydantic import TypeAdapter

from coleman.artifacts import ArtifactWriter
from coleman.corpus import MaterializationSpec, SourceCache
from coleman.corpus.archives import deterministic_zip_bytes
from coleman.corpus.controlled import convert_controlled_source, publish_controlled_dataset, validate_controlled_dataset
from coleman.corpus.controlled_contracts import (
    ControlledDataset,
    ControlledObservation,
    ControlledReport,
    ControlledSpec,
    Label,
)
from coleman.corpus.materialization import materialize_source, validate_cache_separation, validate_source_storage
from coleman.corpus.materialization_contracts import PackageMember
from coleman.results.domain import fingerprint
from coleman.transports.c_contracts import (
    CCapturePlan,
    CCaptureReceipt,
    CCaptureRecipe,
    CRoleCapture,
    CRoleReference,
    CRoleStart,
)
from coleman.transports.c_process import c_runtime_environment, run_c_role
from coleman.transports.c_protocol import c_role_capture, validate_c_role_bytes
from coleman.transports.c_source import (
    PreparedCInputs,
    c_harness_plan,
    owned_c_package,
    prepare_c_inputs,
)
from coleman.transports.junit_capture import _cleanup, _CleanupError, _contained, _member_bytes, _single_producer
from coleman.transports.junit_contracts import ProducerElapsed
from coleman.transports.junit_source import digest
from coleman.transports.producer_identity import installed_producer_identity

_PLAN = "plan.json"
_COMPLETE = "complete.json"
_RECEIPT = "receipt.json"
_JSON_SUFFIX = ".json"


def _path(root: Path, name: str) -> Path:
    """Require contained trusted-local artifact storage, rejecting symlinks and nested Git worktrees."""
    path = _contained(root, root / name)
    validate_source_storage(path.parent)
    return path


def _read(path: Path, maximum: int) -> bytes:
    """Read one bounded metadata file; excess bytes reject before decoding and I/O errors propagate."""
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        raise ValueError("C capture metadata exceeds its byte limit")
    return payload


@contextmanager
def _docker_base():
    """Prepare a temporary empty client config only when execution/recovery needs Docker; never pull."""
    docker = shutil.which("docker")
    if docker is None:
        raise OSError("C source capture requires the documented local Docker runtime")
    with tempfile.TemporaryDirectory(prefix="coleman-c-client-") as temporary:
        config = Path(temporary)
        (config / "config.json").write_text("{}\n")
        yield (docker, "--config", str(config))


def _load_role(state: Path, role: str, plan: CCapturePlan) -> CRoleCapture:
    """Revalidate all stored raw/parsed facts and frozen identities before using a role checkpoint."""
    capture = CRoleCapture.from_json(_read(_path(state, role + _JSON_SUFFIX), CRoleCapture.MAXIMUM_BYTES))
    harness = c_harness_plan(plan.recipe, plan.context.fingerprint, plan.trial_id, role)
    validate_c_role_bytes(capture, plan.recipe, harness)
    return capture


def _capture_role(
    base: tuple[str, ...],
    plan: CCapturePlan,
    prepared: PreparedCInputs,
    cache: SourceCache,
    writer: ArtifactWriter,
    state: Path,
    role: str,
) -> CRoleCapture:
    """Recover uncommitted attempts as interrupted; never silently rerun an acknowledged physical role.

    A start marker precedes spawning. A role packet is atomically published only
    after container cleanup. Process launch errors retain unavailable counters;
    cleanup failures leave the marker for explicit same-trial recovery.
    """
    path = _path(state, role + _JSON_SUFFIX)
    if path.exists():
        return _load_role(state, role, plan)
    harness = c_harness_plan(plan.recipe, prepared.context.fingerprint, plan.trial_id, role)
    ownership = fingerprint([harness.fingerprint, str(writer.run_root.resolve())]).removeprefix("sha256:")
    container = "coleman-c-" + ownership[:24]
    start = CRoleStart(
        schema_version=1,
        context_fingerprint=prepared.context.fingerprint,
        trial_id=plan.trial_id,
        role=harness.role,
        container_name=container,
    )
    marker = _path(state, role + ".started.json")
    if marker.exists():
        if CRoleStart.from_json(_read(marker, CRoleStart.MAXIMUM_BYTES)) != start:
            raise ValueError("C start marker identity mismatch")
        _cleanup(base, container)
        status, diagnostics = "interrupted", "Recovered a started/uncommitted role; no outcomes or ticks inferred"
    else:
        writer.write_immutable(marker, start.to_json() + "\n")
        try:
            streams = run_c_role(base, plan.recipe, prepared, cache, harness, container)
        except _CleanupError:
            raise
        except OSError as exc:
            status = "process_error"
            diagnostics = (type(exc).__name__ + ": " + str(exc)).replace("\n", " ").replace("\r", " ")[:4096]
        else:
            capture = c_role_capture(plan.recipe, harness, streams)
            validate_c_role_bytes(capture, plan.recipe, harness)
            writer.write_immutable(path, capture.to_json() + "\n")
            return capture
    capture = CRoleCapture(
        schema_version=1,
        context_fingerprint=prepared.context.fingerprint,
        trial_id=plan.trial_id,
        role=harness.role,
        status=status,
        termination=None,
        exit_code=None,
        producer_elapsed=ProducerElapsed(value=None, status="unavailable"),
        stdout_base64="",
        stderr_base64="",
        header=None,
        cases=(),
        trailer=None,
        diagnostics=diagnostics,
    )
    writer.write_immutable(path, capture.to_json() + "\n")
    return capture


def _observations(captures: tuple[CRoleCapture, ...]) -> tuple[ControlledObservation, ...]:
    """Project actual completed cases into the unchanged generic domain, binding their raw role packet."""
    return tuple(
        ControlledObservation(
            trial_id=capture.trial_id,
            role=capture.role,
            test_id=case.test_id,
            outcome=case.outcome,
            elapsed=case.elapsed,
            capture_started_at=case.capture_started_at,
            capture_finished_at=case.capture_finished_at,
            diagnostics_sha256=digest(capture.to_json().encode()),
        )
        for capture in captures
        for case in capture.cases
    )


def _publish(
    plan: CCapturePlan,
    captures: tuple[CRoleCapture, ...],
    cache: SourceCache,
    writer: ArtifactWriter,
) -> tuple[ControlledDataset, Path]:
    """Publish a deterministic report envelope through existing source cache and controlled converter.

    Only fully observed cases are projected; absent cases remain unobserved and
    one-sided pairs inconclusive. Explicit test ns retain their original meaning.
    An incomplete capture may publish evidence but never a completion marker.
    """
    report = ControlledReport(
        schema_version=1, context_fingerprint=plan.context.fingerprint, observations=_observations(captures)
    )
    entries = tuple((capture.role + "-capture.json", capture.to_json().encode()) for capture in captures)
    entries += (("report.json", report.to_json().encode()),)
    payload = deterministic_zip_bytes(
        entries,
        maximum_member_bytes=CRoleCapture.MAXIMUM_BYTES,
        maximum_total_bytes=2 * CRoleCapture.MAXIMUM_BYTES + ControlledReport.MAXIMUM_BYTES,
    )
    owner = owned_c_package(cache, "capture-envelope.zip", payload, plan.recipe.recipe)
    spec = MaterializationSpec.model_validate(
        {
            **owner.spec.model_dump(),
            "archive": "zip",
            "members": tuple(
                PackageMember(name=name, sha256=digest(data), size_bytes=len(data)) for name, data in entries
            ),
        }
    )
    receipt = materialize_source(spec, cache, lambda: io.BytesIO(payload))
    source = ControlledSpec(
        schema_version=1, context=plan.context, report_materialization=receipt, report_member="report.json"
    )
    dataset = convert_controlled_source(source, cache)
    return dataset, publish_controlled_dataset(dataset, cache, writer)


def _receipt(plan: CCapturePlan, dataset: ControlledDataset, captures: tuple[CRoleCapture, ...]) -> CCaptureReceipt:
    """Construct an index from actual independently validated results; collected never means scientific admission."""
    roles = tuple(
        CRoleReference(
            role=capture.role,
            packet_sha256=digest(capture.to_json().encode()),
            status=capture.status,
            producer_elapsed=capture.producer_elapsed,
        )
        for capture in captures
    )
    return CCaptureReceipt(
        schema_version=1,
        plan_fingerprint=plan.fingerprint,
        context_fingerprint=plan.context.fingerprint,
        dataset_fingerprint=dataset.fingerprint,
        roles=roles,
        status="collected" if all(item.status == "collected" for item in captures) else "incomplete",
    )


class CCaptureOutcome(NamedTuple):
    """Typed paths and results; source-state records and native learning checkpoints remain distinct."""

    dataset: ControlledDataset
    captures: tuple[CRoleCapture, ...]
    dataset_path: Path
    state_directory: Path
    receipt: CCaptureReceipt


def validate_c_capture(
    receipt: CCaptureReceipt, plan: CCapturePlan, dataset: ControlledDataset, cache: SourceCache
) -> None:
    """Validate an offline index against context, all original raw packets and normalized observations.

    Recomputed hashes do not authorize rewritten outcomes: raw protocol is
    reparsed with the recipe and exact role plan. Unknown installed producers
    reject, without silently relabeling legacy results. Existing generic source
    datasets remain readable through their unchanged core API. This may verify
    or republish identical owned cache metadata; it never starts Docker/network.
    """
    receipt = CCaptureReceipt.model_validate(receipt)
    plan = CCapturePlan.model_validate(plan)
    dataset = ControlledDataset.model_validate(dataset)
    if plan.producer_sha256 != installed_producer_identity():
        raise ValueError("Unknown C capture producer; use its original reviewed package")
    prepared = prepare_c_inputs(plan.recipe, cache, plan.environment)
    if prepared.context != plan.context or dataset.spec.context != plan.context:
        raise ValueError("C capture context disagrees with actual pinned recipe inputs")
    validate_controlled_dataset(dataset, cache)
    packet = dataset.spec.report_materialization
    captures = []
    for role in ("buggy", "fixed"):
        name = role + "-capture.json"
        if len([member for member in packet.spec.members if member.name == name]) != 1:
            raise ValueError("C capture dataset requires both distinct raw role packets")
        capture = CRoleCapture.from_json(_member_bytes(cache, packet, name))
        harness = c_harness_plan(plan.recipe, plan.context.fingerprint, plan.trial_id, role)
        validate_c_role_bytes(capture, plan.recipe, harness)
        captures.append(capture)
    actual = tuple(captures)
    report = ControlledReport(
        schema_version=1, context_fingerprint=plan.context.fingerprint, observations=_observations(actual)
    )
    canonical = tuple(sorted(report.observations, key=lambda item: item.logical_key))
    if canonical != dataset.observations or _receipt(plan, dataset, actual) != receipt:
        raise ValueError("C index or normalized observations disagree with original raw evidence")


def _prepare_plan(
    recipe: CCaptureRecipe,
    cache: SourceCache,
    writer: ArtifactWriter,
    trial_id: str,
    producer: str,
    state: Path,
    *,
    fetch: bool,
) -> tuple[CCapturePlan, PreparedCInputs]:
    """Rebuild an existing frozen context or durably bind fresh inputs before any subject process.

    Missing plans cannot be reconstructed over existing ownership/evidence. Old
    source versions are never rewritten into a newly prepared environment.
    Metadata-only Docker queries are explicit; actual subject execution is separate.
    """
    plan_path = _path(state, _PLAN)
    if plan_path.exists():
        plan = CCapturePlan.from_json(_read(plan_path, CCapturePlan.MAXIMUM_BYTES))
        if (plan.recipe, plan.trial_id, plan.producer_sha256) != (recipe, trial_id, producer):
            raise ValueError("C stored plan disagrees with recipe/trial/producer")
        prepared = prepare_c_inputs(recipe, cache, plan.environment, fetch=fetch)
        if prepared.context != plan.context:
            raise ValueError("C stored context disagrees with actual pinned inputs")
    else:
        if any(
            _path(state, name).exists()
            for name in (
                "buggy.json",
                "fixed.json",
                "buggy.started.json",
                "fixed.started.json",
                _RECEIPT,
                _COMPLETE,
            )
        ):
            raise ValueError("C evidence cannot exist without its original pre-execution plan")
        with _docker_base() as base:
            environment = c_runtime_environment(base, recipe, cache)
        prepared = prepare_c_inputs(recipe, cache, environment, fetch=fetch)
        plan = CCapturePlan(
            schema_version=1,
            recipe=recipe,
            trial_id=trial_id,
            producer_sha256=producer,
            environment=environment,
            context=prepared.context,
        )
        if len(plan.to_json().encode()) > plan.MAXIMUM_BYTES:
            raise ValueError("C pre-execution plan exceeds its byte limit")
        writer.write_immutable(plan_path, plan.to_json() + "\n")
    return plan, prepared


def capture_c_source(
    recipe: CCaptureRecipe, cache: SourceCache, writer: ArtifactWriter, trial_id: str, *, fetch: bool = False
) -> CCaptureOutcome:
    """Collect one paired source trial, with immutable ownership, offline replay and honest interruption.

    Inputs/output must reside outside Git and in separate roots. Acquisition is
    offline unless explicitly enabled; the pinned image must already be present.
    One POSIX producer owns a recipe/trial/implementation generation. Completed
    roles replay without Docker; an uncommitted attempt becomes interrupted and
    needs a new trial ID for fresh observation. Before executing a missing role,
    actual runtime facts must still match the pre-execution plan. A changed
    producer uses a new generation. Completion is written last, only after both
    protocols, raw bytes and generic dataset validate. Source/test errors remain
    evidence; invalid schema/state raises and OS failures propagate.
    """
    recipe = CCaptureRecipe.model_validate(recipe)
    trial_id = TypeAdapter(Label).validate_python(trial_id)
    if type(fetch) is not bool:
        raise ValueError("C fetch flag must be boolean")
    if len(recipe.to_json().encode()) > recipe.MAXIMUM_BYTES:
        raise ValueError("C recipe exceeds its byte limit")
    validate_source_storage(writer.run_root)
    validate_cache_separation(cache.root, writer.run_root)
    validate_source_storage(writer.run_root / "corpus-controlled")
    producer = installed_producer_identity()
    generation = fingerprint(["c-capture-v1", recipe.fingerprint, trial_id, producer]).removeprefix("sha256:")
    state = _path(writer.run_root, "c-captures/" + generation)
    validate_source_storage(state)
    with _single_producer(state):
        plan, prepared = _prepare_plan(recipe, cache, writer, trial_id, producer, state, fetch=fetch)
        for role in ("buggy", "fixed"):
            if _path(state, role + _JSON_SUFFIX).exists():
                _load_role(state, role, plan)
        missing = [role for role in ("buggy", "fixed") if not _path(state, role + _JSON_SUFFIX).exists()]
        if missing and any(_path(state, name).exists() for name in (_RECEIPT, _COMPLETE)):
            raise ValueError("C index/completion cannot replace a missing raw role packet")
        if missing:
            with _docker_base() as base:
                if c_runtime_environment(base, recipe, cache) != plan.environment:
                    raise ValueError("C runtime changed within this trial; use a new trial ID")
                for role in missing:
                    _capture_role(base, plan, prepared, cache, writer, state, role)
        captures = tuple(_load_role(state, role, plan) for role in ("buggy", "fixed"))
        dataset, dataset_path = _publish(plan, captures, cache, writer)
        validate_controlled_dataset(dataset, cache)
        receipt = _receipt(plan, dataset, captures)
        validate_c_capture(receipt, plan, dataset, cache)
        writer.write_immutable(_path(state, _RECEIPT), receipt.to_json() + "\n")
        if receipt.status == "collected":
            writer.write_immutable(_path(state, _COMPLETE), receipt.to_json() + "\n")
        elif _path(state, _COMPLETE).exists():
            raise ValueError("Incomplete C evidence cannot have a completion marker")
    return CCaptureOutcome(dataset, captures, dataset_path, state, receipt)
