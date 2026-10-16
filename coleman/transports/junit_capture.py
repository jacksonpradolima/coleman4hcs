"""Optional local Docker/JUnit producer; native TCPCI experiments continue to use the existing runner."""

from __future__ import annotations

import csv
import io
import os
import platform
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import NamedTuple

from pydantic import TypeAdapter

from coleman.artifacts import ArtifactWriter
from coleman.corpus import MaterializationReceipt, SourceCache
from coleman.corpus.controlled import convert_controlled_source, publish_controlled_dataset
from coleman.corpus.controlled_contracts import (
    ControlledDataset,
    ControlledElapsed,
    ControlledObservation,
    ControlledReport,
    ControlledSpec,
    Label,
)
from coleman.corpus.materialization import materialize_source, validate_cache_separation, validate_source_storage
from coleman.corpus.materialization_contracts import MaterializationSpec, PackageMember
from coleman.corpus.source_views import source_view_bytes
from coleman.results.domain import decode_json, fingerprint
from coleman.transports.junit_contracts import (
    JunitCaptureRecipe,
    JunitEnvironment,
    JunitRoleCapture,
    JunitRoleStart,
    ProducerElapsed,
)
from coleman.transports.junit_process import collect_process
from coleman.transports.junit_protocol import role_capture, validate_role_bytes
from coleman.transports.junit_source import (
    HARNESS_NAME,
    PreparedJunitInputs,
    digest,
    java_commands,
    owned_package,
    prepare_junit_inputs,
)

_JAVA_SUFFIX = ".java"


def _contained(root: Path, path: Path) -> Path:
    """Require a cooperative trusted-local output path; no symlink or resolved parent escape is adopted."""
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("JUnit output must remain within its trusted output tree")
    return path


@contextmanager
def _single_producer(root: Path) -> Iterator[None]:
    """Serialize one context/trial producer across POSIX processes; crash releases the advisory lock.

    Lock files are neither completed captures nor native learning checkpoints.
    This assumes trusted local parent directories, like the existing source cache.
    """
    if os.name != "posix":
        raise OSError("JUnit producer requires POSIX advisory locks")
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    path = _contained(root, root / "producer.lock")
    with path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                "Another producer owns this context/trial; no duplicate source execution was started"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _docker_query(base: tuple[str, ...], args: tuple[str, ...]):
    """Read only specifically selected public runtime metadata, never tags, containers, contexts or credentials."""
    result = collect_process(base + args, timeout_seconds=15, maximum_output_bytes=256 * 1024)
    if result.termination != "exited" or result.exit_code != 0:
        raise OSError("Pinned local Docker runtime unavailable; prepare the documented image before capture")
    return decode_json(result.stdout)


def _runtime_environment(
    base: tuple[str, ...], recipe: JunitCaptureRecipe, cache: SourceCache
) -> MaterializationReceipt:
    """Verify an already installed image and record filtered architecture/version facts; never pull an image."""
    image = "eclipse-temurin@" + recipe.runtime.image_digest
    info = _docker_query(
        base,
        ("image", "inspect", image, "--format", "[{{json .Os}},{{json .Architecture}},{{json .Size}},{{json .Id}}]"),
    )
    versions = _docker_query(
        base, ("version", "--format", "[{{json .Client.Version}},{{json .Server.Version}},{{json .Server.Arch}}]")
    )
    kernel = _docker_query(base, ("info", "--format", "{{json .KernelVersion}}"))
    if not isinstance(info, list) or len(info) != 4 or not isinstance(versions, list) or len(versions) != 3:
        raise ValueError("Unexpected filtered Docker runtime metadata")
    target = recipe.runtime.platform.split("/")[1]
    aliases = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "amd64", "amd64": "amd64", "AMD64": "amd64"}
    host = aliases.get(platform.machine())
    if not all(isinstance(value, str) for value in versions):
        raise ValueError("Unexpected Docker version/architecture scalar")
    server = aliases.get(versions[2])
    if info[0] != "linux" or info[1] != target or host is None or server is None:
        raise ValueError("Local image/host architecture does not satisfy the reviewed runtime recipe")
    if recipe.runtime.emulated != (len({host, server, target}) != 1):
        raise ValueError("Visible architecture translation disagrees with the declared emulation profile")
    environment = JunitEnvironment.model_validate(
        {
            "schema_version": 1,
            "kind": "filtered-local-junit-runtime-v1",
            "host_os": platform.system(),
            "host_architecture": platform.machine(),
            "host_release": platform.release(),
            "producer_python_version": platform.python_version(),
            "producer_package_version": version("coleman"),
            "docker_client_version": versions[0],
            "docker_server_version": versions[1],
            "docker_server_architecture": versions[2],
            "container_kernel": kernel,
            "image_os": info[0],
            "image_architecture": info[1],
            "image_size_bytes": info[2],
            "image_config_digest": info[3],
        }
    )
    return owned_package(
        cache, "execution-environment.json", environment.to_json().encode(), "filtered-local-junit-runtime-v1"
    )


def _member_bytes(cache: SourceCache, receipt: MaterializationReceipt, name: str) -> bytes:
    """Rehash the actual bounded bytes copied to the compiler, after the cache's own verification."""
    member = next(item for item in receipt.spec.members if item.name == name)
    if member.size_bytes > 32 * 1024**2:
        raise ValueError("JUnit staging member exceeds its byte bound")
    path = cache.member_path(receipt, name)
    with path.open("rb") as stream:
        payload = stream.read(member.size_bytes + 1)
    if len(payload) != member.size_bytes or digest(payload) != member.sha256:
        raise ValueError("JUnit staged bytes disagree with the pinned member")
    return payload


def _stage_view(cache: SourceCache, receipt: MaterializationReceipt, root: Path) -> None:
    """Stage only declared regular-file bytes in a fresh private temporary tree, never extracting arbitrary paths."""
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    for member in receipt.spec.members:
        path = _contained(root, root / member.name)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        for directory in (path.parent, *path.parent.parents):
            if directory == root.parent:
                break
            directory.chmod(0o755)
        path.write_bytes(_member_bytes(cache, receipt, member.name))
        path.chmod(0o644)


def _mount(source: Path, target: str) -> str:
    """Encode Docker's CSV mount argument, including an explicit temporary path containing commas."""
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="").writerow(["type=bind", f"source={source}", f"target={target}", "readonly"])
    return output.getvalue()


def _source_arguments(receipt: MaterializationReceipt, prefix: str, target: str) -> str:
    """Return quoted javac argument-file entries; safe materialization names cannot contain quotes/control syntax."""
    names = [
        item.name
        for item in receipt.spec.members
        if item.name.startswith(prefix + "/") and item.name.endswith(_JAVA_SUFFIX)
    ]
    return "".join(f'"{target}/{name}"\n' for name in sorted(names))


def _run_role(
    base: tuple[str, ...],
    recipe: JunitCaptureRecipe,
    prepared: PreparedJunitInputs,
    cache: SourceCache,
    role: str,
    container_name: str,
):
    """Execute one offline, non-root, read-only common-fixture JVM under container/parent guards."""
    source = prepared.buggy if role == "buggy" else prepared.fixed
    with tempfile.TemporaryDirectory(prefix="coleman-junit-stage-") as temporary:
        stage = Path(temporary)
        _stage_view(cache, source, stage / "source")
        _stage_view(cache, prepared.fixture, stage / "fixture")
        dependencies = stage / "deps"
        dependencies.mkdir(mode=0o755)
        dependencies.chmod(0o755)
        for index, receipt in enumerate(prepared.dependencies):
            path = dependencies / f"{index}.jar"
            path.write_bytes(_member_bytes(cache, receipt, receipt.spec.members[0].name))
            path.chmod(0o644)
        harness = stage / "harness"
        harness.mkdir(mode=0o755)
        harness.chmod(0o755)
        (harness / HARNESS_NAME).write_bytes(prepared.harness)
        (harness / "production-sources.txt").write_text(_source_arguments(source, recipe.production_prefix, "/source"))
        (harness / "fixture-sources.txt").write_text(
            _source_arguments(prepared.fixture, recipe.fixture_prefix, "/fixture")
        )
        classes = [
            item.name.removeprefix(recipe.fixture_prefix + "/").removesuffix(_JAVA_SUFFIX).replace("/", ".")
            for item in prepared.fixture.spec.members
            if item.name.startswith(recipe.fixture_prefix + "/") and item.name.endswith(_JAVA_SUFFIX)
        ]
        (harness / "classes.txt").write_text("\n".join(sorted(classes)) + "\n")
        for path in harness.iterdir():
            path.chmod(0o644)
        uid, gid = (os.getuid(), os.getgid()) if os.getuid() else (65534, 65534)
        args = base + (
            "run",
            "--pull=never",
            "--rm",
            "--name",
            container_name,
            "--platform",
            recipe.runtime.platform,
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--cpus",
            "2",
            "--memory",
            str(recipe.limits.memory_bytes),
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=134217728,mode=1777",
            "--user",
            f"{uid}:{gid}",
            "--env",
            "TZ=" + recipe.runtime.timezone,
            "--env",
            "LANG=C.UTF-8",
            "--env",
            "LC_ALL=C.UTF-8",
        )
        for name in ("source", "fixture", "deps", "harness"):
            args += ("--mount", _mount(stage / name, "/" + name))
        compile_command, observe_command = java_commands(len(prepared.dependencies))
        script = "mkdir /tmp/classes && cd /fixture && " + compile_command + " && exec " + observe_command
        try:
            return collect_process(
                args + ("eclipse-temurin@" + recipe.runtime.image_digest, "sh", "-c", script),
                timeout_seconds=recipe.limits.timeout_seconds,
                maximum_output_bytes=recipe.limits.maximum_output_bytes,
            )
        finally:
            _cleanup(base, container_name)


class _CleanupError(OSError):
    """Leave a start marker recoverable when producer-owned container cleanup cannot be confirmed."""


def _cleanup(base: tuple[str, ...], name: str) -> None:
    """Remove only this deterministic producer-owned container name; an absent container is normal."""
    try:
        result = collect_process(base + ("rm", "-f", name), timeout_seconds=15, maximum_output_bytes=65536)
    except OSError as exc:
        raise _CleanupError("Producer-owned container cleanup failed before acknowledgment") from exc
    absent = result.stderr.decode("utf-8", errors="replace").strip() == (
        "Error response from daemon: No such container: " + name
    )
    if result.termination != "exited" or (result.exit_code != 0 and not absent):
        raise _CleanupError("Producer-owned container cleanup failed; recover its deterministic context/trial role")


def _load_role(path: Path, recipe: JunitCaptureRecipe) -> JunitRoleCapture:
    """Read a bounded role checkpoint and revalidate its actual raw stream/case correspondence offline."""
    with path.open("rb") as stream:
        payload = stream.read(16 * 1024**2 + 1)
    if len(payload) > 16 * 1024**2:
        raise ValueError("JUnit role checkpoint exceeds 16 MiB")
    capture = JunitRoleCapture.from_json(payload)
    validate_role_bytes(capture, recipe)
    return capture


def _capture_role(
    base: tuple[str, ...],
    recipe: JunitCaptureRecipe,
    prepared: PreparedJunitInputs,
    cache: SourceCache,
    writer: ArtifactWriter,
    state: Path,
    trial_id: str,
    role: str,
) -> JunitRoleCapture:
    """Reuse completed captures; recover started/uncommitted roles as interrupted without silent source reruns."""
    path = _contained(writer.run_root, state / (role + ".json"))
    marker = _contained(writer.run_root, state / (role + ".started.json"))
    expected = (prepared.context.fingerprint, trial_id, role)
    if path.exists():
        capture = _load_role(path, recipe)
        if (capture.context_fingerprint, capture.trial_id, capture.role) != expected:
            raise ValueError("JUnit role checkpoint identity mismatch")
        return capture
    container_name = "coleman-junit-" + state.name[:24] + "-" + role
    if marker.exists():
        with marker.open("rb") as stream:
            payload = stream.read(4097)
        if len(payload) > 4096:
            raise ValueError("JUnit start marker exceeds 4 KiB")
        started = JunitRoleStart.from_json(payload)
        if (started.context_fingerprint, started.trial_id, started.role) != expected:
            raise ValueError("JUnit start marker identity mismatch")
        _cleanup(base, container_name)
        capture = JunitRoleCapture.model_validate(
            {
                "schema_version": 1,
                "context_fingerprint": expected[0],
                "trial_id": trial_id,
                "role": role,
                "status": "interrupted",
                "exit_code": None,
                "capture_started_at": started.capture_started_at,
                "capture_finished_at": datetime.now(UTC),
                "producer_elapsed": ProducerElapsed(value=None, status="unavailable"),
                "stdout_base64": None,
                "stderr_base64": None,
                "header": None,
                "cases": (),
                "problem": "Recovered a started but uncommitted source capture; no outcomes or elapsed ticks inferred",
            }
        )
    else:
        now = datetime.now(UTC)
        started = JunitRoleStart.model_validate(
            {
                "schema_version": 1,
                "context_fingerprint": expected[0],
                "trial_id": trial_id,
                "role": role,
                "capture_started_at": now,
            }
        )
        writer.write_immutable(marker, started.to_json() + "\n")
        try:
            streams = _run_role(base, recipe, prepared, cache, role, container_name)
        except _CleanupError:
            raise
        except OSError as exc:
            capture = JunitRoleCapture.model_validate(
                {
                    "schema_version": 1,
                    "context_fingerprint": expected[0],
                    "trial_id": trial_id,
                    "role": role,
                    "status": "process_error",
                    "exit_code": None,
                    "capture_started_at": now,
                    "capture_finished_at": datetime.now(UTC),
                    "producer_elapsed": ProducerElapsed(value=None, status="unavailable"),
                    "stdout_base64": None,
                    "stderr_base64": None,
                    "header": None,
                    "cases": (),
                    "problem": (type(exc).__name__ + ": " + str(exc)).replace("\n", " ").replace("\r", " ")[:1024],
                }
            )
        else:
            capture = role_capture(recipe, expected[0], trial_id, role, now, datetime.now(UTC), streams)
    writer.write_immutable(path, capture.to_json() + "\n")
    return capture


class JunitCaptureOutcome(NamedTuple):
    """Typed local paths/results; source packets and domain dataset retain their independent v1 contracts."""

    dataset: ControlledDataset
    captures: tuple[JunitRoleCapture, ...]
    dataset_path: Path
    state_directory: Path


def _publish_capture(
    recipe: JunitCaptureRecipe,
    prepared: PreparedJunitInputs,
    captures: tuple[JunitRoleCapture, ...],
    cache: SourceCache,
    writer: ArtifactWriter,
) -> tuple[ControlledDataset, Path]:
    """Package normalized observations plus both raw role packets; source diagnostics always have pinned members."""
    entries = tuple((capture.role + "-capture.json", capture.to_json().encode()) for capture in captures)
    pins = {name: digest(payload) for name, payload in entries}
    observations = tuple(
        ControlledObservation(
            trial_id=capture.trial_id,
            role=capture.role,
            test_id=case.test_id,
            outcome=case.outcome,
            elapsed=ControlledElapsed(value=case.elapsed_ns, status="observed"),
            capture_started_at=case.capture_started_at,
            capture_finished_at=case.capture_finished_at,
            diagnostics_sha256=pins[capture.role + "-capture.json"],
        )
        for capture in captures
        for case in capture.cases
    )
    report = ControlledReport(
        schema_version=1, context_fingerprint=prepared.context.fingerprint, observations=observations
    )
    entries += (("report.json", report.to_json().encode()),)
    payload = source_view_bytes(entries)
    owner = owned_package(cache, "capture-envelope.zip", payload, recipe.recipe)
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
        schema_version=1, context=prepared.context, report_materialization=receipt, report_member="report.json"
    )
    dataset = convert_controlled_source(source, cache)
    return dataset, publish_controlled_dataset(dataset, cache, writer)


def capture_junit_source(
    recipe: JunitCaptureRecipe, cache: SourceCache, writer: ArtifactWriter, trial_id: str, *, fetch: bool = False
) -> JunitCaptureOutcome:
    """Collect one paired source trial with verified inputs, filtered runtime provenance and immutable recovery.

    Default acquisition is offline and the public pinned runtime image must
    already exist locally. Raw output roots must be physically outside Git
    worktrees, checked by the shared cache guard before runtime/source access.
    No image is pulled and no network is available to the
    subject. Concurrent same-trial producers fail before execution. Completed
    roles replay; a started/uncommitted role becomes interrupted/inconclusive,
    requiring a new trial ID for fresh collection. Neither source capture nor
    unexpected assertions establish CI chronology, independent faults or admission.
    Native learning checkpoints and TCPCI experiments remain the native runner's
    responsibility. Invalid source/schema/output states raise; OS failures outside
    a role propagate. Role failures remain versioned raw evidence, never passes.
    """
    recipe = JunitCaptureRecipe.model_validate(recipe)
    trial_id = TypeAdapter(Label).validate_python(trial_id)
    validate_source_storage(writer.run_root)
    validate_cache_separation(cache.root, writer.run_root)
    docker = shutil.which("docker")
    if docker is None:
        raise OSError("JUnit capture requires the documented local Docker runtime")
    with tempfile.TemporaryDirectory(prefix="coleman-junit-client-") as temporary:
        config = Path(temporary)
        (config / "config.json").write_text("{}\n")
        base = (docker, "--config", str(config))
        environment = _runtime_environment(base, recipe, cache)
        prepared = prepare_junit_inputs(recipe, cache, environment, fetch=fetch)
        session = fingerprint(["junit-capture-session-v1", prepared.context.fingerprint, trial_id]).removeprefix(
            "sha256:"
        )
        state = _contained(writer.run_root, writer.run_root / "junit-captures" / session)
        with _single_producer(state):
            writer.write_immutable(state / "context.json", prepared.context.to_json() + "\n")
            captures = tuple(
                _capture_role(base, recipe, prepared, cache, writer, state, trial_id, role)
                for role in ("buggy", "fixed")
            )
            dataset, path = _publish_capture(recipe, prepared, captures, cache, writer)
    return JunitCaptureOutcome(dataset, captures, path, state)
