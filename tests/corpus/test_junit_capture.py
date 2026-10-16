"""Owned producer tests exercise actual cache/contracts/publication; only the external Docker process is substituted."""

import io
import json
from collections import Counter
from datetime import UTC, datetime

import pytest

from coleman.artifacts import ArtifactWriter
from coleman.corpus import SourceCache, validate_materialization
from coleman.corpus.controlled_contracts import ControlledDataset
from coleman.corpus.materialization import materialize_source
from coleman.transports import junit
from coleman.transports import junit_capture as capture
from coleman.transports.junit_contracts import JunitCaptureRecipe, JunitRoleCapture, JunitRoleStart
from coleman.transports.junit_process import ProcessStreams
from coleman.transports.junit_source import classpath, digest, harness_bytes, owned_package, prepare_junit_inputs
from tests.corpus.test_junit_protocol import changed, encoded, recipe, records


def prepared_inputs(tmp_path):
    spec = changed(recipe(), harness_sha256=digest(harness_bytes()))
    cache = SourceCache(tmp_path / "cache")
    for view in (spec.buggy, spec.fixed, spec.fixture):
        for package in view.inputs:
            materialize_source(package, cache, lambda: io.BytesIO(b"Owned schema-only source fixture.\n"))
    materialize_source(spec.dependencies[0], cache, lambda: io.BytesIO(b"Owned schema-only dependency, not a real JAR"))
    environment = owned_package(cache, "environment.json", b"Owned environment fixture", "owned-v1")
    return spec, cache, environment


def substitute_docker(monkeypatch, environment, run):
    monkeypatch.setattr(capture.shutil, "which", lambda _: "/owned/docker")
    monkeypatch.setattr(capture, "_runtime_environment", lambda *args: environment)
    monkeypatch.setattr(capture, "_run_role", run)
    monkeypatch.setattr(capture, "_cleanup", lambda *args: None)


def test_offline_inputs_bind_actual_implementation_and_reject_changed_harness(tmp_path):
    spec, cache, environment = prepared_inputs(tmp_path)
    prepared = prepare_junit_inputs(spec, cache, environment)
    assert prepared.context.test_ids == spec.test_ids
    assert prepared.context.runtime.execution_model == "container"
    assert prepared.harness == harness_bytes()
    assert len(prepared.context.inputs) == 13
    for item in prepared.context.inputs:
        validate_materialization(item.materialization, cache)
    wrong = changed(spec, harness_sha256=digest(b"another harness"))
    with pytest.raises(ValueError, match="harness disagrees"):
        prepare_junit_inputs(wrong, cache, environment)
    invalid_fetch = json.loads("1")
    with pytest.raises(ValueError, match="boolean"):
        prepare_junit_inputs(spec, cache, environment, fetch=invalid_fetch)


def test_paired_collection_checkpoint_replay_actual_core_and_cli(tmp_path, monkeypatch, capsys):
    spec, cache, environment = prepared_inputs(tmp_path)
    calls = []

    def run(*args):
        role = args[-2]
        calls.append(role)
        rows = records()
        rows[1]["failures"] = int(role == "buggy")
        return ProcessStreams(encoded(rows), b"compiler diagnostic", 0, 0, "exited")

    substitute_docker(monkeypatch, environment, run)
    writer = ArtifactWriter(tmp_path / "results")
    first = capture.capture_junit_source(spec, cache, writer, "owned-trial")
    replay = capture.capture_junit_source(spec, cache, writer, "owned-trial")
    assert first == replay
    assert calls == ["buggy", "fixed"]
    assert first.dataset.pairs[0].classification == "expected"
    assert first.dataset.admission == "not_evaluated"
    assert first.dataset.generalization == "insufficient_evidence"
    assert ControlledDataset.from_json(first.dataset_path.read_bytes()) == first.dataset
    assert first.dataset.observations[0].elapsed.value == 0
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(spec.to_json())
    code = junit.main(
        [
            "--recipe",
            str(recipe_path),
            "--cache",
            str(cache.root),
            "--output",
            str(writer.run_root),
            "--trial",
            "owned-trial",
        ]
    )
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["pairs"] == {"expected": 1}
    assert calls == ["buggy", "fixed"]
    with pytest.raises(ValueError, match="not historical CI"):
        first.dataset.require_temporal_evaluation()
    packet_path = first.state_directory / "buggy.json"
    packet = JunitRoleCapture.from_json(packet_path.read_bytes())
    packet_path.write_text(changed(packet, trial_id="wrong-trial").to_json())
    with pytest.raises(ValueError, match="identity mismatch"):
        capture.capture_junit_source(spec, cache, writer, "owned-trial")


def test_process_interruption_has_no_false_completion_and_recovers_without_rerun(tmp_path, monkeypatch):
    spec, cache, environment = prepared_inputs(tmp_path)
    calls = []

    def run(*args):
        calls.append(args[-2])
        if len(calls) == 1:
            raise KeyboardInterrupt
        return ProcessStreams(encoded(records()), b"", 0, 1, "exited")

    substitute_docker(monkeypatch, environment, run)
    writer = ArtifactWriter(tmp_path / "results")
    with pytest.raises(KeyboardInterrupt):
        capture.capture_junit_source(spec, cache, writer, "interrupted-trial")
    assert not list(writer.run_root.glob("corpus-controlled/*.json"))
    assert not list(writer.run_root.glob("junit-captures/*/buggy.json"))
    result = capture.capture_junit_source(spec, cache, writer, "interrupted-trial")
    assert calls == ["buggy", "fixed"]
    assert [item.status for item in result.captures] == ["interrupted", "collected"]
    interrupted = result.captures[0]
    assert interrupted.stdout_base64 is None
    assert interrupted.stderr_base64 is None
    assert interrupted.producer_elapsed.value is None
    assert result.dataset.pairs[0].classification == "inconclusive"
    assert len(result.dataset.observations) == 1
    assert capture.capture_junit_source(spec, cache, writer, "interrupted-trial") == result
    marker = result.state_directory / "buggy.started.json"
    started = JunitRoleStart.from_json(marker.read_bytes())
    (result.state_directory / "buggy.json").unlink()
    marker.write_text(changed(started, trial_id="wrong").to_json())
    with pytest.raises(ValueError, match="marker identity mismatch"):
        capture.capture_junit_source(spec, cache, writer, "interrupted-trial")


def test_launch_failure_preserves_unavailable_values_and_cli_reports_incomplete(tmp_path, monkeypatch, capsys):
    spec, cache, environment = prepared_inputs(tmp_path)

    def run(*args):
        raise OSError("Owned launch failure")

    substitute_docker(monkeypatch, environment, run)
    writer = ArtifactWriter(tmp_path / "results")
    result = capture.capture_junit_source(spec, cache, writer, "failed-launch")
    assert result.dataset.observations == ()
    assert result.dataset.pairs == ()
    assert all(item.status == "process_error" for item in result.captures)
    assert all(item.stdout_base64 is None for item in result.captures)
    path = tmp_path / "recipe.json"
    path.write_text(spec.to_json())
    assert (
        junit.main(
            [
                "--recipe",
                str(path),
                "--cache",
                str(cache.root),
                "--output",
                str(writer.run_root),
                "--trial",
                "failed-launch",
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out)["pairs"] == {}


def test_staging_exact_bytes_javac_arguments_and_postverification_tamper(tmp_path):
    spec, cache, environment = prepared_inputs(tmp_path)
    prepared = prepare_junit_inputs(spec, cache, environment)
    root = tmp_path / "stage"
    capture._stage_view(cache, prepared.buggy, root)
    assert (root / "src/java/owned/CheckCase.java").read_bytes() == b"Owned schema-only source fixture.\n"
    assert (root / "src/java").stat().st_mode & 0o777 == 0o755
    assert (
        capture._source_arguments(prepared.buggy, "src/java", "/source") == '"/source/src/java/owned/CheckCase.java"\n'
    )
    cache.member_path(prepared.buggy, "src/java/owned/CheckCase.java").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="disagrees"):
        capture._member_bytes(cache, prepared.buggy, "src/java/owned/CheckCase.java")


def test_concurrent_producer_lock_release_and_containment(tmp_path):
    state = tmp_path / "state"
    with (
        capture._single_producer(state),
        pytest.raises((ValueError, OSError), match="producer"),
        capture._single_producer(state),
    ):
        pytest.fail("Concurrent writer acquired the same trial")
    with capture._single_producer(state):
        pass
    with pytest.raises(ValueError, match="trusted output"):
        capture._contained(tmp_path, tmp_path / ".." / "escape")
    link = tmp_path / "link"
    link.symlink_to(state)
    with pytest.raises(ValueError, match="trusted output"):
        capture._contained(tmp_path, link)


def test_preset_is_metadata_only_and_current_harness_pinned():
    spec = junit.load_junit_recipe()
    assert len(spec.test_ids) == 95
    assert spec.harness_sha256 == digest(harness_bytes())
    assert len(spec.fixture.inputs) == 23
    assert spec.dependencies[0].source.sha256 is not None
    assert JunitCaptureRecipe.from_json(spec.to_json()) == spec
    assert Counter(item.license.license_id for item in spec.dependencies) == {"Apache-2.0": 1, "CPL-1.0": 1}


@pytest.mark.parametrize("count", [0, 17])
def test_classpath_limits(count):
    with pytest.raises(ValueError, match="1..16"):
        classpath(count)


def test_cleanup_is_scoped_and_does_not_hide_daemon_failure(monkeypatch):
    calls = []

    def collect(args, **kwargs):
        calls.append(args)
        return ProcessStreams(b"", b"Error response from daemon: No such container: owned-name\n", 1, 1, "exited")

    monkeypatch.setattr(capture, "collect_process", collect)
    capture._cleanup(("docker",), "owned-name")
    assert calls == [("docker", "rm", "-f", "owned-name")]
    monkeypatch.setattr(
        capture, "collect_process", lambda *args, **kwargs: ProcessStreams(b"", b"daemon unavailable", 1, 1, "exited")
    )
    with pytest.raises(OSError, match="cleanup failed"):
        capture._cleanup(("docker",), "owned-name")


def test_missing_runtime_rejects_before_collection(tmp_path, monkeypatch):
    spec, cache, _ = prepared_inputs(tmp_path)
    monkeypatch.setattr(capture.shutil, "which", lambda _: None)
    writer = ArtifactWriter(tmp_path / "results")
    with pytest.raises(OSError, match="local Docker"):
        capture.capture_junit_source(spec, cache, writer, "owned-trial")


def test_bounded_recipe_and_checkpoint_reads(tmp_path):
    path = tmp_path / "too-large.json"
    path.write_bytes(b" " * (2 * 1024**2 + 1))
    with pytest.raises(ValueError, match="recipe exceeds"):
        junit.load_junit_recipe(path)
    path.write_bytes(b" " * (16 * 1024**2 + 1))
    spec = recipe()
    with pytest.raises(ValueError, match="checkpoint exceeds"):
        capture._load_role(path, spec)


def test_filtered_runtime_provenance_has_no_ambient_metadata(tmp_path, monkeypatch):
    spec = recipe()
    calls = []

    def query(base, args):
        calls.append(args)
        if args[0] == "image":
            return ["linux", "arm64", 1234, digest(b"owned image config")]
        if args[0] == "version":
            return ["24.0.2", "24.0.2", "arm64"]
        return "owned-kernel"

    monkeypatch.setattr(capture, "_docker_query", query)
    monkeypatch.setattr(capture.platform, "machine", lambda: "arm64")
    cache = SourceCache(tmp_path / "cache")
    receipt = capture._runtime_environment(("docker",), spec, cache)
    payload = cache.member_path(receipt, "execution-environment.json").read_bytes()
    metadata = json.loads(payload)
    assert metadata["image_size_bytes"] == 1234
    assert metadata["producer_python_version"]
    assert "RepoTags" not in str(calls)
    assert "RepoDigests" not in str(calls)
    assert "hostname" not in metadata
    wrong = changed(spec, runtime=changed(spec.runtime, emulated=True))
    with pytest.raises(ValueError, match="emulation profile"):
        capture._runtime_environment(("docker",), wrong, cache)
    monkeypatch.setattr(capture, "_docker_query", lambda *args: [])
    with pytest.raises(ValueError, match="runtime metadata"):
        capture._runtime_environment(("docker",), spec, cache)


def test_docker_query_checks_process_and_strict_json(monkeypatch):
    monkeypatch.setattr(
        capture, "collect_process", lambda *args, **kwargs: ProcessStreams(b'{"owned":1}\n', b"", 0, 1, "exited")
    )
    assert capture._docker_query(("docker",), ("version",)) == {"owned": 1}
    monkeypatch.setattr(capture, "collect_process", lambda *args, **kwargs: ProcessStreams(b"", b"", 1, 1, "exited"))
    with pytest.raises(OSError, match="runtime unavailable"):
        capture._docker_query(("docker",), ("version",))


def test_actual_role_command_stages_only_declared_inputs_and_guards(tmp_path, monkeypatch):
    spec, cache, environment = prepared_inputs(tmp_path)
    prepared = prepare_junit_inputs(spec, cache, environment)
    calls = []

    def collect(args, **kwargs):
        calls.append((args, kwargs))
        if "run" in args:
            assert "--network" in args
            assert args[args.index("--network") + 1] == "none"
            assert "--read-only" in args
            assert "--pull=never" in args
            assert args[args.index("--cap-drop") + 1] == "ALL"
            mounts = [args[index + 1] for index, value in enumerate(args) if value == "--mount"]
            assert len(mounts) == 4
            assert all("readonly" in value for value in mounts)
            return ProcessStreams(encoded(records()), b"", 0, 123, "exited")
        return ProcessStreams(b"owned-name", b"", 0, 1, "exited")

    monkeypatch.setattr(capture, "collect_process", collect)
    streams = capture._run_role(("docker",), spec, prepared, cache, "buggy", "owned-name")
    assert streams.elapsed_ns == 123
    assert calls[-1][0] == ("docker", "rm", "-f", "owned-name")
    assert calls[0][1]["maximum_output_bytes"] == spec.limits.maximum_output_bytes


@pytest.mark.parametrize(
    "instant",
    ["2026-01-01T00:00:00.123456789Z", "٢٠٢٦-01-01T00:00:00Z", "2026-01-01T00:00:00+00:00"],
)
def test_role_precision_and_unknown_schema_reject_before_truncation(instant):
    row = records()[1]
    row["capture_started_at"] = instant
    from coleman.transports.junit_contracts import JunitCase

    with pytest.raises(ValueError, match="six fractional digits"):
        JunitCase.model_validate(row)
    row["capture_started_at"] = "2026-01-01T00:00:00Z"
    row["schema_version"] = 2
    with pytest.raises(ValueError, match="version"):
        JunitCase.model_validate(row)


@pytest.mark.parametrize("launch_error", [False, True])
def test_cleanup_failure_leaves_recoverable_start_without_committing_a_role(tmp_path, monkeypatch, launch_error):
    spec, cache, environment = prepared_inputs(tmp_path)
    monkeypatch.setattr(capture.shutil, "which", lambda _: "/owned/docker")
    monkeypatch.setattr(capture, "_runtime_environment", lambda *args: environment)
    executions = []
    fail_cleanup = True

    def collect(args, **kwargs):
        if "run" in args:
            executions.append(args)
            return ProcessStreams(encoded(records()), b"", 0, 1, "exited")
        if fail_cleanup:
            if launch_error:
                raise OSError("Owned cleanup process could not start")
            return ProcessStreams(b"", b"owned daemon unavailable during cleanup", 1, 1, "exited")
        return ProcessStreams(b"", b"", 0, 1, "exited")

    monkeypatch.setattr(capture, "collect_process", collect)
    writer = ArtifactWriter(tmp_path / "results")
    with pytest.raises(OSError, match="cleanup failed"):
        capture.capture_junit_source(spec, cache, writer, "cleanup-trial")
    assert len(executions) == 1
    assert not list(writer.run_root.glob("junit-captures/*/buggy.json"))
    assert not list(writer.run_root.glob("corpus-controlled/*.json"))
    assert len(list(writer.run_root.glob("junit-captures/*/buggy.started.json"))) == 1
    fail_cleanup = False
    recovered = capture.capture_junit_source(spec, cache, writer, "cleanup-trial")
    assert len(executions) == 2
    assert [item.status for item in recovered.captures] == ["interrupted", "collected"]
    assert recovered.dataset.pairs[0].classification == "inconclusive"


@pytest.mark.parametrize("status", ["protocol_error", "process_error"])
def test_checkpoint_cannot_discard_complete_raw_observations_by_changing_status(tmp_path, status):
    spec = recipe()
    now = datetime(2026, 1, 1, tzinfo=UTC)
    packet = capture.role_capture(
        spec,
        digest(b"context"),
        "owned-trial",
        "buggy",
        now,
        now,
        ProcessStreams(encoded(records()), b"", 0, 1, "exited"),
    )
    payload = packet.model_dump(mode="json")
    payload.update(status=status, header=None, cases=[])
    path = tmp_path / "edited-role.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        capture._load_role(path, spec)


@pytest.mark.parametrize("known_exit,known_elapsed", [(True, False), (False, True), (True, True)])
def test_streamless_checkpoint_cannot_retain_observed_process_facts(tmp_path, known_exit, known_elapsed):
    spec = recipe()
    now = datetime(2026, 1, 1, tzinfo=UTC)
    packet = capture.role_capture(
        spec,
        digest(b"context"),
        "owned-trial",
        "buggy",
        now,
        now,
        ProcessStreams(encoded(records()), b"", 0, 1, "exited"),
    )
    payload = packet.model_dump(mode="json")
    payload.update(status="process_error", stdout_base64=None, stderr_base64=None, header=None, cases=[])
    if not known_exit:
        payload["exit_code"] = None
    if not known_elapsed:
        payload["producer_elapsed"].update(value=None, status="unavailable")
    path = tmp_path / "edited-streamless-role.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        capture._load_role(path, spec)


@pytest.mark.parametrize("missing_fact", ["exit_code", "producer_elapsed"])
def test_observed_streams_cannot_discard_process_facts(tmp_path, missing_fact):
    spec = recipe()
    now = datetime(2026, 1, 1, tzinfo=UTC)
    packet = capture.role_capture(
        spec,
        digest(b"context"),
        "owned-trial",
        "buggy",
        now,
        now,
        ProcessStreams(b"owned process failure", b"", 7, 0, "exited"),
    )
    payload = packet.model_dump(mode="json")
    if missing_fact == "exit_code":
        payload[missing_fact] = None
    else:
        payload[missing_fact].update(value=None, status="unavailable")
    path = tmp_path / "edited-observed-role.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        capture._load_role(path, spec)


@pytest.mark.parametrize("status", ["timeout", "output_limit"])
def test_checkpoint_guard_labels_must_agree_with_observed_bounds(tmp_path, status):
    spec = recipe()
    now = datetime(2026, 1, 1, tzinfo=UTC)
    packet = capture.role_capture(
        spec,
        digest(b"context"),
        "owned-trial",
        "buggy",
        now,
        now,
        ProcessStreams(encoded(records()), b"", 0, 0, "exited"),
    )
    payload = packet.model_dump(mode="json")
    payload.update(status=status, header=None, cases=[])
    path = tmp_path / "edited-guard-role.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        capture._load_role(path, spec)
