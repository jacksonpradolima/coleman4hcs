"""Typed public metadata feasibility, explicit units, strict provenance and actual immutable local recovery."""

from __future__ import annotations

import json
import multiprocessing
from datetime import datetime
from pathlib import Path

import pytest

from coleman.application.baseline_lock import baseline_lock
from coleman.artifacts import ArtifactWriter
from tests.corpus.public_actions_support import example_module, request, response, snapshot, wire


@pytest.fixture
def module():
    """Use the real complete collector/normalizer, with no principal component replacement."""
    return example_module()


def install_owned_responses(module, monkeypatch, *, completed=True):
    """Replace only the external HTTP boundary; persistence, parsing, runtime capture and verification stay real."""
    values = dict(zip(("run", "jobs"), wire(completed=completed), strict=True))
    calls = []

    def fetch(url, part):
        calls.append(part)
        result = response(module, request(module), part, values[part])
        assert result.url == url
        return result

    monkeypatch.setattr(module, "fetch_response", fetch)
    return calls


def test_versioned_roundtrip_golden_and_explicit_missingness(module):
    value = snapshot(module)
    assert module.Snapshot.from_json(value.to_json()) == value
    fixture = Path(__file__).parent / "fixtures/public-actions-snapshot-v1.json"
    assert value.to_json() + "\n" == fixture.read_text()
    rows = {item.name: item for item in value.measurements}
    assert rows["job_99_recorded_envelope_seconds"].value == 60.0
    assert rows["job_99_recorded_envelope_seconds"].meaning != rows["occupied_runner_seconds"].meaning
    assert rows["occupied_runner_seconds"].value is None
    assert rows["occupied_runner_seconds"].status == "unavailable"
    assert rows["expected_test_count"].value is None
    assert rows["collector_policy_inference_seconds"].status == "not_applicable"
    assert len(value.reality) == 35
    assert next(row for row in value.reality if row.axis == "source_execution_model").status == "synthetic"
    assert next(row for row in value.reality if row.axis == "outcome_availability").values is None
    assert next(row for row in value.reality if row.axis == "shards").values is None
    assert value.upstream_first_available_at is None
    assert value.decision_sealed_at is None
    assert value.conclusion == "insufficient_evidence_for_forward_tcpci"


def test_zero_unknown_and_unavailable_are_distinct(module):
    zero = snapshot(module, zero=True)
    assert zero.jobs[0].started_at == zero.jobs[0].completed_at
    zero_measure = module.job_measurement(zero.jobs[0])
    assert zero_measure.value == 0.0
    assert zero_measure.status == "observed"
    pending = snapshot(module, completed=False)
    pending_measure = module.job_measurement(pending.jobs[0])
    assert pending_measure.value is None
    assert pending_measure.status == "unavailable"
    data = zero.jobs[0].model_dump()
    data["started_at"] = None
    unknown_job = module.Job.model_validate(data)
    unknown_measure = module.job_measurement(unknown_job)
    assert unknown_measure.value is None
    assert unknown_measure.status == "unknown"


def test_source_receipt_projection_does_not_backdate_a_label_or_claim_shadow_feedback(module):
    value = snapshot(module)
    history = value.history()
    assert len(history.revisions) == 1
    assert history.revisions[0].metadata.target_end == value.jobs[0].completed_at
    assert history.as_of(datetime.fromisoformat("2026-01-01T10:02:00Z")).observations == ()
    assert len(history.as_of(value.created_at).observations) == 1
    assert history.revisions[0].received_at == value.created_at
    assert snapshot(module, completed=False).history().revisions == ()
    assert history.revisions[0].measurement.meaning == "provider-recorded-job-envelope-seconds-v1"


@pytest.mark.parametrize("version", [0, 2, True, 1.0, "1"])
def test_unknown_or_untyped_artifact_versions_reject(module, version):
    payload = json.loads(snapshot(module).to_json())
    payload["schema_version"] = version
    with pytest.raises(ValueError):
        module.Snapshot.from_json(json.dumps(payload))


@pytest.mark.parametrize("mutation", ["unit", "meaning", "nonfinite", "negative", "physical_inference", "test_verdict"])
def test_contract_rejects_units_nonfinite_negative_work_and_unobserved_evidence(module, mutation):
    payload = json.loads(snapshot(module).to_json())
    item = next(row for row in payload["measurements"] if row["name"] == "job_99_recorded_envelope_seconds")
    if mutation == "unit":
        item["unit"] = "ms"
    elif mutation == "meaning":
        item["meaning"] = "physical-whole-runner-occupied-seconds-v1"
    elif mutation == "nonfinite":
        item["value"] = float("nan")
    elif mutation == "negative":
        item["value"] = -1.0
    elif mutation == "physical_inference":
        occupied = next(row for row in payload["measurements"] if row["name"] == "occupied_runner_seconds")
        occupied.update(value=60.0, status="observed")
    else:
        payload["measurements"].append({**item, "name": "failing_test_count", "value": 0.0})
    encoded = json.dumps(payload)
    with pytest.raises(ValueError):
        module.Snapshot.from_json(encoded)


def test_semantic_identity_ignores_json_object_order_and_raw_receipt_noise(module):
    original = snapshot(module)
    req = request(module)
    raw_run, raw_jobs = wire()
    run_reordered = dict(reversed(list(raw_run.items())))
    a = response(module, req, "run", raw_run)
    b = response(module, req, "run", run_reordered)
    assert a.body_sha256 != b.body_sha256
    assert module.parse_run(a, req) == module.parse_run(b, req)
    changed = module.Snapshot.model_validate(
        {
            **original.model_dump(),
            "responses": (b.fingerprint, original.responses[1]),
            "created_at": datetime.fromisoformat("2026-01-01T10:02:02Z"),
        }
    )
    assert changed.fingerprint != original.fingerprint
    assert changed.logical_id == original.logical_id
    assert changed.semantic_fingerprint == original.semantic_fingerprint
    raw_jobs["jobs"] *= 2
    raw_jobs["total_count"] = 2
    duplicated = response(module, req, "jobs", raw_jobs)
    jobs = module.parse_jobs(duplicated, original.run)
    duplicate_payload = {
        **original.model_dump(),
        "jobs": jobs,
        "measurements": module.measurements(jobs),
        "reality": module.reality(req, original.run, jobs),
    }
    with pytest.raises(ValueError, match="Duplicate provider job"):
        module.Snapshot.model_validate(duplicate_payload)


@pytest.mark.parametrize("mutation", ["private", "private_head", "missing_head", "wrong_run", "wrong_repo"])
def test_public_scope_rejects_before_raw_source_publication(module, tmp_path, monkeypatch, mutation):
    run, _ = wire()
    if mutation == "private":
        run["repository"]["private"] = True
    elif mutation == "private_head":
        run["head_repository"]["private"] = True
    elif mutation == "missing_head":
        run["head_repository"] = None
    elif mutation == "wrong_run":
        run["id"] = 43
    else:
        run["repository"]["full_name"] = "coleman-owned/other-ci"
    req = request(module)
    monkeypatch.setattr(module, "fetch_response", lambda url, part: response(module, req, part, run))
    with pytest.raises(ValueError):
        module.capture(req, tmp_path, fetch=True)
    assert list(tmp_path.rglob("run.json")) == []
    assert list(tmp_path.rglob("complete.json")) == []


@pytest.mark.parametrize(
    "mutation", ["truncated", "extra_count", "wrong_attempt", "wrong_revision", "negative_interval"]
)
def test_provider_inventory_and_source_correspondence_reject(module, mutation):
    req = request(module)
    raw_run, raw_jobs = wire()
    run = module.parse_run(response(module, req, "run", raw_run), req)
    if mutation == "truncated":
        raw_jobs["total_count"] = 101
    elif mutation == "extra_count":
        raw_jobs["total_count"] = 0
    elif mutation == "wrong_attempt":
        raw_jobs["jobs"][0]["run_attempt"] = 2
    elif mutation == "wrong_revision":
        raw_jobs["jobs"][0]["head_sha"] = "2" * 40
    else:
        raw_jobs["jobs"][0]["completed_at"] = "2026-01-01T09:59:00Z"
    value = response(module, req, "jobs", raw_jobs)
    with pytest.raises(ValueError):
        module.parse_jobs(value, run)


def test_actual_artifact_completion_exact_retry_and_corruption_rejection(module, tmp_path, monkeypatch):
    calls = install_owned_responses(module, monkeypatch)
    req = request(module)
    root, value = module.capture(req, tmp_path, fetch=True)
    assert calls == ["run", "jobs"]
    assert module.read_capture(root) == value
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert module.capture(req, tmp_path) == (root, value)
    assert calls == ["run", "jobs"]
    assert before == {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    original = (root / "jobs.json").read_bytes()
    (root / "jobs.json").write_bytes(original + b" ")
    # The versioned response's canonical fingerprint is unchanged by outer JSON whitespace;
    # original provider bytes remain a separately checked inner digest.
    assert module.read_capture(root) == value
    payload = json.loads(original)
    payload["body"] = payload["body"].replace("owned-unit-tests", "changed-unit-tests")
    (root / "jobs.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest differs"):
        module.read_capture(root)


def test_interrupted_completion_resumes_response_checkpoints_without_restamping(module, tmp_path, monkeypatch):
    calls = install_owned_responses(module, monkeypatch)
    req = request(module)
    actual = ArtifactWriter.write_immutable

    def interrupted(writer, path, content):
        if path.name == "complete.json":
            raise OSError("owned interrupted completion")
        actual(writer, path, content)

    monkeypatch.setattr(ArtifactWriter, "write_immutable", interrupted)
    with pytest.raises(OSError, match="owned interrupted"):
        module.capture(req, tmp_path, fetch=True)
    root = next(tmp_path.rglob("snapshot.json")).parent
    assert not (root / "complete.json").exists()
    before = {part: (root / (part + ".json")).read_bytes() for part in ("run", "jobs", "snapshot")}
    monkeypatch.setattr(ArtifactWriter, "write_immutable", actual)
    recovered_root, recovered = module.capture(req, tmp_path)
    assert recovered_root == root
    assert recovered == module.Snapshot.from_json(before["snapshot"])
    assert calls == ["run", "jobs"]
    assert before == {part: (root / (part + ".json")).read_bytes() for part in before}
    assert module.read_capture(root) == recovered


def test_conflicting_interrupted_projection_cannot_publish_completion(module, tmp_path, monkeypatch):
    install_owned_responses(module, monkeypatch)
    req = request(module)
    actual = ArtifactWriter.write_immutable

    def interrupted(writer, path, content):
        if path.name == "complete.json":
            raise OSError("owned interruption")
        actual(writer, path, content)

    monkeypatch.setattr(ArtifactWriter, "write_immutable", interrupted)
    with pytest.raises(OSError):
        module.capture(req, tmp_path, fetch=True)
    path = next(tmp_path.rglob("snapshot.json"))
    payload = json.loads(path.read_bytes())
    payload["jobs"][0]["name"] = "changed-but-otherwise-valid"
    path.write_text(module.Snapshot.model_validate(payload).to_json() + "\n")
    monkeypatch.setattr(ArtifactWriter, "write_immutable", actual)
    with pytest.raises(ValueError, match="actual source projection"):
        module.capture(req, tmp_path)
    assert not (path.parent / "complete.json").exists()


def hold_lock(path, ready, release):
    """Hold the actual existing kernel lock in another process; death/exit releases ownership."""
    with baseline_lock(Path(path)):
        ready.set()
        release.wait(15)


def test_real_process_concurrency_preserves_completed_source_receipts(module, tmp_path, monkeypatch):
    install_owned_responses(module, monkeypatch)
    req = request(module)
    root, value = module.capture(req, tmp_path, fetch=True)
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    owner = context.Process(target=hold_lock, args=(str(root / "capture.lock"), ready, release))
    owner.start()
    try:
        assert ready.wait(10)
        with pytest.raises(RuntimeError, match="capture is already active"):
            module.capture(req, tmp_path)
    finally:
        release.set()
        owner.join(10)
        if owner.is_alive():
            owner.kill()
            owner.join(5)
    assert owner.exitcode == 0
    assert module.capture(req, tmp_path) == (root, value)


def die_after_committed_run(output):
    """Leave a real durable response checkpoint while the kernel owns the collector lock, then die abruptly."""
    import os

    module = example_module()
    req = request(module)
    actual = ArtifactWriter.write_immutable

    def write_then_die(writer, path, content):
        actual(writer, path, content)
        if path.name == "run.json":
            os._exit(73)

    def owned_http(url, part):
        return response(module, req, part, wire()[0 if part == "run" else 1])

    ArtifactWriter.write_immutable = write_then_die
    module.fetch_response = owned_http
    module.capture(req, Path(output), fetch=True)


def test_process_death_releases_lock_and_preserves_first_committed_receipt(module, tmp_path, monkeypatch):
    """A killed collector cannot publish completion; recovery fetches only the missing response."""
    context = multiprocessing.get_context("spawn")
    owner = context.Process(target=die_after_committed_run, args=(str(tmp_path),))
    owner.start()
    owner.join(15)
    if owner.is_alive():
        owner.kill()
        owner.join(5)
    assert owner.exitcode == 73
    path = next(tmp_path.rglob("run.json"))
    original = path.read_bytes()
    assert not (path.parent / "complete.json").exists()
    calls = install_owned_responses(module, monkeypatch)
    root, recovered = module.capture(request(module), tmp_path, fetch=True)
    assert root == path.parent
    assert calls == ["jobs"]
    assert path.read_bytes() == original
    assert module.read_capture(root) == recovered
