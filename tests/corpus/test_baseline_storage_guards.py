"""Frozen source correspondence and cooperative resource limits reject misleading completion evidence."""

import hashlib

import pytest

from coleman.application import baseline_resources
from coleman.application.baseline_execution import (
    NativeBaselineBinding,
    _verify_native_input,
    baseline_dataset_name,
    baseline_input_bytes,
    baseline_native_frame,
    read_baseline_plan,
)
from coleman.application.baseline_resources import BaselineLimitError, BaselineResourceGuard
from coleman.corpus.baseline_results import baseline_coordinates
from tests.corpus.baseline_support import baseline_plan, change


def binding_for(plan, payload):
    """Bind fixture bytes; the real verifier must still prove correspondence with the frozen source."""
    return NativeBaselineBinding(
        schema_version=1,
        plan_file="owned-plan.json",
        plan_fingerprint=plan.fingerprint,
        runtime_fingerprint="sha256:" + "a" * 64,
        coordinate=baseline_coordinates(plan)[0],
        input_sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
        input_size_bytes=len(payload),
    )


def test_native_bytes_are_deterministic_and_not_authorized_by_a_matching_caller_hash(tmp_path):
    plan = baseline_plan()
    payload = baseline_input_bytes(plan)
    assert baseline_input_bytes(plan) == payload
    path = tmp_path / "input.parquet"
    path.write_bytes(payload)
    binding = binding_for(plan, payload)
    _verify_native_input(path, binding, plan)
    path.write_bytes(payload[:-1])
    with pytest.raises(ValueError, match="size changed"):
        _verify_native_input(path, binding, plan)
    changed = bytes([payload[0] ^ 1]) + payload[1:]
    path.write_bytes(changed)
    with pytest.raises(ValueError, match="digest changed"):
        _verify_native_input(path, binding, plan)
    with pytest.raises(ValueError, match="complete frozen source"):
        _verify_native_input(path, binding_for(plan, changed), plan)


def test_plan_reader_and_unsafe_source_label_do_not_infer_paths(tmp_path):
    plan = baseline_plan()
    path = tmp_path / "plan.json"
    path.write_text(plan.to_json())
    assert read_baseline_plan(path) == plan
    changed = change(plan, source=change(plan.source, source_id="../outside"))
    assert baseline_dataset_name(changed) == "source-" + changed.source.fingerprint[7:]
    path.write_bytes(b" " * (16 * 1024**2 + 1))
    with pytest.raises(ValueError, match="exceeds"):
        read_baseline_plan(path)


def test_native_projection_never_imputes_missing_truth():
    plan = baseline_plan()
    build = plan.source.builds[0]
    first = build.tests[0]
    missing = change(first, failure=change(first.failure, value=None, status="unavailable"))
    plan = change(
        plan,
        source=change(plan.source, builds=(change(build, tests=(missing, *build.tests[1:])), *plan.source.builds[1:])),
    )
    with pytest.raises(ValueError, match="complete observed"):
        baseline_native_frame(plan)


@pytest.mark.parametrize(
    "peak,reason", [(None, "native_worker_peak_memory_unavailable"), (65.0, "native_worker_peak_memory_limit_exceeded")]
)
def test_resource_guard_preserves_unknown_or_excess_memory(monkeypatch, tmp_path, peak, reason):
    limits = change(baseline_plan().limits, maximum_peak_memory_mib=64.0)
    monkeypatch.setattr(baseline_resources, "_worker_peak_rss_mib", lambda: peak)
    guard = BaselineResourceGuard(limits, tmp_path)
    with pytest.raises(BaselineLimitError) as caught:
        guard.check()
    assert caught.value.reason == reason
    assert caught.value.peak_mib == peak
    assert caught.value.elapsed_seconds >= 0


def test_resource_guard_reserves_complete_receipt_bytes_and_entries(tmp_path):
    limits = change(baseline_plan().limits, maximum_output_bytes=1024**2, maximum_output_entries=64)
    guard = BaselineResourceGuard(limits, tmp_path)
    with pytest.raises(BaselineLimitError, match="native_output_limit_exceeded"):
        guard.check_publication(limits.maximum_output_bytes + 1)
    path = tmp_path / "native-output"
    path.write_bytes(b"0" * (limits.maximum_output_bytes + 1))
    with pytest.raises(BaselineLimitError, match="native_output_limit_exceeded"):
        guard.check()
    path.unlink()
    for index in range(64):
        (tmp_path / str(index)).touch()
    guard.check()
    with pytest.raises(BaselineLimitError, match="native_output_limit_exceeded"):
        guard.check_publication(100)
