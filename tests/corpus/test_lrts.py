"""Independent owned LRTS source semantics, typed boundaries and immutable cache correspondence."""

import csv
import hashlib
import io
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from coleman.artifacts import ArtifactWriter
from coleman.corpus import MaterializationSpec, SourceCache, load_registry, materialize_source
from coleman.corpus.lrts import convert_lrts_source, publish_lrts_dataset, validate_lrts_dataset
from coleman.corpus.lrts_contracts import METADATA_COLUMNS, LRTSCase, LRTSDataset, LRTSSpec


def metadata_row():
    """Own the source labels and counts; no downloaded rows or historical clocks are reused."""
    row = dict.fromkeys(METADATA_COLUMNS, "")
    row.update(
        project="owned-ci",
        pr_name="PR-owned",
        build_id="1",
        stage_id="single",
        build_duration="4",
        build_timestamp="0",
        build_result="ABORTED",
        ts_duration="2.5",
        passcount="2",
        failcount="1",
        skipcount="1",
        stage_duration_by_suite_sum="2.5",
        test_suite_duration_s="2.5",
        num_pass_class="1",
        num_fail_class="1",
        num_trans_class="1",
        num_pass_method="2",
        num_fail_method="1",
    )
    return row


def raw_report():
    """Own repeated methods, distinct raw nodes, observed zero and unavailable/unknown observations."""
    return {
        "_class": "hudson.tasks.junit.TestResult",
        "duration": 2.5,
        "empty": False,
        "passCount": 2,
        "failCount": 1,
        "skipCount": 1,
        "suites": [
            {
                "name": "owned.Suite",
                "id": None,
                "nodeId": "node-a",
                "enclosingBlocks": ["block-a"],
                "enclosingBlockNames": ["owned linux"],
                "timestamp": None,
                "duration": 2.5,
                "cases": [
                    {
                        "className": "owned.Repeated",
                        "name": "method",
                        "duration": 0,
                        "status": "PASSED",
                        "skipped": False,
                    },
                    {
                        "className": "owned.Repeated",
                        "name": "method",
                        "duration": 2.5,
                        "status": "REGRESSION",
                        "skipped": False,
                    },
                ],
            },
            {
                "name": "owned.Suite",
                "id": "suite-b",
                "nodeId": "node-b",
                "enclosingBlocks": None,
                "enclosingBlockNames": [],
                "timestamp": "",
                "duration": None,
                "cases": [
                    {"className": "owned.Unknown", "name": "method", "duration": None, "status": None, "skipped": True},
                    {
                        "className": "owned.Unknown",
                        "name": "method",
                        "duration": None,
                        "status": None,
                        "skipped": False,
                    },
                ],
            },
        ],
    }


def metadata_bytes(rows):
    """Encode the exact declared metadata layout with standard CSV framing."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=METADATA_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def materialized(tmp_path, *, rows=None, report=None, metadata_payload=None, report_payload=None):
    """Use real owned materialization for two independent pinned sources in one external cache."""
    outside = {**metadata_row(), "project": "other-ci", "build_id": "2", "ts_duration": "uninterpreted"}
    metadata_payload = (
        metadata_payload
        if metadata_payload is not None
        else metadata_bytes(rows if rows is not None else [metadata_row(), outside])
    )
    report_payload = (
        report_payload
        if report_payload is not None
        else json.dumps(report if report is not None else raw_report(), sort_keys=True, allow_nan=False).encode()
    )
    cache = SourceCache(tmp_path / "cache")
    receipts = []
    for label, payload, member in (
        ("metadata", metadata_payload, "metadata.csv"),
        ("report", report_payload, "report.json"),
    ):
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        package = MaterializationSpec.model_validate(
            {
                "schema_version": 1,
                "package_id": f"owned-lrts-{label}-v1",
                "archive": "raw",
                "source": {
                    "uri": f"urn:coleman:owned:lrts-{label}-v1",
                    "version": "owned-lrts-format-v1",
                    "citation": "Repository-authored schema fixture; not third-party LRTS data",
                    "sha256": digest,
                    "size_bytes": len(payload),
                    "availability": "not_verified",
                },
                "license": load_registry().card("ci-serial-boundary-v1").manifest.license,
                "members": [{"name": member, "sha256": digest, "size_bytes": len(payload)}],
            }
        )
        receipts.append(materialize_source(package, cache, lambda payload=payload: io.BytesIO(payload)))
    spec = LRTSSpec(
        schema_version=1,
        metadata_materialization=receipts[0],
        metadata_member="metadata.csv",
        report_materialization=receipts[1],
        report_member="report.json",
        project="owned-ci",
        pull_request="PR-owned",
        build_id=1,
        report_origin_path="owned-ci/PR/PR-owned/testReport_build1.json",
        history="constructed",
    )
    return spec, cache


def test_owned_roundtrip_golden_source_counts_and_temporal_blocker(tmp_path):
    spec, cache = materialized(tmp_path)
    dataset = convert_lrts_source(spec, cache)
    assert dataset.metadata_input_rows == 2
    assert dataset.metadata_outside_build == 1
    assert dataset.metadata_duplicates_excluded == 0
    assert len(dataset.cases) == 4
    assert dataset.repeated_method_labels == 2
    assert tuple(case.outcome for case in dataset.cases) == ("passed", "failed", "not_run", "unknown")
    assert dataset.metadata[0].build_result == "ABORTED"
    assert dataset.feedback_scope == "reported_cases_only"
    assert dataset.suites[1].enclosing_blocks is None
    assert dataset.suites[1].enclosing_block_names == ()
    assert dataset.suites[1].raw_timestamp == ""
    assert dataset.cases[0].duration.value == 0
    assert dataset.cases[0].duration.status == "observed"
    assert dataset.cases[2].duration.value is None
    assert dataset.cases[2].duration.status == "unavailable"
    assert LRTSDataset.from_json(dataset.to_json()) == dataset
    golden = Path(__file__).with_name("fixtures") / "lrts-owned-slice-v1.json"
    assert dataset.to_json() + "\n" == golden.read_text()
    validate_lrts_dataset(dataset, cache)
    with pytest.raises(ValueError, match="availability"):
        dataset.require_temporal_evaluation()


@pytest.mark.parametrize(
    "text",
    ["NaN", "Infinity", "-1", "1e-999999999", "1e999999999", "١", "1_0", "+1", " 1", "0.10000000000000000000001"],
)
def test_numeric_source_rejects_nonfinite_underflow_rounding_and_non_json_spellings(tmp_path, text):
    row = {**metadata_row(), "stage_duration_by_suite_sum": text}
    spec, cache = materialized(tmp_path, rows=[row])
    with pytest.raises(ValueError):
        convert_lrts_source(spec, cache)


@pytest.mark.parametrize("text", ["-1", "1.0", "01", "true", "null", "9223372036854775808"])
def test_method_counts_require_exact_nonnegative_integer_tokens(tmp_path, text):
    row = {**metadata_row(), "num_pass_method": text}
    spec, cache = materialized(tmp_path, rows=[row])
    with pytest.raises(ValueError):
        convert_lrts_source(spec, cache)


def test_binary64_precision_nulls_signed_zero_and_clock_meaning_are_preserved(tmp_path):
    row = {
        **metadata_row(),
        "stage_duration_by_suite_sum": "846.3639999999999",
        "test_suite_duration_s": "",
        "build_timestamp": "-0.0",
    }
    spec, cache = materialized(tmp_path, rows=[row])
    dataset = convert_lrts_source(spec, cache)
    context = dataset.metadata[0]
    assert context.measurement("lrts_suite_duration_sum").value == 846.3639999999999
    assert context.measurement("lrts_processed_class_duration_sum").value is None
    assert context.measurement("lrts_processed_class_duration_sum").status == "unavailable"
    scheduled = context.measurement("lrts_scheduled_epoch_seconds")
    assert scheduled.value == 0
    assert scheduled.meaning == "lrts-source-truncated-scheduled-unix-epoch-seconds-binary64-v1"
    assert '"value":-0.0' not in dataset.to_json()
    with pytest.raises(KeyError):
        context.measurement("missing")
    fractional = {**metadata_row(), "build_duration": "1.5"}
    other, other_cache = materialized(tmp_path / "fractional", rows=[fractional])
    with pytest.raises(ValueError, match="whole seconds"):
        convert_lrts_source(other, other_cache)


@pytest.mark.parametrize(
    "status,skipped,outcome",
    [
        ("PASSED", False, "passed"),
        ("FIXED", False, "passed"),
        ("FAILED", False, "failed"),
        ("REGRESSION", False, "failed"),
        ("SKIPPED", True, "not_run"),
        (None, False, "unknown"),
    ],
)
def test_source_statuses_never_infer_unique_faults_or_flakiness(tmp_path, status, skipped, outcome):
    spec, cache = materialized(tmp_path)
    original = convert_lrts_source(spec, cache).cases[0]
    payload = {**original.model_dump(), "source_status": status, "skipped": skipped}
    assert LRTSCase.model_validate(payload).outcome == outcome
    invalid = {**original.model_dump(), "source_status": "FAILED", "skipped": True}
    with pytest.raises(ValueError, match="contradicts"):
        LRTSCase.model_validate(invalid)


def test_unknown_versions_units_and_unchecked_artifacts_reject(tmp_path):
    spec, cache = materialized(tmp_path)
    dataset = convert_lrts_source(spec, cache)
    for changed in (
        {"schema_version": 2},
        {"schema_version": True},
        {"recipe": "lrts-source-assessment-v2"},
        {"report_origin_path": "other/PR/PR-owned/testReport_build1.json"},
        {"metadata_member": "missing"},
    ):
        payload = {**spec.model_dump(), **changed}
        with pytest.raises(ValueError):
            LRTSSpec.model_validate(payload)
    for version in (None, 0, 2, True):
        payload = {**dataset.model_dump(), "schema_version": version}
        with pytest.raises(ValueError):
            LRTSDataset.model_validate(payload)
    payload = dataset.model_dump()
    payload["suites"][0]["cases"][0]["duration"]["unit"] = "ms"
    with pytest.raises(ValueError, match="units"):
        LRTSDataset.model_validate(payload)
    unchecked = dataset.model_copy(update={"metadata_input_rows": 999})
    with pytest.raises(ValueError, match="conservation"):
        unchecked.to_json()
    with pytest.raises(ValueError):
        LRTSDataset.from_json('{"BuildId":1,"Duration":0,"Verdict":0}')


def test_duplicate_metadata_conservation_and_conflicts(tmp_path):
    row = metadata_row()
    spec, cache = materialized(tmp_path, rows=[row, row])
    dataset = convert_lrts_source(spec, cache)
    assert dataset.metadata_input_rows == 2
    assert dataset.metadata_duplicates_excluded == 1
    assert len(dataset.metadata) == 1
    conflicting = {**row, "build_result": "SUCCESS"}
    other, other_cache = materialized(tmp_path / "conflict", rows=[row, conflicting])
    with pytest.raises(ValueError, match="Conflicting"):
        convert_lrts_source(other, other_cache)


def test_declared_and_raw_count_duration_or_schema_contradictions_reject(tmp_path):
    for label, mutate in (
        ("count", lambda report: report.update(passCount=3)),
        ("duration", lambda report: report.update(duration=math.nextafter(2.5, math.inf))),
        ("class", lambda report: report.update(_class="future.JenkinsReport")),
    ):
        report = raw_report()
        mutate(report)
        spec, cache = materialized(tmp_path / label, report=report)
        with pytest.raises(ValueError):
            convert_lrts_source(spec, cache)


@pytest.mark.parametrize(
    "payload",
    [
        b"{}",
        b"[]",
        b"null",
        b'{"suites":null}',
        b'{"suites":[],"suites":[]}',
        b"NaN",
        b'{"suites":[{}]}',
        b'{"suites":[{"cases":null}]}',
    ],
)
def test_wrong_or_duplicate_json_contracts_fail_without_artifacts(tmp_path, payload):
    spec, cache = materialized(tmp_path, report_payload=payload)
    with pytest.raises(ValueError):
        convert_lrts_source(spec, cache)


def test_diagnostics_and_plugin_payloads_do_not_become_decision_features(tmp_path):
    report = raw_report()
    report["suites"][0]["cases"][0].update(
        stdout="owned diagnostic, not a feature", age=-1, failedSince=12, testActions=[{"owned_action": 7}]
    )
    spec, cache = materialized(tmp_path, report=report)
    dataset = convert_lrts_source(spec, cache)
    assert "owned diagnostic" not in dataset.to_json()
    assert "owned_action" not in dataset.to_json()
    assert dataset.cases[0].outcome == "passed"


@pytest.mark.parametrize(
    "token",
    [
        b"false",
        b'"1"',
        b"NaN",
        b"Infinity",
        b"-1",
        b"1e-999999999",
        b"1e999999999999999999999",
        b"1000000000001",
        b"0.10000000000000000000001",
    ],
)
def test_raw_duration_tokens_do_not_coerce_or_silently_round(tmp_path, token):
    payload = json.dumps(raw_report(), sort_keys=True, separators=(",", ":")).encode()
    assert b'"duration":0' in payload
    payload = payload.replace(b'"duration":0', b'"duration":' + token, 1)
    spec, cache = materialized(tmp_path, report_payload=payload)
    with pytest.raises(ValueError):
        convert_lrts_source(spec, cache)


def test_empty_report_remains_reported_empty_not_a_complete_passing_suite(tmp_path):
    row = metadata_row()
    for column in (
        "passcount",
        "failcount",
        "skipcount",
        "num_pass_class",
        "num_fail_class",
        "num_trans_class",
        "num_pass_method",
        "num_fail_method",
        "ts_duration",
        "stage_duration_by_suite_sum",
        "test_suite_duration_s",
    ):
        row[column] = "0"
    report = {
        "_class": "hudson.tasks.junit.TestResult",
        "duration": 0,
        "empty": True,
        "passCount": 0,
        "failCount": 0,
        "skipCount": 0,
        "suites": [],
    }
    spec, cache = materialized(tmp_path, rows=[row], report=report)
    dataset = convert_lrts_source(spec, cache)
    assert dataset.cases == ()
    assert dataset.reported_empty is True
    assert dataset.feedback_scope == "reported_cases_only"
    with pytest.raises(ValueError, match="complete-suite"):
        dataset.require_temporal_evaluation()


def test_unknown_metadata_counts_are_not_filled_from_raw_report_aggregates(tmp_path):
    row = {**metadata_row(), "passcount": "", "num_pass_method": ""}
    spec, cache = materialized(tmp_path, rows=[row])
    dataset = convert_lrts_source(spec, cache)
    assert dataset.metadata[0].reported_pass_methods is None
    assert dataset.metadata[0].processed_pass_methods is None
    assert dataset.report_pass_methods == 2


def test_zero_build_labels_and_absent_selected_build_are_actionable_errors(tmp_path):
    for label, row in (
        ("zero", {**metadata_row(), "build_id": "0"}),
        ("outside", {**metadata_row(), "project": "other-ci"}),
    ):
        spec, cache = materialized(tmp_path / label, rows=[row])
        with pytest.raises(ValueError):
            convert_lrts_source(spec, cache)


def test_numeric_literal_limit_does_not_round_a_valid_quantity(tmp_path):
    valid = {**metadata_row(), "stage_duration_by_suite_sum": "1." + "0" * 62}
    spec, cache = materialized(tmp_path / "valid", rows=[valid])
    assert convert_lrts_source(spec, cache).metadata[0].measurement("lrts_suite_duration_sum").value == 1
    oversized = {**metadata_row(), "stage_duration_by_suite_sum": "1." + "0" * 63}
    other, other_cache = materialized(tmp_path / "oversized", rows=[oversized])
    with pytest.raises(ValueError, match="literal"):
        convert_lrts_source(other, other_cache)


def test_metadata_framing_and_field_limits_apply_before_selection(tmp_path):
    first = metadata_bytes([metadata_row()])
    header, row = first.split(b"\n", 1)
    payloads = (
        b"",
        header + b"\n\n",
        header + b"\n" + row.rstrip(b"\n") + b"\r" + row,
        metadata_bytes([{**metadata_row(), "project": "other-ci", "pr_base_branch": "x" * 1025}]),
        header + b"\n1,2\n",
    )
    for index, payload in enumerate(payloads):
        spec, cache = materialized(tmp_path / str(index), metadata_payload=payload)
        with pytest.raises(ValueError):
            convert_lrts_source(spec, cache)
    spec, cache = materialized(tmp_path / "crlf", metadata_payload=first.replace(b"\n", b"\r\n"))
    assert convert_lrts_source(spec, cache).metadata_input_rows == 1


def test_known_verdict_counts_cannot_be_changed_while_preserving_total(tmp_path):
    report = raw_report()
    report.update(passCount=0, failCount=2, skipCount=2)
    spec, cache = materialized(tmp_path, report=report)
    with pytest.raises(ValueError, match="known verdicts"):
        convert_lrts_source(spec, cache)


def test_selected_metadata_and_raw_counts_must_refer_to_the_same_report(tmp_path):
    row = {**metadata_row(), "passcount": "3"}
    spec, cache = materialized(tmp_path, rows=[row])
    with pytest.raises(ValueError, match="counts contradict"):
        convert_lrts_source(spec, cache)


@pytest.mark.parametrize(
    "path,value",
    [
        (("suites", 0, "position"), 9),
        (("suites", 0, "cases", 0, "position"), 9),
        (("metadata", 0, "project"), "other-ci"),
        (("metadata", 0, "measurements", 0, "name"), "unknown-measurement"),
        (("suites", 0, "cases", 0, "duration", "value"), 1000000000001.0),
    ],
)
def test_serialized_source_positions_and_metadata_bindings_cannot_be_forged(tmp_path, path, value):
    spec, cache = materialized(tmp_path)
    payload = convert_lrts_source(spec, cache).model_dump()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        LRTSDataset.model_validate(payload)


def test_serialized_metadata_cannot_duplicate_a_stage_as_another_observation(tmp_path):
    spec, cache = materialized(tmp_path)
    payload = convert_lrts_source(spec, cache).model_dump()
    payload["metadata"] *= 2
    payload["metadata_input_rows"] += 1
    with pytest.raises(ValueError, match="Duplicate"):
        LRTSDataset.model_validate(payload)


def test_resource_bounds_and_filtered_layout_rejection(tmp_path):
    spec, cache = materialized(tmp_path)
    for changed in (
        {"maximum_metadata_rows": 1},
        {"maximum_cases": 3},
        {"maximum_suites": 1},
        {"maximum_line_bytes": 128},
    ):
        limited = LRTSSpec.model_validate({**spec.model_dump(), **changed})
        with pytest.raises(ValueError):
            convert_lrts_source(limited, cache)
    payload = metadata_bytes([metadata_row()]).replace(b"num_fail_method", b"num_fail_method,filtered")
    other, other_cache = materialized(tmp_path / "filtered", metadata_payload=payload)
    with pytest.raises(ValueError, match="header"):
        convert_lrts_source(other, other_cache)


def test_metadata_order_json_key_order_and_numeric_spelling_preserve_execution_identity(tmp_path):
    a, cache = materialized(tmp_path / "a", rows=[metadata_row(), {**metadata_row(), "stage_id": "other-stage"}])
    report = json.dumps(raw_report(), indent=2).encode()
    b, other_cache = materialized(
        tmp_path / "b",
        rows=[{**metadata_row(), "stage_id": "other-stage"}, {**metadata_row(), "ts_duration": "2.5000"}],
        report_payload=report,
    )
    first = convert_lrts_source(a, cache)
    second = convert_lrts_source(b, other_cache)
    assert first.metadata == second.metadata
    assert first.suites == second.suites
    assert first.fingerprint != second.fingerprint

    def execution_ids(dataset):
        """Compare logical source executions independently of byte spelling and object-key order."""
        return tuple(
            case.execution_id("owned-ci", "PR-owned", 1, suite.position)
            for suite in dataset.suites
            for case in suite.cases
        )

    assert execution_ids(first) == execution_ids(second)


def test_model_invariants_are_not_source_correspondence(tmp_path):
    spec, cache = materialized(tmp_path)
    dataset = convert_lrts_source(spec, cache)
    payload = dataset.model_dump()
    payload["suites"][0]["cases"][0]["duration"]["value"] = 1.0
    changed = LRTSDataset.model_validate(payload)
    with pytest.raises(ValueError, match="source bytes"):
        validate_lrts_dataset(changed, cache)
    writer = ArtifactWriter(tmp_path / "exports")
    with pytest.raises(ValueError):
        publish_lrts_dataset(changed, cache, writer)
    assert not list(writer.run_root.rglob("*.json"))


def test_idempotent_concurrent_and_interrupted_publication(tmp_path, monkeypatch):
    spec, cache = materialized(tmp_path)
    dataset = convert_lrts_source(spec, cache)
    writer = ArtifactWriter(tmp_path / "exports")
    with ThreadPoolExecutor(max_workers=4) as executor:
        paths = list(executor.map(lambda _: publish_lrts_dataset(dataset, cache, writer), range(4)))
    assert len(set(paths)) == 1
    interrupted_writer = ArtifactWriter(tmp_path / "interrupted")
    real_link = os.link

    def interrupted(source, target, *args, **kwargs):
        if str(target).endswith(".json"):
            raise OSError("interrupted LRTS publication")
        return real_link(source, target, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "link", interrupted)
        with pytest.raises(OSError, match="interrupted"):
            publish_lrts_dataset(dataset, cache, interrupted_writer)
    assert not list(interrupted_writer.run_root.rglob("*.json"))
    assert not list(interrupted_writer.run_root.rglob("*.tmp"))
    assert publish_lrts_dataset(dataset, cache, interrupted_writer).exists()


def test_publication_containment_and_source_cache_mutation(tmp_path, monkeypatch):
    spec, cache = materialized(tmp_path)
    dataset = convert_lrts_source(spec, cache)
    overlap = ArtifactWriter(cache.root)
    with pytest.raises(ValueError, match="separate"):
        publish_lrts_dataset(dataset, cache, overlap)
    root = tmp_path / "exports"
    root.mkdir()
    (root / "corpus-lrts").symlink_to(cache.root, target_is_directory=True)
    writer = ArtifactWriter(root)
    with pytest.raises(ValueError, match="output tree"):
        publish_lrts_dataset(dataset, cache, writer)
    path = cache.member_path(spec.report_materialization, spec.report_member)
    data = path.read_bytes()
    real_open = Path.open
    reads = 0

    def changed_after_verification(self, *args, **kwargs):
        nonlocal reads
        if self == path and args and args[0] == "rb":
            reads += 1
            if reads == 3:
                with real_open(self, "wb") as stream:
                    stream.write(data.replace(b"2.5", b"2.6"))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", changed_after_verification)
    with pytest.raises(ValueError, match="Parsed LRTS report bytes"):
        convert_lrts_source(spec, cache)
