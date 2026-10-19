"""Authored CTest evidence: precise states, incomplete logs and unchanged legacy HCS meaning."""

import itertools
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import pytest

from coleman.corpus.hcs_contracts import HCSJob, HCSReceipt
from coleman.corpus.hcs_raw import read_hcs_slice
from coleman.corpus.hcs_trace import parse_hcs_ctest_trace, validate_hcs_ctest_trace
from coleman.corpus.hcs_trace_contracts import HCSTraceEvidence
from coleman.corpus.hcs_trace_correspondence import HCSTraceCorrespondence, corroborate_hcs_ctest_trace
from coleman.selection.contracts import Quantity, UnitSemantics
from tests.corpus.hcs_support import prepare_hcs

OWNED = b"""Test project /owned/ci/build
1/3 Test #1: A ................... Passed 0.00 sec
2/3 Test #2: B ...................***Failed 2.00 sec
3/3 Test #3: C ................... Passed 3.00 sec
67% tests passed, 1 tests failed out of 3
Total Test time (real) = 5.02 sec
"""


def owned_job():
    """Return one authored metadata identity; these dates do not become availability clocks."""
    return HCSJob(
        job_id=1,
        pipeline_id=101,
        commit_id="a" * 40,
        variant="linux.x",
        status="failed",
        started_at="2020-01-01T00:00:00Z",
        finished_at="2020-01-01T00:01:00Z",
    )


def changed(value, **fields):
    """Revalidate a complete typed copy rather than bypassing nested contract invariants."""
    return type(value).model_validate({**value.model_dump(), **fields})


def test_authored_versioned_golden_roundtrip_and_independent_expected_states():
    trace = parse_hcs_ctest_trace(OWNED, owned_job())
    golden = Path(__file__).with_name("fixtures") / "hcs-ctest-owned-v1.json"
    assert trace.to_json() + "\n" == golden.read_text()
    assert HCSTraceEvidence.from_json(golden.read_bytes()) == trace
    invocation = trace.runs[0]
    assert invocation.state == "complete"
    assert [row.binary_verdict for row in invocation.executions] == [0, 1, 0]
    assert invocation.executions[0].elapsed.value == 0
    assert invocation.wall is not None
    assert invocation.wall.value == Decimal("5.02")
    assert trace.admission == "not_granted"
    assert trace.outcome_availability == trace.predecision_membership == "unverified"


@pytest.mark.parametrize("version", [None, 0, 2, True, "1"])
def test_unknown_future_missing_and_coerced_versions_reject(version):
    payload = json.loads(parse_hcs_ctest_trace(OWNED, owned_job()).to_json())
    if version is None:
        del payload["schema_version"]
    else:
        payload["schema_version"] = version
    wire = json.dumps(payload)
    with pytest.raises(ValueError):
        HCSTraceEvidence.from_json(wire)


@pytest.mark.parametrize(
    "raw,status,verdict,failed",
    [
        ("Passed", "passed", 0, 0),
        ("***Failed", "failed", 1, 1),
        ("***Timeout", "timeout", None, 1),
        ("***Not Run", "not_run", None, 1),
        ("***Exception: SegFault", "exception", None, 1),
        ("***Skipped", "skipped", None, 0),
    ],
)
def test_zero_is_reported_while_nonbinary_process_states_remain_unavailable(raw, status, verdict, failed):
    payload = (
        "Test project /owned\n"
        f"1/1 Test #1: a ..............{raw} 0.00 sec\n"
        f"0% tests passed, {failed} tests failed out of 1\n"
        "Total Test time (real) = 0.00 sec\n"
    ).encode()
    row = parse_hcs_ctest_trace(payload, owned_job()).runs[0].executions[0]
    assert row.status == status
    assert row.binary_verdict == verdict
    assert row.elapsed.value == 0


@pytest.mark.parametrize("seconds", ["NaN", "inf", "Infinity", "-1", "-0.0", "1e3", "0.0000000000001"])
def test_nonfinite_negative_and_unsupported_precision_do_not_become_zero(seconds):
    payload = OWNED.replace(b"0.00 sec", seconds.encode() + b" sec")
    with pytest.raises(ValueError):
        parse_hcs_ctest_trace(payload, owned_job())


def test_every_interrupted_line_prefix_remains_partial_until_the_wall_summary():
    lines = OWNED.splitlines(keepends=True)
    for count in range(1, len(lines)):
        trace = parse_hcs_ctest_trace(b"".join(lines[:count]), owned_job())
        assert trace.runs[0].state == "partial"
        assert trace.runs[0].wall is None
    empty = parse_hcs_ctest_trace(b"configuration failed\n", owned_job())
    assert empty.runs == ()
    assert parse_hcs_ctest_trace(OWNED, owned_job()).runs[0].state == "complete"


@pytest.mark.parametrize(
    "payload",
    [
        OWNED.replace(b"Test project /owned/ci/build\n", b""),
        OWNED.replace(b"#2: B", b"#1: B"),
        OWNED.replace(b"2/3 Test", b"2/4 Test"),
        OWNED.replace(b"#2: B", b"#2: A"),
        OWNED.replace(b"***Failed", b"Unknown"),
        OWNED.replace(b"1 tests failed", b"0 tests failed"),
        OWNED.replace(b"1/3 Test #1:", b"1/3 Test #1:" * 2),
        OWNED + b"1/3 Test #1: A .... Passed 0.00 sec\n",
        OWNED + b"Total Test time (real) = 0.00 sec\n",
    ],
)
def test_conflicting_duplicate_malformed_and_reordered_evidence_reject(payload):
    with pytest.raises(ValueError):
        parse_hcs_ctest_trace(payload, owned_job())


def test_ansi_noise_and_parallel_result_order_preserve_value_independent_execution_identity():
    base = parse_hcs_ctest_trace(OWNED, owned_job())
    lines = OWNED.splitlines(keepends=True)
    identities = {row.index: base.execution_id("owned", 1, row.index) for row in base.runs[0].executions}
    for order in itertools.permutations(lines[1:4]):
        payload = b"owned build output\n\x1b[32m" + lines[0] + b"\x1b[0m" + b"".join(order) + b"".join(lines[4:])
        trace = parse_hcs_ctest_trace(payload, owned_job())
        assert trace.runs[0].state == "complete"
        assert trace.source_sha256 != base.source_sha256
        assert {row.index: trace.execution_id("owned", 1, row.index) for row in trace.runs[0].executions} == identities
    altered = parse_hcs_ctest_trace(OWNED.replace(b"2.00 sec", b"2.50 sec"), owned_job())
    assert altered.fingerprint != base.fingerprint
    assert altered.execution_id("owned", 1, 2) == base.execution_id("owned", 1, 2)


def test_repeated_invocations_are_separate_and_identical_concurrent_parses_are_idempotent():
    repeated = parse_hcs_ctest_trace(OWNED + OWNED, owned_job())
    assert len(repeated.runs) == 2
    assert repeated.execution_id("owned", 1, 1) != repeated.execution_id("owned", 2, 1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda _: parse_hcs_ctest_trace(OWNED, owned_job()).to_json(), range(12)))
    assert len(set(values)) == 1


def test_parallel_completion_counter_is_distinct_from_original_test_index(tmp_path):
    recipe, cache = prepare_hcs(tmp_path)
    source = read_hcs_slice(recipe, cache)
    lines = OWNED.splitlines(keepends=True)
    completion_order = [lines[3], lines[1], lines[2]]
    result_lines = [
        str(position).encode() + b"/" + line.split(b"/", 1)[1] for position, line in enumerate(completion_order, 1)
    ]
    payload = lines[0] + b"".join(result_lines) + b"".join(lines[4:])
    trace = parse_hcs_ctest_trace(payload, source.jobs[0])
    assert [row.index for row in trace.runs[0].executions] == [3, 1, 2]
    assert [row.completion_position for row in trace.runs[0].executions] == [1, 2, 3]
    assert trace.runs[0].state == "complete"
    assert corroborate_hcs_ctest_trace(trace, source).matched_observations == 3
    baseline = parse_hcs_ctest_trace(OWNED, source.jobs[0])
    assert trace.execution_id("owned", 1, 3) == baseline.execution_id("owned", 1, 3)


def test_filtered_invocation_retains_sparse_original_indices_larger_than_result_count():
    raw = b"""Test project /owned
1/2 Test #7: one ........ Passed 0.00 sec
2/2 Test #10: another ... Passed 1.00 sec
100% tests passed, 0 tests failed out of 2
Total Test time (real) = 1.02 sec
"""
    invocation = parse_hcs_ctest_trace(raw, owned_job()).runs[0]
    assert invocation.state == "complete"
    assert invocation.expected_results == 2
    assert [row.index for row in invocation.executions] == [7, 10]


@pytest.mark.parametrize("project,ordinal,index", [("", 1, 1), (" x", 1, 1), ("x", True, 1), ("x", 3, 1)])
def test_logical_identity_rejects_ambiguous_coordinates(project, ordinal, index):
    trace = parse_hcs_ctest_trace(OWNED, owned_job())
    with pytest.raises(ValueError):
        trace.execution_id(project, ordinal, index)


def test_incompatible_units_forged_state_and_raw_status_cannot_survive_reading():
    trace = parse_hcs_ctest_trace(OWNED, owned_job())
    invocation = trace.runs[0]
    row = invocation.executions[0]
    milliseconds = Quantity(value=Decimal(0), semantics=UnitSemantics(unit="ms", meaning="owned"))
    with pytest.raises(ValueError, match="units"):
        changed(row, elapsed=milliseconds)
    with pytest.raises(ValueError, match="status"):
        changed(row, raw_status="***Failed")
    with pytest.raises(ValueError, match="completion"):
        changed(invocation, state="partial")


def test_resource_and_encoding_bounds_reject_before_claiming_a_completed_result():
    for payload in [b"x" * (2 * 1024**2 + 1), b"x" * 16385, b"\n" * 131073, b"\x00", OWNED * 17]:
        with pytest.raises(ValueError):
            parse_hcs_ctest_trace(payload, owned_job())
    with pytest.raises(UnicodeDecodeError):
        parse_hcs_ctest_trace(b"\xff", owned_job())


def test_actual_raw_bytes_are_required_to_validate_evidence_not_self_declared_hashes():
    trace = parse_hcs_ctest_trace(OWNED, owned_job())
    validate_hcs_ctest_trace(trace, OWNED)
    changed_bytes = OWNED.replace(b"2.00 sec", b"2.01 sec")
    with pytest.raises(ValueError, match="original log"):
        validate_hcs_ctest_trace(trace, changed_bytes)


def test_repeated_long_malformed_lines_do_not_cause_wildcard_regex_backtracking():
    script = r"""
from coleman.corpus.hcs_trace import parse_hcs_ctest_trace
from coleman.corpus.hcs_contracts import HCSJob
job=HCSJob(job_id=1,pipeline_id=1,commit_id='a'*40,variant='owned',status='failed',started_at=None,finished_at=None)
raw=('Test project /owned\n1/1 Test #1: '+('a .... '*1500)+'invalid\n').encode()
for _ in range(100):
    try:
        parse_hcs_ctest_trace(raw,job)
    except ValueError:
        continue
    raise AssertionError('Malformed result was accepted')
print('bounded malformed results rejected')
"""
    process = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    assert process.stdout.strip() == "bounded malformed results rejected"


def test_original_csv_correspondence_preserves_exact_duplicates_and_old_receipts(tmp_path):
    recipe, cache = prepare_hcs(tmp_path)
    source = read_hcs_slice(recipe, cache)
    original = source.model_dump_json()
    trace = parse_hcs_ctest_trace(OWNED, source.jobs[0])
    proof = corroborate_hcs_ctest_trace(trace, source)
    assert proof.matched_observations == 3
    assert proof.duplicate_rows == 1
    assert proof.legacy_nonpass_rows == 1
    assert proof.admission == "not_granted"
    assert HCSTraceCorrespondence.from_json(proof.to_json()) == proof
    assert corroborate_hcs_ctest_trace(trace, source).to_json() == proof.to_json()
    assert source.model_dump_json() == original
    old = Path(__file__).with_name("fixtures") / "hcs-owned-completion-v1.json"
    assert HCSReceipt.from_json(old.read_bytes()).to_json() + "\n" == old.read_text()


@pytest.mark.parametrize("kind", ["duration", "verdict", "coverage", "partial", "repeated", "metadata"])
def test_correspondence_rejects_unknown_or_ambiguous_original_evidence(tmp_path, kind):
    recipe, cache = prepare_hcs(tmp_path)
    source = read_hcs_slice(recipe, cache)
    payload = OWNED
    if kind == "duration":
        payload = payload.replace(b"2.00 sec", b"2.01 sec")
    elif kind == "verdict":
        payload = payload.replace(b"***Failed", b"Passed").replace(b"1 tests failed", b"0 tests failed")
    elif kind == "coverage":
        payload = payload.replace(b"#1: A", b"#1: unknown")
    elif kind == "partial":
        payload = b"\n".join(payload.splitlines()[:-1])
    elif kind == "repeated":
        payload += OWNED
    job = source.jobs[0]
    if kind == "metadata":
        job = changed(job, variant="other")
    trace = parse_hcs_ctest_trace(payload, job)
    with pytest.raises(ValueError):
        corroborate_hcs_ctest_trace(trace, source)


@pytest.mark.parametrize("position", [True, 1.0, "1", 0, -1, 500001])
def test_source_positions_do_not_coerce_boolean_float_or_text_into_provenance(tmp_path, position):
    recipe, cache = prepare_hcs(tmp_path)
    source = read_hcs_slice(recipe, cache)
    trace = parse_hcs_ctest_trace(OWNED, source.jobs[0])
    proof = corroborate_hcs_ctest_trace(trace, source)
    positions = (position,) + proof.physical_positions[1:]
    with pytest.raises(ValueError):
        changed(proof, physical_positions=positions)
