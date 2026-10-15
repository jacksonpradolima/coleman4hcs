"""Native adapter execution, fault injection, metamorphic properties and import boundaries."""

import ast
import base64
import hashlib
import io
import json
import subprocess
import sys
import threading
from decimal import Decimal
from pathlib import Path

import pytest

from coleman.application import (
    ApplicationError,
    ApplicationService,
    Caller,
    Handle,
    Limits,
    Parameters,
    PortableNativeProvenance,
    ServerPolicy,
    Submit,
    default_registry,
)
from coleman.application.contracts import ExperimentResult, PortableFile, ResultBundle
from coleman.application.registry import Configuration, NativeConfiguration
from coleman.application.worker import WorkItem, _check_resources, _guard, execute, main
from coleman.selection import CoverageProblem, CoverageResult


def test_budget_monotonicity_and_decimal_spelling_metamorphic_properties():
    base = default_registry()[0]
    objectives = []
    for value in range(6):
        config = base.resolve(Parameters(budget=Decimal(value)), Limits())
        result = execute(WorkItem(configuration=config, limits=Limits())).result
        assert isinstance(result, CoverageResult)
        assert result.cost is not None
        assert result.objective is not None
        assert result.status == "optimal"
        assert result.cost.value <= value
        objectives.append(result.objective.value)
    assert objectives == sorted(objectives)
    first = base.resolve(Parameters(budget=Decimal("3.000")), Limits())
    second = base.resolve(Parameters(budget=Decimal("3")), Limits())
    assert first.fingerprint == second.fingerprint
    assert isinstance(base.problem, CoverageProblem)
    payload = base.problem.model_dump()
    payload["candidates"] = tuple(reversed(payload["candidates"]))
    payload["outcomes"] = tuple(reversed(payload["outcomes"]))
    permuted = Configuration(
        schema_version=1, spec_id=base.spec_id, kind="coverage", problem=CoverageProblem.model_validate(payload)
    )
    assert execute(WorkItem(configuration=base, limits=Limits())) == execute(
        WorkItem(configuration=permuted, limits=Limits())
    )


def test_native_execution_adapter_in_process_preserves_real_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    native = next(item for item in default_registry() if item.kind == "experiment")
    result = execute(WorkItem(configuration=native, limits=Limits()))
    assert isinstance(result.result, ExperimentResult)
    assert result.result.work_units == 2
    assert (tmp_path / "run" / result.result.run_id / "provenance.json").exists()
    assert ResultBundle.from_json(result.to_json()) == result


def test_portable_provenance_hides_host_context_and_explicitly_binds_original_bytes(tmp_path, monkeypatch):
    work = tmp_path / "host-private-root"
    work.mkdir()
    monkeypatch.chdir(work)
    bundle = execute(WorkItem(configuration=default_registry()[2], limits=Limits()))
    assert isinstance(bundle.result, ExperimentResult)
    file = next(item for item in bundle.result.files if item.name == "provenance.json")
    content = base64.b64decode(file.content_base64)
    portable = json.loads(content)
    parsed = PortableNativeProvenance.from_json(content)
    assert parsed.to_json().encode() == content
    assert str(work).encode() not in content
    assert portable["schema_version"] == 1
    assert portable["artifact_type"] == "portable-native-provenance-v1"
    assert portable["redacted_fields"] == ["cwd", "git", "uv_lock_hash"]
    original = (work / "run" / bundle.result.run_id / "provenance.json").read_bytes()
    assert json.loads(original)["cwd"] == str(work)
    assert portable["source_sha256"] == "sha256:" + hashlib.sha256(original).hexdigest()
    assert file.sha256 != hashlib.sha256(original).hexdigest()
    assert all(str(work).encode() not in base64.b64decode(item.content_base64) for item in bundle.result.files)
    future = json.dumps({**portable, "schema_version": 2})
    with pytest.raises(ValueError):
        PortableNativeProvenance.from_json(future)


def test_combined_native_and_bundle_output_limit_is_enforced(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    native = default_registry()[2]
    limits = Limits(max_output_bytes=20000)
    bundle = execute(WorkItem(configuration=native, limits=limits))
    assert isinstance(bundle.result, ExperimentResult)
    bundle_bytes = len(bundle.to_json().encode())
    assert bundle_bytes < limits.max_output_bytes
    assert bundle_bytes + sum(file.size_bytes for file in bundle.result.files) > limits.max_output_bytes
    caller = Caller(subject="local")
    with ApplicationService((native,), root=tmp_path / "service") as service:
        initial = service.submit(
            Submit(schema_version=1, request_id="output", spec_id=native.spec_id, limits=limits), caller=caller
        )
        final = service.wait(initial.job_id, caller=caller)
        assert final.state == "failed"
        assert final.error == "resource_limit"
        assert final.handle is None


@pytest.mark.parametrize(
    "changes",
    [
        {"execution": {"parallel_pool_size": 2}},
        {"execution": {"seed": None}},
        {"execution": {"independent_executions": 0}},
        {"hooks": {"plugins": ["arbitrary.module"]}},
        {"telemetry": {"enabled": True}},
        {"checkpoint": {"enabled": True}},
        {"experiment": {"policies": ["Unapproved"]}},
        {"experiment": {"rewards": ["Unapproved"]}},
        {"results": {"sink": "clickhouse"}},
        {"results": {"out_dir": "../outside"}},
        {"experiment": {"datasets_dir": "/tmp"}},
    ],
)
def test_unsupported_native_registrations_fail_before_execution(changes):
    native = default_registry()[2].problem
    assert isinstance(native, NativeConfiguration)
    spec = json.loads(native.spec_json)
    for section, values in changes.items():
        spec[section].update(values)
    invalid = {**native.model_dump(), "spec_json": json.dumps(spec)}
    with pytest.raises(ValueError):
        NativeConfiguration.model_validate(invalid)


@pytest.mark.parametrize(
    "mode,values",
    [("fixed_time", [1e308]), ("subset_size", [1e308]), ("ratio", [0.333, 0.334]), ("fixed_time", [3, 3])],
)
def test_native_budget_directory_limits_are_checked_before_admission(mode, values):
    native = default_registry()[2].problem
    assert isinstance(native, NativeConfiguration)
    spec = json.loads(native.spec_json)
    spec["experiment"]["budget"] = {"mode": mode, "values": values}
    invalid = {**native.model_dump(), "spec_json": json.dumps(spec), "work_units": 2 * len(values)}
    with pytest.raises(ValueError, match="budget directory"):
        NativeConfiguration.model_validate(invalid)


@pytest.mark.parametrize("mode,values", [("fixed_time", [1e230]), ("subset_size", [1e230]), ("ratio", [0.25, 0.75])])
def test_native_budget_directory_segments_remain_compatible(mode, values, tmp_path):
    from coleman.budget import BudgetMode
    from coleman.runner import _budget_directory_segment

    native = default_registry()[2].problem
    assert isinstance(native, NativeConfiguration)
    spec = json.loads(native.spec_json)
    spec["experiment"]["budget"] = {"mode": mode, "values": values}
    valid = {**native.model_dump(), "spec_json": json.dumps(spec), "work_units": 2 * len(values)}
    validated = NativeConfiguration.model_validate(valid)
    assert validated.work_units == 2 * len(values)
    for value in values:
        directory = tmp_path / _budget_directory_segment(BudgetMode(mode), value)
        directory.mkdir()
        assert directory.is_dir()


def test_units_are_not_silently_converted():
    config = default_registry()[0]
    payload = config.model_dump(mode="json")
    payload["problem"]["constraints"]["budget"]["semantics"]["unit"] = "ms"
    with pytest.raises(ValueError):
        Configuration.model_validate(payload)


def test_result_source_binding_checks_mathematics_not_only_claimed_digest():
    source = default_registry()[0]
    different = source.resolve(Parameters(budget=Decimal(2)), Limits())
    bundle = execute(WorkItem(configuration=different, limits=Limits()))
    forged = ResultBundle.model_validate({**bundle.model_dump(), "configuration_fingerprint": source.fingerprint})
    assert not source.matches_result(forged)
    assert different.matches_result(bundle)


def test_aggregate_worker_memory_slots_delay_a_second_admitted_job(tmp_path):
    caller = Caller(subject="local")
    policy = ServerPolicy(max_running=2, max_active_memory_bytes=536870912)
    with ApplicationService(default_registry(), root=tmp_path, policy=policy) as service:
        with service._condition:
            first = service.submit(
                Submit(schema_version=1, request_id="first", spec_id="native-ci-smoke-v1"), caller=caller
            )
            second = service.submit(
                Submit(schema_version=1, request_id="second", spec_id="native-ci-smoke-v1"), caller=caller
            )
            service._advance()
            assert service.status(first.job_id, caller=caller).state == "running"
            assert service.status(second.job_id, caller=caller).state == "queued"
            service.cancel(first.job_id, caller=caller)
        assert service.wait(second.job_id, caller=caller).state == "completed"


def test_memory_limited_execution_does_not_expand_the_pending_queue(tmp_path):
    caller = Caller(subject="local")
    policy = ServerPolicy(max_running=4, max_queued=1, max_active_memory_bytes=536870912)
    with ApplicationService(default_registry(), root=tmp_path, policy=policy) as service, service._condition:
        first = service.submit(Submit(schema_version=1, request_id="one", spec_id="native-ci-smoke-v1"), caller=caller)
        second = service.submit(Submit(schema_version=1, request_id="two", spec_id="native-ci-smoke-v1"), caller=caller)
        third = Submit(schema_version=1, request_id="three", spec_id="native-ci-smoke-v1")
        with pytest.raises(ApplicationError) as error:
            service.submit(third, caller=caller)
        assert error.value.code == "resource_limit"
        assert service.status(first.job_id, caller=caller).state == "running"
        assert service.status(second.job_id, caller=caller).state == "queued"


def test_zero_pending_capacity_allows_only_immediate_starts_and_evidence_imports(tmp_path):
    caller = Caller(subject="local")
    policy = ServerPolicy(max_running=4, max_queued=0, max_active_memory_bytes=536870912)
    evidence = execute(WorkItem(configuration=default_registry()[0], limits=Limits())).to_json()
    with ApplicationService(default_registry(), root=tmp_path, policy=policy) as service, service._condition:
        active = service.submit(
            Submit(schema_version=1, request_id="active", spec_id="native-ci-smoke-v1"), caller=caller
        )
        assert active.state == "running"
        pending = Submit(schema_version=1, request_id="pending", spec_id="native-ci-smoke-v1")
        with pytest.raises(ApplicationError) as error:
            service.submit(pending, caller=caller)
        assert error.value.code == "resource_limit"
        imported = service.import_result(evidence, caller=caller)
        assert imported.state == "completed"
        service.cancel(active.job_id, caller=caller)


@pytest.mark.parametrize("field,value", [("work_units", 1), ("work_units", 3), ("max_candidates", 1)])
def test_native_resource_claims_replay_actual_input_dimensions(field, value):
    native = default_registry()[2].problem
    assert isinstance(native, NativeConfiguration)
    invalid = {**native.model_dump(), field: value}
    with pytest.raises(ValueError, match="resource declarations"):
        NativeConfiguration.model_validate(invalid)


@pytest.mark.parametrize(
    "change",
    [
        "null_duration",
        "nonfinite",
        "negative",
        "duplicate",
        "missing",
        "float_build",
        "overflow32",
        "underflow32",
        "build_overflow64",
        "prio_float",
        "prio_string",
        "prio_overflow",
        "prio_underflow",
        "last_run_nested",
    ],
)
def test_native_invalid_rows_refused_during_registration(change):
    import polars as pl

    native = default_registry()[2].problem
    assert isinstance(native, NativeConfiguration)
    table = pl.read_parquet(base64.b64decode(native.inputs[0].content_base64))
    if change == "null_duration":
        table = table.with_columns(pl.lit(None, dtype=pl.Float64).alias("Duration"))
    elif change in {"nonfinite", "negative"}:
        table = table.with_columns(pl.lit(float("inf") if change == "nonfinite" else -1.0).alias("Duration"))
    elif change == "duplicate":
        table = pl.concat([table, table.head(1)])
    elif change == "missing":
        table = table.drop("CalcPrio")
    elif change in {"overflow32", "underflow32"}:
        table = table.with_columns(pl.lit(1e100 if change == "overflow32" else 1e-100).alias("Duration"))
    elif change == "build_overflow64":
        table = table.with_columns(pl.Series("BuildId", [2**63] * 3 + [2**63 + 1] * 3, dtype=pl.UInt64))
    elif change in {"prio_float", "prio_string", "prio_overflow", "prio_underflow"}:
        priority = {"prio_float": 0.0, "prio_string": "0", "prio_overflow": 2**31, "prio_underflow": -(2**31) - 1}[
            change
        ]
        table = table.with_columns(pl.lit(priority).alias("CalcPrio"))
    elif change == "last_run_nested":
        table = table.with_columns(pl.Series("LastRun", [["date"]] * table.height, dtype=pl.List(pl.String)))
    else:
        table = table.with_columns(pl.col("BuildId").cast(pl.Float64))
    stream = io.BytesIO()
    table.write_parquet(stream)
    content = stream.getvalue()
    item = PortableFile(
        name=native.inputs[0].name,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        content_base64=base64.b64encode(content).decode(),
    )
    invalid = {**native.model_dump(), "inputs": (item,)}
    with pytest.raises(ValueError):
        NativeConfiguration.model_validate(invalid)


@pytest.mark.parametrize("value", [0.0, float.fromhex("0x1p-149"), float.fromhex("0x1.fffffep+127")])
def test_zero_and_representable_native_duration_endpoints_remain_valid(value):
    import polars as pl

    from coleman.bandit.base import BANDIT_SCHEMA

    native = default_registry()[2].problem
    assert isinstance(native, NativeConfiguration)
    table = pl.read_parquet(base64.b64decode(native.inputs[0].content_base64))
    table = table.with_columns(pl.Series("Duration", [value, 2.0, 1.0] * 2, dtype=pl.Float64))
    stream = io.BytesIO()
    table.write_parquet(stream)
    content = stream.getvalue()
    item = PortableFile(
        name=native.inputs[0].name,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        content_base64=base64.b64encode(content).decode(),
    )
    validated = NativeConfiguration.model_validate({**native.model_dump(), "inputs": (item,)})
    assert validated.work_units == 2
    actual = table["Duration"].cast(pl.Float64).cast(BANDIT_SCHEMA["Duration"])
    assert actual.is_finite().all()
    assert actual[0] == value


@pytest.mark.parametrize("value", [-(2**63), 0, 2**63 - 1])
def test_native_build_id_int64_boundaries_remain_valid(value):
    import polars as pl

    native = default_registry()[2].problem
    assert isinstance(native, NativeConfiguration)
    table = pl.read_parquet(base64.b64decode(native.inputs[0].content_base64))
    other = value + 1 if value < 2**63 - 1 else value - 1
    table = table.with_columns(pl.Series("BuildId", [value] * 3 + [other] * 3, dtype=pl.Int64))
    stream = io.BytesIO()
    table.write_parquet(stream)
    content = stream.getvalue()
    item = PortableFile(
        name=native.inputs[0].name,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        content_base64=base64.b64encode(content).decode(),
    )
    validated = NativeConfiguration.model_validate({**native.model_dump(), "inputs": (item,)})
    assert validated.work_units == 2


@pytest.mark.parametrize(
    "case", ["priority_min", "priority_max", "decimal_duration", "last_run_integer", "last_run_date"]
)
def test_supported_native_scalar_conversions_remain_compatible(case):
    from datetime import date

    import polars as pl

    from coleman.bandit.base import BANDIT_SCHEMA

    native = default_registry()[2].problem
    assert isinstance(native, NativeConfiguration)
    table = pl.read_parquet(base64.b64decode(native.inputs[0].content_base64))
    if case in {"priority_min", "priority_max"}:
        priority = -(2**31) if case == "priority_min" else 2**31 - 1
        table = table.with_columns(pl.lit(priority, dtype=pl.Int64).alias("CalcPrio"))
    elif case == "decimal_duration":
        table = table.with_columns(pl.Series("Duration", [Decimal("2.0")] * table.height, dtype=pl.Decimal(10, 1)))
    elif case == "last_run_integer":
        table = table.with_columns(pl.lit(0).alias("LastRun"))
    else:
        table = table.with_columns(pl.lit(date(2026, 1, 1)).alias("LastRun"))
    stream = io.BytesIO()
    table.write_parquet(stream)
    content = stream.getvalue()
    item = PortableFile(
        name=native.inputs[0].name,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        content_base64=base64.b64encode(content).decode(),
    )
    validated = NativeConfiguration.model_validate({**native.model_dump(), "inputs": (item,)})
    assert validated.work_units == 2
    assert table["CalcPrio"].cast(BANDIT_SCHEMA["CalcPrio"]).null_count() == 0
    assert table["LastRun"].cast(BANDIT_SCHEMA["LastRun"]).null_count() == 0


def test_artifact_io_failure_is_safe_and_does_not_consume_admission(tmp_path, monkeypatch):
    from coleman.artifacts import ArtifactWriter

    caller = Caller(subject="local")
    with ApplicationService(default_registry(), root=tmp_path) as service:
        with monkeypatch.context() as failed_io:

            def interrupted(*args):
                raise OSError("private local storage diagnostic")

            failed_io.setattr(ArtifactWriter, "write_immutable", interrupted)
            command = Submit(schema_version=1, request_id="io", spec_id="weighted-overlap-3")
            with pytest.raises(ApplicationError) as error:
                service.submit(command, caller=caller)
            assert error.value.code == "unavailable_result"
            assert "private" not in str(error.value)
        job = service.submit(Submit(schema_version=1, request_id="io", spec_id="weighted-overlap-3"), caller=caller)
        assert service.wait(job.job_id, caller=caller).state == "completed"


@pytest.mark.parametrize("exit_code,partial", [(1, "partial"), (0, '{"schema_version":1')])
def test_interrupted_or_invalid_worker_write_cannot_complete_a_job(tmp_path, exit_code, partial):
    caller = Caller(subject="local")
    with ApplicationService(default_registry(), root=tmp_path) as service:
        with service._condition:
            view = service.submit(
                Submit(schema_version=1, request_id="fault", spec_id="weighted-overlap-3"), caller=caller
            )
            job = service._jobs[view.job_id]
            # Fault injection uses a real child and filesystem; the normal
            # integration tests use unmodified native solvers and runner.
            service._reap_worker(job)
            script = (
                "from pathlib import Path; import os; "
                f"Path('result.json').write_text({partial!r}); os._exit({exit_code})"
            )
            job.process = subprocess.Popen([sys.executable, "-c", script], cwd=job.directory)
            job.view = type(view).model_validate({**view.model_dump(), "state": "running"})
        final = service.wait(view.job_id, caller=caller)
        assert final.state == "failed"
        assert final.handle is None
        assert not list(tmp_path.rglob("result.json"))


def test_shutdown_while_queued_is_idempotent_and_reaps_session(tmp_path):
    service = ApplicationService(default_registry(), root=tmp_path)
    with service._condition:
        service.submit(
            Submit(schema_version=1, request_id="close", spec_id="weighted-overlap-3"), caller=Caller(subject="x")
        )
        # Release before joining the supervisor during close.
    service.close()
    service.close()
    assert not list(tmp_path.iterdir())


def test_completion_deadline_rechecked_after_work_file_cleanup(tmp_path):
    caller = Caller(subject="local")
    with ApplicationService(default_registry(), root=tmp_path) as service, service._condition:
        view = service.submit(
            Submit(schema_version=1, request_id="deadline", spec_id="weighted-overlap-3"), caller=caller
        )
        job = service._jobs[view.job_id]
        job.admitted -= view.limits.deadline_seconds + 1
        receipt = Handle(schema_version=1, job_id=view.job_id, digest="sha256:" + "0" * 64)
        service._finish(job, "completed", handle=receipt)
        assert job.view.state == "expired"
        assert job.view.handle is None
        assert not job.directory.exists()


def test_resource_checks_before_work_and_atomic_last_publication(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Control only the test process's sticky historical RSS; subprocess integration
    # tests above exercise the real resource sampler and native runner unchanged.
    monkeypatch.setattr("coleman.telemetry.resources._get_peak_rss_mib", lambda: 1.0)
    item = WorkItem(configuration=default_registry()[0], limits=Limits())
    (tmp_path / "request.json").write_text(item.model_dump_json())
    main()
    result = ResultBundle.from_json((tmp_path / "result.json").read_bytes()).result
    assert isinstance(result, CoverageResult)
    assert result.selected_ids == ("A", "C")
    _check_resources(Limits())
    stopped = threading.Event()
    stopped.set()
    _guard(Limits(), stopped)


def test_resource_guard_unknown_rss_and_real_output_boundaries(tmp_path, monkeypatch):
    import coleman.application.worker as worker

    monkeypatch.chdir(tmp_path)

    def stopped(code):
        raise SystemExit(code)

    limits = Limits()
    small_output = Limits(max_output_bytes=1024)
    monkeypatch.setattr(worker.os, "_exit", stopped)
    monkeypatch.setattr("coleman.telemetry.resources._get_peak_rss_mib", lambda: None)
    with pytest.raises(SystemExit, match="74"):
        _check_resources(limits)
    monkeypatch.setattr("coleman.telemetry.resources._get_peak_rss_mib", lambda: 1.0)
    (tmp_path / "run").mkdir()
    (tmp_path / "run/data").write_bytes(b"x" * 1025)
    with pytest.raises(SystemExit, match="74"):
        _check_resources(small_output)
    (tmp_path / "run/data").unlink()
    (tmp_path / "run/link").symlink_to(tmp_path / "request.json")
    with pytest.raises(SystemExit, match="74"):
        _check_resources(limits)


def test_application_domain_imports_do_not_point_to_transports_or_cli():
    root = Path(__file__).parents[2] / "coleman"
    for package in ["application", "selection", "allocation", "results"]:
        for path in (root / package).rglob("*.py"):
            tree = ast.parse(path.read_text())
            names = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    names.append(node.module or "")
            assert not any(
                name == "mcp" or name.startswith(("mcp.", "coleman.cli", "coleman.transports")) for name in names
            ), path


def test_raw_legacy_artifact_is_not_silently_reinterpreted(tmp_path):
    legacy = Path(__file__).parents[1] / "fixtures/domain-results/native-v3.2.json"
    original = legacy.read_bytes()
    payload = original.decode()
    caller = Caller(subject="local")
    with ApplicationService(default_registry(), root=tmp_path, policy=ServerPolicy()) as service:
        with pytest.raises(ApplicationError) as error:
            service.import_result(payload, caller=caller)
        assert error.value.code == "invalid_input"
    assert legacy.read_bytes() == original


@pytest.mark.parametrize("operation", ["rmtree", "unlink"])
def test_cleanup_oserror_keeps_real_supervisor_alive_and_other_jobs_progressing(tmp_path, monkeypatch, operation):
    import shutil
    import time

    caller = Caller(subject="cleanup")
    failed = threading.Event()
    real_rmtree, real_unlink = shutil.rmtree, Path.unlink

    def rmtree(path, *args, **kwargs):
        if operation == "rmtree" and Path(path).name == "run" and not failed.is_set():
            failed.set()
            raise OSError("injected cleanup error")
        return real_rmtree(path, *args, **kwargs)

    def unlink(path, *args, **kwargs):
        if operation == "unlink" and path.name == "request.json" and not failed.is_set():
            failed.set()
            raise OSError("injected cleanup error")
        return real_unlink(path, *args, **kwargs)

    with (
        ApplicationService(default_registry(), root=tmp_path, policy=ServerPolicy(max_running=1)) as service,
        monkeypatch.context() as faults,
    ):
        faults.setattr(shutil, "rmtree", rmtree)
        faults.setattr(Path, "unlink", unlink)
        first = service.submit(
            Submit(schema_version=1, request_id="cleanup-failure", spec_id="native-ci-smoke-v1"), caller=caller
        )
        second = service.submit(
            Submit(schema_version=1, request_id="after-cleanup", spec_id="weighted-overlap-3"), caller=caller
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            one, two = service.status(first.job_id, caller=caller), service.status(second.job_id, caller=caller)
            if one.state == "failed" and two.state == "completed":
                break
            time.sleep(0.02)
        assert failed.is_set()
        assert one.state == "failed"
        assert one.error == "unavailable_result"
        assert one.handle is None
        assert service._thread.is_alive()
        assert service._jobs[first.job_id].process is None
        assert two.state == "completed"
        assert service.wait(first.job_id, caller=caller) == one
        assert service.wait(second.job_id, caller=caller) == two
