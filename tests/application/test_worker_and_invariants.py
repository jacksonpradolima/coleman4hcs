"""Native adapter execution, fault injection, metamorphic properties and import boundaries."""

import ast
import base64
import hashlib
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
    Limits,
    Parameters,
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
        assert result.cost is not None and result.objective is not None
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
    with pytest.raises(ValueError):
        NativeConfiguration.model_validate({**native.model_dump(), "spec_json": json.dumps(spec)})


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


@pytest.mark.parametrize("field,value", [("work_units", 1), ("work_units", 3), ("max_candidates", 1)])
def test_native_resource_claims_replay_actual_input_dimensions(field, value):
    native = default_registry()[2].problem
    assert isinstance(native, NativeConfiguration)
    with pytest.raises(ValueError, match="resource declarations"):
        NativeConfiguration.model_validate({**native.model_dump(), field: value})


@pytest.mark.parametrize("change", ["null_duration", "nonfinite", "negative", "duplicate", "missing", "float_build"])
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
    else:
        table = table.with_columns(pl.col("BuildId").cast(pl.Float64))
    import io

    stream = io.BytesIO()
    table.write_parquet(stream)
    content = stream.getvalue()
    item = PortableFile(
        name=native.inputs[0].name,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        content_base64=base64.b64encode(content).decode(),
    )
    with pytest.raises(ValueError):
        NativeConfiguration.model_validate({**native.model_dump(), "inputs": (item,)})


def test_artifact_io_failure_is_safe_and_does_not_consume_admission(tmp_path, monkeypatch):
    from coleman.artifacts import ArtifactWriter

    caller = Caller(subject="local")
    with ApplicationService(default_registry(), root=tmp_path) as service:
        with monkeypatch.context() as failed_io:

            def interrupted(*args):
                raise OSError("private local storage diagnostic")

            failed_io.setattr(ArtifactWriter, "write_immutable", interrupted)
            with pytest.raises(ApplicationError) as error:
                service.submit(Submit(schema_version=1, request_id="io", spec_id="weighted-overlap-3"), caller=caller)
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

    monkeypatch.setattr(worker.os, "_exit", stopped)
    monkeypatch.setattr("coleman.telemetry.resources._get_peak_rss_mib", lambda: None)
    with pytest.raises(SystemExit, match="74"):
        _check_resources(Limits())
    monkeypatch.setattr("coleman.telemetry.resources._get_peak_rss_mib", lambda: 1.0)
    (tmp_path / "run").mkdir()
    (tmp_path / "run/data").write_bytes(b"x" * 1025)
    with pytest.raises(SystemExit, match="74"):
        _check_resources(Limits(max_output_bytes=1024))
    (tmp_path / "run/data").unlink()
    (tmp_path / "run/link").symlink_to(tmp_path / "request.json")
    with pytest.raises(SystemExit, match="74"):
        _check_resources(Limits())


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
    with ApplicationService(default_registry(), root=tmp_path, policy=ServerPolicy()) as service:
        with pytest.raises(ApplicationError) as error:
            service.import_result(original.decode(), caller=Caller(subject="local"))
        assert error.value.code == "invalid_input"
    assert legacy.read_bytes() == original
