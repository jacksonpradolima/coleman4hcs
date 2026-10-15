"""Trusted configuration registry; requests can select IDs but cannot supply executable configuration."""

from __future__ import annotations

import base64
import hashlib
import math
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.allocation import AllocationProblem
from coleman.application.contracts import (
    ApplicationContract,
    ApplicationError,
    ExperimentResult,
    Limits,
    Parameters,
    PortableFile,
    ResultBundle,
    SafeId,
)
from coleman.selection import CoverageProblem, Quantity
from coleman.selection.resources import SearchLimits, inspect_workload, preflight_reason
from coleman.spec.models import RunSpec

if TYPE_CHECKING:
    import polars as pl


class NativeConfiguration(ApplicationContract):
    """Snapshot of one approved native TCPCI RunSpec and bounded local Parquet inputs.

    Host setup reads files once. Execution uses the snapshot, never the original
    path. V1 supports sequential Random/RNFail or TimeRank experiments with local
    Parquet, no hooks, extensions, remote sinks, telemetry or checkpoint resume.
    Existing unrestricted library run/checkpoint interfaces remain unchanged.
    """

    spec_json: Annotated[str, Field(max_length=65536)]
    inputs: Annotated[tuple[PortableFile, ...], Field(min_length=1, max_length=8)]
    work_units: Annotated[int, Field(strict=True, ge=1, le=100000)]
    max_candidates: Annotated[int, Field(strict=True, ge=1, le=1000)]

    @model_validator(mode="after")
    def _approved(self) -> Self:
        """Reject executable/network configuration and non-normalized output locations."""
        spec = RunSpec.model_validate_json(self.spec_json)
        _execution_contract(spec)
        _experiment_contract(spec)
        _storage_contract(spec)
        names = tuple(item.name for item in self.inputs)
        expected = tuple(sorted(f"{name}/features-engineered.parquet" for name in spec.experiment.datasets))
        if names != expected or len(set(names)) != len(names) or sum(item.size_bytes for item in self.inputs) > 1048576:
            raise ValueError("Native inputs do not match approved datasets")
        for name in spec.experiment.datasets:
            from pydantic import TypeAdapter

            TypeAdapter(SafeId).validate_python(name)
        facts = tuple(_inspect_native_input(item) for item in self.inputs)
        work = sum(builds for builds, _ in facts) * spec.execution.independent_executions
        work *= len(spec.experiment.rewards) * len(spec.experiment.budget.values)
        if self.work_units != work or self.max_candidates != max(candidates for _, candidates in facts):
            raise ValueError("Native resource declarations do not match snapshotted input")
        return self


class Configuration(ApplicationContract):
    """Immutable registered native problem or RunSpec snapshot, capped before admission."""

    spec_id: SafeId
    kind: Literal["coverage", "allocation", "experiment"]
    problem: CoverageProblem | AllocationProblem | NativeConfiguration

    @model_validator(mode="after")
    def _kind(self) -> Self:
        """Bind configuration kind without accepting discriminator ambiguities."""
        types = {"coverage": CoverageProblem, "allocation": AllocationProblem, "experiment": NativeConfiguration}
        if not isinstance(self.problem, types[self.kind]):
            raise ValueError("Configuration kind mismatch")
        if len(self.to_payload_bytes()) > 1572864:
            raise ValueError("Configuration snapshot exceeds 1.5 MiB")
        return self

    def to_payload_bytes(self) -> bytes:
        """Measure nested JSON without recursively invoking serialization validation."""
        return self.model_dump_json().encode("utf-8")

    def matches_result(self, bundle: ResultBundle) -> bool:
        """Bind actual native source/work identity, not merely a claimed configuration digest."""
        from coleman.allocation import AllocationResult
        from coleman.selection import CoverageResult
        from coleman.spec.run_id import compute_run_id

        if (bundle.spec_id, bundle.kind, bundle.configuration_fingerprint) != (
            self.spec_id,
            self.kind,
            self.fingerprint,
        ):
            return False
        if isinstance(bundle.result, (CoverageResult, AllocationResult)):
            return bundle.result.problem == self.problem
        return (
            isinstance(self.problem, NativeConfiguration)
            and isinstance(bundle.result, ExperimentResult)
            and bundle.result.work_units == self.problem.work_units
            and bundle.result.run_id == compute_run_id(RunSpec.model_validate_json(self.problem.spec_json))
        )

    def resolve(self, parameters: Parameters, limits: Limits) -> Configuration:
        """Apply only documented typed overrides, then reject oversized work before queue admission."""
        problem = self.problem
        if parameters.budget is not None:
            if not isinstance(problem, CoverageProblem):
                raise ApplicationError("invalid_input", "Budget override is available only for coverage")
            constraints = problem.constraints.model_dump()
            constraints["budget"] = Quantity(value=parameters.budget, semantics=problem.constraints.budget.semantics)
            problem = CoverageProblem.model_validate({**problem.model_dump(), "constraints": constraints})
        if isinstance(problem, CoverageProblem):
            native = SearchLimits(
                max_work=limits.max_work,
                deadline_seconds=None,
                max_working_bytes=min(1073741824, limits.max_memory_bytes),
                max_candidates=limits.max_candidates,
            )
            if preflight_reason(
                problem, "branch_bound_v1", native, inspect_workload(problem, "branch_bound_v1", native)
            ):
                raise ApplicationError("resource_limit", "Coverage exceeds admitted native dimensions or memory")
        elif isinstance(problem, AllocationProblem):
            from coleman.allocation.resources import inspect_allocation

            if (
                len(problem.actions) > limits.max_candidates
                or inspect_allocation(problem).estimated_working_bytes > limits.max_memory_bytes
            ):
                raise ApplicationError("resource_limit", "Allocation exceeds admitted dimensions or memory")
        elif problem.work_units > limits.max_work or problem.max_candidates > limits.max_candidates:
            raise ApplicationError("resource_limit", "Experiment exceeds admitted build/agent work or candidates")
        return Configuration(schema_version=1, spec_id=self.spec_id, kind=self.kind, problem=problem)


def register_experiment(spec_id: str, spec: RunSpec) -> Configuration:
    """Snapshot trusted local inputs and normalize execution paths once during host setup.

    Reads at most eight Parquet inputs totaling 1 MiB and at most 10,000 rows each.
    Work is builds × executions × rewards × budget values. Data fingerprints bind
    complete file bytes; seeds reproduce scheduling, not independent observations.
    Invalid/unsupported native settings raise ValueError before any job executes.
    """
    from pydantic import TypeAdapter

    spec = RunSpec.model_validate(spec.model_dump())
    if not 1 <= len(spec.experiment.datasets) <= 8:
        raise ValueError("Expected one to eight approved datasets")
    inputs: list[PortableFile] = []
    builds = candidates = size = 0
    for name in sorted(spec.experiment.datasets):
        TypeAdapter(SafeId).validate_python(name)
        path = Path(spec.experiment.datasets_dir) / name / "features-engineered.parquet"
        if path.stat().st_size + size > 1048576:
            raise ValueError("Native input byte limit exceeded")
        content = path.read_bytes()
        size += len(content)
        inputs.append(
            PortableFile(
                name=f"{name}/features-engineered.parquet",
                sha256=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
                content_base64=base64.b64encode(content).decode("ascii"),
            )
        )
        actual_builds, actual_candidates = _inspect_native_input(inputs[-1])
        builds += actual_builds
        candidates = max(candidates, actual_candidates)
    spec.experiment.datasets_dir = "inputs"
    spec.experiment.experiment_dir = "run/experiments"
    spec.results.out_dir = "run"
    spec.checkpoint.base_dir = "run/checkpoints"
    native = NativeConfiguration(
        schema_version=1,
        spec_json=spec.model_dump_json(),
        inputs=tuple(inputs),
        work_units=builds
        * spec.execution.independent_executions
        * len(spec.experiment.rewards)
        * len(spec.experiment.budget.values),
        max_candidates=candidates,
    )
    return Configuration(schema_version=1, spec_id=spec_id, kind="experiment", problem=native)


def default_registry() -> tuple[Configuration, ...]:
    """Load packaged, public synthetic CI fixtures; no paths, network or plugin discovery."""
    from importlib.resources import files

    from coleman.benchmarks import load_suite

    suite = load_suite()
    decisions = tuple(
        Configuration(schema_version=1, spec_id=case.case_id, kind=case.kind, problem=case.problem)
        for case in suite.cases[:2]
    )
    native = Configuration.from_json(files("coleman.application").joinpath("cases", "native-ci-v1.json").read_bytes())
    return (*decisions, native)


def _execution_contract(spec: RunSpec) -> None:
    """Keep approved runs sequential, reproducible and free from extension code or exporters."""
    if spec.execution.parallel_pool_size != 1 or not 1 <= spec.execution.independent_executions <= 16:
        raise ValueError("Native execution must be sequential with one to sixteen repetitions")
    seed = spec.execution.seed
    if seed is None or spec.hooks.plugins or spec.extensions:
        raise ValueError("Native runs require an explicit seed and no client extensions")
    if seed < 0 or seed + spec.execution.independent_executions - 1 > 2**64 - 1:
        raise ValueError("Native seed and all repetition offsets must fit unsigned 64-bit RNG inputs")
    if spec.telemetry.enabled or spec.checkpoint.enabled or spec.hcs_configuration.wts_strategy:
        raise ValueError("Bounded v1 does not enable exporters, checkpoint resume or HCS")
    if spec.algorithm.model_dump():
        raise ValueError("Bounded Random baseline does not accept algorithm overrides")


def _experiment_contract(spec: RunSpec) -> None:
    """Restrict v1 native algorithms and multiplicative experiment dimensions explicitly."""
    if spec.experiment.policies != ["Random"]:
        raise ValueError("Bounded v1 native runs use the Random baseline")
    if not 1 <= len(spec.experiment.rewards) <= 2 or not set(spec.experiment.rewards).issubset({"RNFail", "TimeRank"}):
        raise ValueError("Bounded v1 permits one or two approved native rewards")
    if not 1 <= len(spec.experiment.budget.values) <= 8:
        raise ValueError("Bounded v1 permits one to eight native budget values")
    if len(set(spec.experiment.rewards)) != len(spec.experiment.rewards):
        raise ValueError("Native reward declarations must be unique")
    if not all(math.isfinite(value) for value in spec.experiment.budget.values):
        raise ValueError("Native budgets must be finite")
    _budget_directory_contract(spec)


def _budget_directory_contract(spec: RunSpec) -> None:
    """Require unique native budget paths fitting Linux/macOS 255-byte components.

    Reuse the runner's naming function, including its rounded ratio labels. Reject
    oversized or colliding names before admission; do not rename old artifacts or
    change unrestricted native runner behavior. This check performs no writes.
    """
    from coleman.runner import _budget_directory_segment

    budget = spec.experiment.budget
    segments = tuple(_budget_directory_segment(budget.mode, value) for value in budget.values)
    if any(len(segment.encode("utf-8")) > 255 for segment in segments):
        raise ValueError("Native budget directory segment exceeds 255 bytes")
    if len(set(segments)) != len(segments):
        raise ValueError("Native budget directory segments must be unique")


def _storage_contract(spec: RunSpec) -> None:
    """Keep native side effects within the owned worker directory and local Parquet sink."""
    if spec.results.sink != "parquet" or not spec.results.enabled or spec.results.clickhouse or spec.results.duckdb:
        raise ValueError("Bounded v1 requires local Parquet results")
    if spec.experiment.datasets_dir != "inputs" or spec.experiment.experiment_dir != "run/experiments":
        raise ValueError("Native input and experiment directories must be normalized")
    if spec.results.out_dir != "run" or spec.checkpoint.base_dir != "run/checkpoints":
        raise ValueError("Native result and checkpoint directories must be normalized")


def _inspect_native_input(item: PortableFile) -> tuple[int, int]:
    """Validate bounded trusted Parquet rows and recompute actual build/candidate counts.

    Host setup supplies trusted Parquet, not an adversarial upload. Null/non-finite
    durations are rejected here instead of invoking legacy null-to-zero coercion.
    Existing direct native loaders retain their original compatibility behavior.
    """
    import polars as pl

    try:
        table = pl.read_parquet(base64.b64decode(item.content_base64), n_rows=10001)
    except pl.exceptions.PolarsError as exc:
        raise ValueError("Invalid approved native Parquet input") from exc
    if not 1 <= table.height <= 10000:
        raise ValueError("Expected one to 10,000 native rows")
    required = {"BuildId", "Name", "Duration", "Verdict", "CalcPrio", "LastRun"}
    if not required.issubset(table.columns):
        raise ValueError("Missing native testcase columns")
    if not table.schema["BuildId"].is_integer() or not table.schema["Verdict"].is_integer():
        raise ValueError("Native BuildId and Verdict require integer columns")
    if not table.schema["Duration"].is_numeric() or table.schema["Name"] != pl.String:
        raise ValueError("Native durations require numeric seconds and testcase names require strings")
    if any(table[column].null_count() for column in required):
        raise ValueError("Null native input values are not admitted")
    if table["BuildId"].cast(pl.Int64, strict=False).null_count():
        raise ValueError("Native BuildId values must fit the loader's signed Int64 range")
    _validate_native_durations(table["Duration"])
    _validate_native_auxiliary_columns(table)
    if (
        not table["Verdict"].is_in([0, 1]).all()
        or table.select(pl.struct("BuildId", "Name").is_duplicated().any()).item()
    ):
        raise ValueError("Native verdicts must be binary and testcase names unique within each build")
    return table["BuildId"].n_unique(), max(table.group_by("BuildId").len()["len"].to_list())


def _validate_native_durations(durations: pl.Series) -> None:
    """Reject overflow and positive-to-zero underflow in the existing native dtype path.

    The native loader converts to Float64, then the bandit uses BANDIT_SCHEMA's
    duration dtype (currently Float32). Ordinary native rounding is preserved;
    genuine zero remains valid. No input bytes or native schemas are modified.
    """
    import polars as pl

    from coleman.bandit.base import BANDIT_SCHEMA

    loaded = durations.cast(pl.Float64)
    if not loaded.is_finite().all() or (loaded < 0).any():
        raise ValueError("Native durations require finite nonnegative seconds")
    converted = loaded.cast(BANDIT_SCHEMA["Duration"])
    if not converted.is_finite().all() or ((loaded > 0) & (converted == 0)).any():
        raise ValueError("Native duration dtype would overflow or erase a positive duration")


def _validate_native_auxiliary_columns(table: pl.DataFrame) -> None:
    """Check the remaining consumed native fields before any experiment is admitted.

    Priorities require integer values in the bandit's signed Int32 range. LastRun
    follows the existing loader's string conversion; unsupported/null conversions
    are refused rather than silently replaced. Optional unused columns are not
    reinterpreted. Native schemas and source bytes remain unchanged.
    """
    import polars as pl

    from coleman.bandit.base import BANDIT_SCHEMA

    if not table.schema["CalcPrio"].is_integer():
        raise ValueError("Native CalcPrio requires an integer column")
    try:
        priorities = table["CalcPrio"].cast(BANDIT_SCHEMA["CalcPrio"], strict=False)
        last_runs = table["LastRun"].cast(BANDIT_SCHEMA["LastRun"], strict=False)
    except pl.exceptions.PolarsError as exc:
        raise ValueError("Unsupported native priority or LastRun conversion") from exc
    if priorities.null_count() or last_runs.null_count():
        raise ValueError("Native CalcPrio/LastRun values cannot survive their native dtype conversion")
