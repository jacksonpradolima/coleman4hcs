"""Pure bounded planning over existing sweep primitives; no execution or transport."""

from __future__ import annotations

import math

from coleman.results.domain import canonical_json, fingerprint
from coleman.spec import RunSpec
from coleman.spec.runtime_identity import InstalledRuntime
from coleman.spec.sweep import SweepAxis, SweepSpec, expand_sweep
from coleman.workflows.contracts import DatasetManifest, ExperimentPlan, PlanningRequest, Trial, WorkflowError


def _cardinality(request: PlanningRequest, dataset: DatasetManifest) -> None:
    """Reject oversized/invalid search work before any grid list is materialized."""
    budgets, coefficients = len(request.budget.values), len(request.ucb_exploration)
    variable = [size for size in (budgets, coefficients) if size > 1]
    if request.mode == "zip" and len(set(variable)) > 1 and "UCB" in request.policies:
        raise WorkflowError("Zip variable dimensions must have matching lengths")
    ucb = math.prod(variable) if request.mode == "grid" else max(variable, default=1)
    configurations = budgets + (ucb if "UCB" in request.policies else 0)
    trials = configurations * len(request.seeds)
    limits = request.limits
    if configurations > limits.max_configurations or trials > limits.max_trials:
        raise WorkflowError("Search space exceeds configuration/trial limits before expansion")
    if trials * len(dataset.build_ids) * limits.max_attempts > limits.max_build_work:
        raise WorkflowError("Worst-case native build work, including retries, exceeds the declared cap")
    if request.evaluation == "protected":
        if limits.max_attempts < 2:
            raise WorkflowError(
                "Protected execution requires at least two worker invocations for tuning and evaluation"
            )
        if request.budget.mode.value != "ratio" or len(request.budget.values) != 1 or "UCB" not in request.policies:
            raise WorkflowError(
                "Protected v1 requires one ratio budget and Random/UCB; other budgets support descriptive replay"
            )
        _protected_eligibility(dataset)


def _protected_eligibility(dataset: DatasetManifest) -> None:
    """Require complete declared stage/clock/cost eligibility, never fabricate source admission."""
    if dataset.temporal_plan is None or not dataset.metadata:
        raise WorkflowError("Protected execution requires host-registered temporal/group metadata")
    if dataset.evidence != "constructed_mechanism":
        raise WorkflowError("Insufficient evidence: source registration does not grant empirical protected admission")
    if not all(item.costs_known_at_decision for item in dataset.metadata):
        raise WorkflowError("Protected execution requires declared predecision scheduling costs")
    decisions = dataset.temporal_plan.split(tuple(item.observation for item in dataset.metadata))
    if not all(item.eligible for item in decisions):
        raise WorkflowError("Protected execution rejects late/unavailable outcomes and overlapping groups")
    partitions = tuple(item.partition for item in decisions)
    if set(partitions) != {"train", "validation", "test"}:
        raise WorkflowError("Protected execution requires nonempty train, validation and test stages")
    order = {"train": 0, "validation": 1, "test": 2}
    if list(partitions) != sorted(partitions, key=lambda item: order[str(item)]):
        raise WorkflowError("Native stage continuation requires chronological contiguous partitions")


def base_spec(request: PlanningRequest, policy: str) -> RunSpec:
    """Return normalized native configuration without ambient host paths or side effects.

    Parameters
    ----------
    request : PlanningRequest
        Validated native budgets, RNFail policy parameters and seed coordinates.
    policy : str
        Host-approved Random or UCB native name.

    Returns
    -------
    RunSpec
        Complete normalized specification; execution supplies owned local paths.
        Legacy native run identity and sweep semantics are unchanged.
    """
    return RunSpec.model_validate(
        {
            "execution": {"independent_executions": 1, "parallel_pool_size": 1},
            "experiment": {
                "datasets_dir": "inputs",
                "datasets": [request.dataset_id],
                "policies": [policy],
                "rewards": ["RNFail"],
                "budget": {"mode": request.budget.mode, "values": [request.budget.values[0]]},
            },
            "algorithm": {} if policy == "Random" else {"ucb": {"rnfail": {"c": request.ucb_exploration[0]}}},
            "checkpoint": {"enabled": True, "interval": 1},
            "results": {"out_dir": "native", "sink": "parquet", "top_k_prioritization": 10000},
        }
    )


def resolve_trials(request: PlanningRequest, dataset: DatasetManifest) -> tuple[Trial, ...]:
    """Count before expansion and reuse native grid/zip/seed execution coordinates.

    Parameters
    ----------
    request : PlanningRequest
        Typed dimensions; grid declaration order is immaterial, zip correspondence
        remains meaningful. Random never receives irrelevant UCB axes.
    dataset : DatasetManifest
        Actual frozen membership used to bound worst-case build work.

    Returns
    -------
    tuple[Trial, ...]
        Unique coordinates sorted by content identity, independent of execution order.

    Raises
    ------
    ValueError
        Invalid, duplicate, incompatible or oversized dimensions before expansion.
    """
    request, dataset = PlanningRequest.model_validate(request), DatasetManifest.model_validate(dataset)
    _cardinality(request, dataset)
    trials = []
    for policy in sorted(request.policies):
        params = {}
        if len(request.budget.values) > 1:
            params["experiment.budget.values"] = [[value] for value in request.budget.values]
        if policy == "UCB" and len(request.ucb_exploration) > 1:
            params["algorithm.ucb.rnfail.c"] = list(request.ucb_exploration)
        if not params:
            params = {"experiment.policies": [[policy]]}
        axis = SweepAxis(mode=request.mode, params=params)
        specs = expand_sweep(base_spec(request, policy), SweepSpec(axes=[axis], seeds=sorted(request.seeds)))
        for spec in specs:
            assert spec.execution.seed is not None
            coefficient = spec.algorithm.model_dump()["ucb"]["rnfail"]["c"] if policy == "UCB" else None
            config = {
                "policy": policy,
                "c": coefficient,
                "budget_mode": request.budget.mode.value,
                "budget_value": spec.experiment.budget.values[0],
                "reward": "RNFail",
            }
            resolved = spec.model_dump(mode="json")
            configuration = spec.model_dump(mode="json")
            configuration["execution"]["seed"] = None
            config_id = fingerprint(["coleman.workflow.configuration.v1", dataset.fingerprint, config, configuration])
            trials.append(
                Trial(
                    trial_id=fingerprint([config_id, spec.execution.seed]),
                    configuration_id=config_id,
                    policy=policy,
                    exploration=coefficient,
                    budget_value=spec.experiment.budget.values[0],
                    seed=spec.execution.seed,
                    resolved_spec_json=canonical_json(resolved),
                )
            )
    return tuple(sorted(trials, key=lambda trial: trial.trial_id))


def make_plan(
    request: PlanningRequest, dataset: DatasetManifest, runtime: InstalledRuntime, *, native_threads: int = 1
) -> ExperimentPlan:
    """Freeze resolved inputs, runtime and exact bounded trial coordinates without I/O.

    Parameters
    ----------
    request : PlanningRequest
        Client-selected approved dimensions and discovered defaults.
    dataset : DatasetManifest
        Host-registered exact input/evidence snapshot.
    runtime : InstalledRuntime
        Existing conservative producer/base-runtime identity captured by the host.
    native_threads : int
        Explicit Polars pool size for execution/continuation compatibility. Defaults
        to a one-thread hypothetical plan; the application captures actual host size.

    Returns
    -------
    ExperimentPlan
        Canonical executable plan; equivalent grid ordering retains its fingerprint.

    Raises
    ------
    ValueError
        Ineligible or oversized inputs, before any native execution is admitted.
    """
    request = PlanningRequest.model_validate(request)
    payload = request.model_dump()
    payload["policies"], payload["seeds"] = sorted(request.policies), sorted(request.seeds)
    if request.mode == "grid":
        payload["ucb_exploration"] = sorted(request.ucb_exploration)
        payload["budget"] = {**request.budget.model_dump(), "values": sorted(request.budget.values)}
    canonical = PlanningRequest.model_validate(payload)
    return ExperimentPlan(
        schema_version=1,
        request=canonical,
        dataset=dataset,
        runtime=runtime,
        trials=resolve_trials(canonical, dataset),
        native_threads=native_threads,
    )
