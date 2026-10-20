"""Pure completion/context validation shared by local and portable workflow readers."""

from __future__ import annotations

from coleman.corpus.repeated import _complete_mean
from coleman.corpus.repeated_contracts import MEAN_SUFFIX
from coleman.evaluation.protected import BoundedMeanProtocol, FamilyReport
from coleman.results.domain import decode_json, fingerprint
from coleman.workflows.contracts import ExperimentPlan, WorkflowError
from coleman.workflows.evidence import FrozenWorkflowConfiguration
from coleman.workflows.lifecycle import WorkflowReport


def protected_protocol(plan: ExperimentPlan) -> BoundedMeanProtocol:
    """Resolve the predeclared bounded protocol before protected native execution.

    Parameters
    ----------
    plan : ExperimentPlan
        Frozen objective, thresholds and stable upstream source scope. Changes
        to storage encoding/projection cannot reset this source's access ledger.

    Returns
    -------
    BoundedMeanProtocol
        Existing dimensionless [0, 1] protocol with independent-family assumptions,
        no normality declaration and the full requested uncertainty settings.
    """
    settings = plan.request.protected
    return BoundedMeanProtocol(
        schema_version=1,
        protocol_id="workflow-bounded-paired-v1",
        holdout_id=fingerprint(["workflow.holdout.v1", plan.dataset.source_fingerprint]),
        metric_name=f"workflow_{plan.request.objective}",
        unit="1",
        meaning=f"coleman.native.{plan.request.objective}.binary64.v1" + MEAN_SUFFIX,
        direction="maximize",
        independent_unit="reviewed_repository_family",
        dependence="independent",
        practical_effect=settings.practical_effect,
        minimum_units=settings.minimum_units,
        alpha=settings.alpha,
        outcome_lower=0.0,
        outcome_upper=1.0,
    )


def _configurations(plan: ExperimentPlan, comparison: FamilyReport):
    """Check source, stage, baseline/challenger and entire predeclared metric protocol."""
    frozen = comparison.frozen
    if frozen.data_fingerprint != plan.dataset.fingerprint or frozen.protocol != protected_protocol(plan):
        raise WorkflowError("Protected data/protocol differs from the saved workflow")
    configs = _candidate_contexts(plan, frozen)
    _planned_policies(plan, configs)
    source = plan.dataset.temporal_plan
    if source is None or (frozen.plan.train_end, frozen.plan.validation_end) != (
        source.train_end,
        source.validation_end,
    ):
        raise WorkflowError("Protected temporal stages differ from the original source declarations")
    return configs


def _candidate_contexts(plan, frozen) -> dict[str, FrozenWorkflowConfiguration]:
    """Decode the strict workflow context at the generic frozen-JSON boundary."""
    configs = {
        candidate.candidate_id: FrozenWorkflowConfiguration.model_validate(decode_json(candidate.configuration_json))
        for candidate in frozen.candidates
    }
    if set(configs) != {"baseline", "challenger"} or any(
        (config.workflow_plan, config.source_scope) != (plan.fingerprint, plan.dataset.source_fingerprint)
        for config in configs.values()
    ):
        raise WorkflowError("Protected candidate family differs from the saved workflow")
    for config in configs.values():
        _learning_membership(plan, config)
    return configs


def _learning_membership(plan: ExperimentPlan, configuration: FrozenWorkflowConfiguration) -> None:
    """Bind every claimed seed snapshot to its unique planned candidate trial.

    Validate references without reading or unpickling imported state. Pin digests
    establish consistency, not authenticity of an external producer or source.
    """
    expected = {
        trial.trial_id: trial.seed for trial in plan.trials if trial.configuration_id == configuration.configuration_id
    }
    actual = {state.trial_id: state.seed for state in configuration.learning_state}
    if len(actual) != len(configuration.learning_state) or actual != expected:
        raise WorkflowError("Frozen learning state must cover every planned candidate trial/seed exactly once")
    for state in configuration.learning_state:
        prefix = f"frozen-learning/{state.trial_id.removeprefix('sha256:')}/"
        names = tuple(pin.name for pin in state.checkpoints)
        if len(set(names)) != len(names) or any(not name.startswith(prefix) for name in names):
            raise WorkflowError("Frozen learning checkpoint membership differs from its planned trial")


def _planned_policies(plan, configs) -> None:
    """Require an actual planned same-budget UCB challenger and Random baseline."""
    for name, policy in (("baseline", "Random"), ("challenger", "UCB")):
        if not any(
            trial.configuration_id == configs[name].configuration_id and trial.policy == policy for trial in plan.trials
        ):
            raise WorkflowError("Protected comparison requires a planned UCB challenger and Random baseline")


def _grouped_native(plan, report, configs):
    """Group actual binary64 observations by candidate/family/build, retaining every seed."""
    assert plan.dataset.temporal_plan is not None
    decisions = plan.dataset.split_by_build()
    metadata = {
        item.build_id: item
        for item, decision in zip(plan.dataset.metadata, decisions, strict=True)
        if decision.partition == "test"
    }
    trials = {trial.trial_id: trial for trial in plan.trials}
    grouped = {}
    names = {config.configuration_id: name for name, config in configs.items()}
    for receipt in report.trials:
        trial = trials.get(receipt.trial_id)
        if trial is None or trial.configuration_id not in names:
            raise WorkflowError("Protected completion contains an unselected/unplanned coordinate")
        for row in receipt.rows:
            if row.source_build_id in metadata:
                key = names[trial.configuration_id], metadata[row.source_build_id].unit_id
                grouped.setdefault(key, {}).setdefault(row.source_build_id, []).append(
                    getattr(row, plan.request.objective)
                )
    return grouped


def expected_trials_for_report(plan: ExperimentPlan, report: WorkflowReport) -> set[str]:
    """Require full planned membership and protected means grounded in actual native rows.

    Parameters
    ----------
    plan : ExperimentPlan
        Complete immutable source, coordinates, metrics, stages and resource context.
    report : WorkflowReport
        Proposed completed local/portable report; integrity is not authentication.

    Returns
    -------
    set[str]
        Complete expected logical trial IDs, including every declared selected seed.

    Raises
    ------
    ValueError
        Wrong plan/protocol/family, invented inference units or means inconsistent
        with native observations. Existing corpus arithmetic performs aggregation;
        this function does not implement another inferential/statistical engine.
    """
    if report.plan_fingerprint != plan.fingerprint:
        raise WorkflowError("Workflow report has a different plan")
    interpretation = "descriptive_only" if plan.request.evaluation == "descriptive" else "insufficient_evidence"
    if report.interpretation != interpretation:
        raise WorkflowError("Workflow report interpretation contradicts its declared evaluation mode")
    _validate_attempts(plan, report)
    _validate_choices(plan, report)
    if plan.request.evaluation == "descriptive":
        if report.comparison is not None:
            raise WorkflowError("Descriptive replay cannot claim protected inference")
        _exact_choices(plan, report)
        return {trial.trial_id for trial in plan.trials}
    if report.comparison is None:
        raise WorkflowError("Protected completion requires its frozen comparison")
    comparison = report.comparison
    configs = _configurations(plan, comparison)
    _protected_choices(report, configs)
    _exact_choices(plan, report)
    grouped = _grouped_native(plan, report, configs)
    if {(reference.candidate_id, reference.metadata.group_id) for reference in comparison.inputs} != set(grouped):
        raise WorkflowError("Protected inference units differ from the source repository families")
    for reference in comparison.inputs:
        builds = grouped[reference.candidate_id, reference.metadata.group_id]
        mean = _complete_mean(tuple(tuple(values) for _, values in sorted(builds.items())))
        if reference.record.measurement(comparison.frozen.protocol.metric_name).value != mean:
            raise WorkflowError("Protected unit means differ from actual native observations")
    return {
        trial.trial_id
        for trial in plan.trials
        if trial.configuration_id in {config.configuration_id for config in configs.values()}
    }


def _protected_choices(report, configs) -> None:
    """Require reported choices to reference the actual frozen comparison family."""
    if (
        report.selected_configuration is not None
        and report.selected_configuration not in {config.configuration_id for config in configs.values()}
    ) or (
        report.fallback_configuration is not None
        and report.fallback_configuration != configs["baseline"].configuration_id
    ):
        raise WorkflowError("Protected selection/fallback differs from the frozen candidate family")


def _validate_attempts(plan, report) -> None:
    """Bind each native row to an explicit physical reservation without granting seed independence."""
    events = {event.attempt_id: event for event in report.physical_attempts}
    trials = {trial.trial_id for trial in plan.trials}
    if len(events) != len(report.physical_attempts) or any(
        event.trial_id not in trials or event.plan_fingerprint != plan.fingerprint for event in events.values()
    ):
        raise WorkflowError("Workflow report has duplicate/unplanned physical attempts")
    _attempt_limits(plan, events.values())
    for receipt in report.trials:
        for row in receipt.rows:
            event = events.get(row.attempt_id)
            if event is None or event.trial_id != receipt.trial_id:
                raise WorkflowError("Native runtime sample is not bound to its actual physical attempt")


def _attempt_limits(plan, events) -> None:
    """Require retained physical ordinals, worker starts and stages to respect the saved plan."""
    grouped = {}
    allowed = {"descriptive"} if plan.request.evaluation == "descriptive" else {"tuning", "evaluation"}
    for event in events:
        if (
            not event.ordinal <= event.worker_invocation <= plan.request.limits.max_attempts
            or event.stage not in allowed
        ):
            raise WorkflowError("Physical attempt exceeds its planned invocation/ordinal/stage contract")
        grouped.setdefault(event.trial_id, []).append(event.ordinal)
    if any(sorted(ordinals) != list(range(1, len(ordinals) + 1)) for ordinals in grouped.values()):
        raise WorkflowError("Physical attempt ordinals must be unique and contiguous within each logical trial")


def _validate_choices(plan, report) -> None:
    """Bind optional reported choices to actual configurations and a same-budget Random fallback."""
    coordinates = {trial.configuration_id: trial for trial in plan.trials}
    if any(
        choice is not None and choice not in coordinates
        for choice in (report.selected_configuration, report.fallback_configuration)
    ):
        raise WorkflowError("Workflow selection/fallback references an unplanned configuration")
    selected = coordinates.get(report.selected_configuration)
    fallback = coordinates.get(report.fallback_configuration)
    if fallback is not None and (
        fallback.policy != "Random" or (selected is not None and fallback.budget_value != selected.budget_value)
    ):
        raise WorkflowError("Workflow fallback must be the planned same-budget Random configuration")


def _exact_choices(plan, report) -> None:
    """Reuse native enumeration's deterministic winner and require its actual Random fallback.

    Protected selection reads retained validation rows only. No outcome is loaded,
    no new hypothesis is selected and no superiority claim follows from this check.
    """
    from coleman.application.workflow_evaluation import choose_configuration

    winner = choose_configuration(plan, report.trials)
    coordinates = {trial.configuration_id: trial for trial in plan.trials}
    fallback = next(
        trial.configuration_id
        for trial in plan.trials
        if trial.policy == "Random" and trial.budget_value == coordinates[winner].budget_value
    )
    if report.selected_configuration != winner or report.fallback_configuration != fallback:
        raise WorkflowError("Workflow selection/fallback differs from its deterministic native result")
