"""Frozen native-state handoff and audited comparisons using existing corpus statistics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from coleman.application.baseline_storage import read_baseline_artifact
from coleman.application.workflow_artifacts import pin_file, read_attempts, verify_trial
from coleman.artifacts import ArtifactWriter
from coleman.corpus.repeated import RepeatedCorpusReport
from coleman.corpus.repeated_contracts import (
    BUDGET_RATIO,
    CorpusBuild,
    CorpusRepetition,
    RepetitionDesign,
    repetition_execution_id,
)
from coleman.evaluation.protected import (
    ComparisonHypothesis,
    EvaluationAudit,
    FrozenCandidate,
    FrozenEvaluation,
    TemporalPlan,
    evaluate,
    freeze,
)
from coleman.results.domain import DomainRecord, Fingerprint, Measurement, ResultProvenance
from coleman.selection.contracts import Quantity, UnitSemantics
from coleman.workflows.contracts import ExperimentPlan, Trial, WorkflowContract, WorkflowError
from coleman.workflows.evidence import FrozenLearningState, FrozenWorkflowConfiguration
from coleman.workflows.lifecycle import FilePin, TrialResult, WorkflowReport


class EvaluationHandoff(WorkflowContract):
    """Immutable protected continuation decision, persisted before holdout access.

    Parameters
    ----------
    design : RepetitionDesign
        Existing full seed/build/family/clock/metric aggregation contract.
    frozen : FrozenEvaluation
        Existing candidates, learned-state identity and entire hypothesis family.
    selected_configuration, challenger_configuration, baseline_configuration : str
        Validation choice and best UCB versus same-budget Random comparison.
        A tie chooses Random, then smaller c, then canonical configuration identity.
    checkpoints : tuple[FilePin, ...]
        Private immutable prefix state snapshots; never transported or executed
        from an imported bundle. Continuation uses the original native stores.
    """

    design: RepetitionDesign
    frozen: FrozenEvaluation
    selected_configuration: Fingerprint
    challenger_configuration: Fingerprint
    baseline_configuration: Fingerprint
    checkpoints: tuple[FilePin, ...]


def choose_configuration(plan: ExperimentPlan, results: tuple[TrialResult, ...], *, ucb_only: bool = False) -> str:
    """Choose the largest declared objective mean with a conservative deterministic tie rule.

    Parameters
    ----------
    plan : ExperimentPlan
        Frozen objective and complete coordinates.
    results : tuple[TrialResult, ...]
        Complete stage receipts. Protected selection uses validation rows only.
    ucb_only : bool
        Select a challenger for comparison even when Random wins validation.

    Returns
    -------
    str
        Stable configuration fingerprint; this is a descriptive choice, not evidence
        that a policy is better. Equal binary64 observations are averaged rationally.
    """
    values: dict[str, list[float]] = {}
    coordinates = {item.trial_id: item for item in plan.trials}
    validation_steps = set(range(1, len(plan.dataset.build_ids) + 1))
    if plan.request.evaluation == "protected":
        assert plan.dataset.temporal_plan is not None
        validation_steps = {
            i
            for i, decision in enumerate(
                plan.dataset.temporal_plan.split(tuple(item.observation for item in plan.dataset.metadata)), 1
            )
            if decision.partition == "validation"
        }
    for result in results:
        trial = coordinates[result.trial_id]
        if ucb_only and trial.policy != "UCB":
            continue
        values.setdefault(trial.configuration_id, []).extend(
            getattr(row, plan.request.objective) for row in result.rows if row.step in validation_steps
        )
    representatives = {trial.configuration_id: trial for trial in plan.trials}

    def order(config: str):
        """Compare exact means before deterministic baseline/parameter/identity tie breaks."""
        trial = representatives[config]
        return (
            -sum((Fraction(value) for value in values[config]), Fraction()) / len(values[config]),
            trial.policy != "Random",
            trial.exploration or 0.0,
            config,
        )

    return min(values, key=order)


def prepare_evaluation(workspace: Path, root: Path, plan: ExperimentPlan) -> EvaluationHandoff:
    """Freeze validation choice, source protocol and actual prefix state before any test access.

    Parameters
    ----------
    workspace, root : pathlib.Path
        Trusted shared audit root and owned job artifact directory.
    plan : ExperimentPlan
        Eligible constructed protected workflow with one ratio budget.

    Returns
    -------
    EvaluationHandoff
        Complete saved handoff. Identical retries reuse it; no protected outcome
        is executed/read by this function. Prefix artifacts are verified first.

    Raises
    ------
    ValueError, OSError
        Missing prefix completion, invalid metadata/state or immutable conflict.
    """
    target = root / "evaluation-handoff.json"
    if target.exists():
        return read_baseline_artifact(target, EvaluationHandoff)
    results = tuple(
        verify_trial(root, root / "evidence" / trial.trial_id[7:] / "tuning" / "complete.json", plan, trial, "tuning")
        for trial in plan.trials
    )
    selected, challenger = choose_configuration(plan, results), choose_configuration(plan, results, ucb_only=True)
    baseline = next(trial.configuration_id for trial in plan.trials if trial.policy == "Random")
    assert plan.dataset.temporal_plan is not None
    source_plan = plan.dataset.temporal_plan
    # The original input membership/partitions stay fixed. Derived metrics cannot
    # be backdated: reserve a declared assembly window before accessing holdout.
    assembly_end = max(
        source_plan.test_end,
        datetime.now(UTC)
        + timedelta(seconds=plan.request.limits.attempt_seconds * plan.request.limits.max_attempts + 60),
    )
    analysis_plan = TemporalPlan.model_validate({**source_plan.model_dump(), "test_end": assembly_end})
    decisions = source_plan.split(tuple(item.observation for item in plan.dataset.metadata))
    builds = tuple(
        CorpusBuild(
            member_id=f"build-{item.build_id}", project_id=item.unit_id, unit_id=item.unit_id, metadata=item.observation
        )
        for item, decision in zip(plan.dataset.metadata, decisions, strict=True)
        if decision.partition == "test"
    )
    metric = plan.request.objective
    meaning = f"coleman.native.{metric}.binary64.v1"
    design = RepetitionDesign(
        schema_version=1,
        fold_id="workflow-protected-v1",
        data_fingerprint=plan.dataset.fingerprint,
        evidence_basis="constructed_mechanism",
        source_admission_fingerprint=None,
        plan=analysis_plan,
        warmup_end=analysis_plan.train_end,
        budget=Quantity(value=Decimal(str(plan.request.budget.values[0])), semantics=BUDGET_RATIO),
        seeds=plan.request.seeds,
        builds=builds,
        raw_metric=f"workflow_{metric}",
        raw_semantics=UnitSemantics(unit="1", meaning=meaning),
        cost_projection="binary32_carried_binary64_seconds_v1",
    )
    pins = []
    by_candidate = []
    for name, config in (("challenger", challenger), ("baseline", baseline)):
        learning = []
        for trial in plan.trials:
            if trial.configuration_id != config:
                continue
            native = root / "native" / trial.trial_id[7:]
            source_files = sorted(path for path in native.rglob("*") if path.is_file() and "checkpoints" in path.parts)
            if not source_files:
                raise WorkflowError("Protected continuation requires actual committed learning state")
            trial_pins = []
            for path in source_files:
                destination = root / "frozen-learning" / trial.trial_id[7:] / path.relative_to(native)
                with path.open("rb") as stream:
                    ArtifactWriter(root).write_immutable_stream(
                        destination,
                        iter(lambda: stream.read(1048576), b""),
                        maximum_bytes=plan.request.limits.max_output_bytes,
                    )
                pin = pin_file(root, destination)
                pins.append(pin)
                trial_pins.append(pin)
            learning.append(
                FrozenLearningState(trial_id=trial.trial_id, seed=trial.seed, checkpoints=tuple(trial_pins))
            )
        by_candidate.append(
            FrozenCandidate.from_configuration(
                name,
                FrozenWorkflowConfiguration(
                    configuration_id=config,
                    source_scope=plan.dataset.source_fingerprint,
                    workflow_plan=plan.fingerprint,
                    learning_state=tuple(learning),
                    corpus_repetition_design=design.fingerprint,
                ).model_dump(mode="json"),
            )
        )
    from coleman.application.workflow_validation import protected_protocol

    frozen = freeze(
        tuple(by_candidate),
        plan.dataset.fingerprint,
        analysis_plan,
        protected_protocol(plan),
        (
            ComparisonHypothesis(
                hypothesis_id="selected-ucb-versus-random", candidate_id="challenger", baseline_id="baseline"
            ),
        ),
    )
    handoff = EvaluationHandoff(
        schema_version=1,
        design=design,
        frozen=frozen,
        selected_configuration=selected,
        challenger_configuration=challenger,
        baseline_configuration=baseline,
        checkpoints=tuple(pins),
    )
    EvaluationAudit(workspace / "audit").save_freeze(frozen)
    ArtifactWriter(root).write_immutable(target, handoff.to_json())
    return handoff


def evaluation_trials(plan: ExperimentPlan, handoff: EvaluationHandoff) -> tuple[Trial, ...]:
    """Resolve the two predeclared same-budget candidates without adding configurations.

    Parameters
    ----------
    plan : ExperimentPlan
        Retained complete source/runtime/configuration context.
    handoff : EvaluationHandoff
        Persisted validation choice and comparison family.

    Returns
    -------
    tuple[Trial, ...]
        All declared seeds for challenger and Random baseline, in canonical order.
    """
    return tuple(
        trial
        for trial in plan.trials
        if trial.configuration_id in (handoff.challenger_configuration, handoff.baseline_configuration)
    )


def require_continuation(root: Path, trial: Trial, prefix_steps: int) -> None:
    """Reject missing/reset protected state before invoking native checkpoint restoration.

    Parameters
    ----------
    root : pathlib.Path
        Trusted job directory containing persisted immutable prefix snapshots.
    trial : Trial
        Selected logical coordinate whose original native paths must be reused.
    prefix_steps : int
        Fully committed training/validation frontier, in processed build ordinals.

    Raises
    ------
    ValueError, OSError
        Missing/backward state or changed initial prefix checkpoint bytes. Advanced
        frontiers still undergo the existing native checkpoint/recovery validation.
        This function never unpickles, repairs or substitutes a fresh run.
    """
    from coleman.results.domain import decode_json

    native = root / "native" / trial.trial_id[7:]
    paths = list(native.rglob("progress_ex1.json"))
    if len(paths) != 1:
        raise WorkflowError("Protected continuation cannot reset missing native learning state")
    with paths[0].open("rb") as stream:
        payload = stream.read(16385)
    if len(payload) > 16384:
        raise WorkflowError("Protected checkpoint progress exceeds its bounded schema")
    step = decode_json(payload).get("step_committed")
    if type(step) is not int or step < prefix_steps:
        raise WorkflowError("Protected checkpoint is behind its frozen prefix")
    if step == prefix_steps:
        handoff = read_baseline_artifact(root / "evaluation-handoff.json", EvaluationHandoff)
        prefix = Path("frozen-learning") / trial.trial_id[7:]
        pins = [pin for pin in handoff.checkpoints if Path(pin.name).is_relative_to(prefix)]
        if not pins:
            raise WorkflowError("Protected trial lacks its frozen learning-state membership")
        for pin in pins:
            actual = native / Path(pin.name).relative_to(prefix)
            current = pin_file(root, actual)
            if (current.sha256, current.size_bytes) != (pin.sha256, pin.size_bytes):
                raise WorkflowError("Native initial continuation state differs from its frozen snapshot")


def compare_continuation(
    workspace: Path, root: Path, plan: ExperimentPlan, handoff: EvaluationHandoff, execute_continuation
) -> WorkflowReport:
    """Reserve audited holdout access before continuing native state and aggregating evidence.

    Parameters
    ----------
    workspace, root : pathlib.Path
        Shared persistent access ledger and private job directory.
    plan : ExperimentPlan
        Frozen eligible constructed workflow, without fabricated empirical admission.
    handoff : EvaluationHandoff
        Persisted full candidates/protocol/prefix-state binding.
    execute_continuation : callable
        Trusted native callback returning actual complete evaluation stage receipts.

    Returns
    -------
    WorkflowReport
        Existing repeated-corpus aggregation and protected comparison evidence.
        Constructed demonstrations always retain an insufficient-evidence product
        interpretation and conservative fallback, independent of a numerical winner.

    Raises
    ------
    ValueError, OSError
        Changed frozen state, expired assembly window, incomplete/native evidence,
        audit exhaustion or unsupported metric. Failed access remains in the ledger.
    """
    for pin in handoff.checkpoints:
        if pin_file(root, root / pin.name) != pin:
            raise WorkflowError("Frozen learning-state snapshot changed")
    audit = EvaluationAudit(workspace / "audit")
    if len(audit.history(handoff.frozen.protocol.holdout_id)) >= 512:
        raise WorkflowError("Protected scope access slots exhausted")
    completed = []

    def observations():
        """Run only after the existing evaluator durably reserves protected access."""
        receipts = execute_continuation()
        completed.extend(receipts)
        assembled = datetime.now(UTC)
        raw = []
        trials = {trial.trial_id: trial for trial in plan.trials}
        for receipt in receipts:
            trial = trials[receipt.trial_id]
            name = "baseline" if trial.configuration_id == handoff.baseline_configuration else "challenger"
            candidate = handoff.frozen.candidate(name)
            for build in handoff.design.builds:
                row = next(row for row in receipt.rows if f"build-{row.source_build_id}" == build.member_id)
                raw.append(
                    CorpusRepetition(
                        candidate_id=name,
                        member_id=build.member_id,
                        seed=trial.seed,
                        state="completed",
                        record=DomainRecord(
                            schema_version=1,
                            namespace="ci.workflow.repetitions",
                            run_id=receipt.native_run_id,
                            execution_id=repetition_execution_id(name, trial.seed),
                            decision_id=build.member_id,
                            data_fingerprint=handoff.frozen.data_fingerprint,
                            available_at=assembled,
                            provenance=ResultProvenance(
                                producer="coleman.workflow.native-projection",
                                producer_version="1",
                                spec_fingerprint=candidate.configuration_fingerprint,
                                evaluation_fingerprint=handoff.frozen.fingerprint,
                                source_revision=receipt.fingerprint,
                            ),
                            measurements=(
                                Measurement(
                                    name=handoff.design.raw_metric,
                                    value=getattr(row, plan.request.objective),
                                    unit="1",
                                    meaning=handoff.design.raw_semantics.meaning,
                                    direction="maximize",
                                    status="observed",
                                ),
                            ),
                        ),
                    )
                )
        repeated = RepeatedCorpusReport(
            schema_version=1, design=handoff.design, frozen=handoff.frozen, rows=tuple(raw), assembled_at=assembled
        )
        ArtifactWriter(root).write_immutable(
            root / "repetitions" / f"{repeated.fingerprint[7:]}.json", repeated.to_json()
        )
        return repeated.paired_observations()

    from coleman.application.workflow_exposure import source_lock, synchronize_exposures

    with source_lock(workspace, plan.dataset.source_fingerprint):
        synchronize_exposures(workspace, handoff.frozen)
        comparison = evaluate(handoff.frozen, audit, observations, actor="local-workflow")
    audit.save_report(comparison)
    return WorkflowReport(
        schema_version=1,
        plan_fingerprint=plan.fingerprint,
        trials=tuple(completed),
        selected_configuration=handoff.selected_configuration,
        fallback_configuration=handoff.baseline_configuration,
        physical_attempts=read_attempts(root, plan),
        interpretation="insufficient_evidence",
        comparison=comparison,
        assumptions=plan.dataset.assumptions
        + (
            "Constructed source; no empirical source admission.",
            "Seeds/builds are aggregated within declared repository families; no automatic production promotion.",
        ),
    )
