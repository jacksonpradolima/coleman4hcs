"""Real retained evidence rejects inconsistent freeze, scope capacity and checkpoint frontiers."""

import copy
import json
import pickle
from dataclasses import replace

import polars as pl
import pytest

from coleman.application.workflow_checkpoint import WorkflowCheckpointStore
from coleman.application.workflow_evaluation import compare_continuation, prepare_evaluation
from coleman.application.workflow_exports import EvidenceBundle
from coleman.application.workflow_registry import owned_fixture, snapshot_history
from coleman.application.workflows import WorkflowService
from coleman.checkpoint.checkpoint_store import LocalCheckpointStore
from coleman.checkpoint.state import CheckpointPayload
from coleman.evaluation.protected import EvaluationAudit, FrozenCandidate, PairedObservation, evaluate, freeze
from coleman.policy import base as policy_base
from coleman.results.domain import fingerprint
from coleman.workflows.contracts import PlanningRequest, WorkflowLimits
from coleman.workflows.lifecycle import ExecutePlan, HostPolicy
from tests.workflows.test_execution import archived_handoff, setup_workflow, wait


@pytest.fixture(scope="module")
def protected_evidence(tmp_path_factory):
    service, preview = setup_workflow(tmp_path_factory.mktemp("review-protected"), evaluation="protected")
    request = ExecutePlan(schema_version=1, request_id="review-protected", plan_id=preview.plan_id)
    job = service.execute(request)
    wait(service, job.job_id, "awaiting_evaluation")
    service.evaluate(job.job_id)
    wait(service, job.job_id)
    return service.export(job.job_id)


def forged_learning_bundle(bundle, mutation, directory):
    """Recompute a genuinely valid generic comparison while falsifying workflow-specific state claims."""
    comparison = bundle.report.comparison
    assert comparison is not None
    original = comparison.frozen
    challenger = original.candidate("challenger")
    configuration = json.loads(challenger.configuration_json)
    learning = configuration["learning_state"]
    if mutation == "missing_seed":
        learning.pop()
    elif mutation == "duplicate_trial":
        learning[1] = copy.deepcopy(learning[0])
    elif mutation == "foreign_trial":
        other = next(t for t in bundle.plan.trials if t.configuration_id != configuration["configuration_id"])
        learning[0]["trial_id"] = other.trial_id
        learning[0]["seed"] = other.seed
    elif mutation == "wrong_seed":
        learning[0]["seed"] = 0
    elif mutation == "foreign_checkpoint":
        pin = learning[0]["checkpoints"][0]
        pin["name"] = "frozen-learning/" + "0" * 64 + "/checkpoint.pkl"
    elif mutation == "duplicate_checkpoint":
        learning[0]["checkpoints"].append(copy.deepcopy(learning[0]["checkpoints"][0]))
    candidates = tuple(
        FrozenCandidate.from_configuration(candidate.candidate_id, configuration)
        if candidate.candidate_id == "challenger"
        else candidate
        for candidate in original.candidates
    )
    modified = freeze(
        candidates, original.data_fingerprint, original.plan, original.protocol, original.comparison_family
    )
    observations = []
    for reference in comparison.inputs:
        provenance = reference.record.provenance.model_copy(
            update={
                "spec_fingerprint": modified.candidate(reference.candidate_id).configuration_fingerprint,
                "evaluation_fingerprint": modified.fingerprint,
            }
        )
        record = reference.record.model_copy(update={"provenance": provenance})
        observations.append(
            PairedObservation(candidate_id=reference.candidate_id, metadata=reference.metadata, record=record)
        )
    audit = EvaluationAudit(directory / "audit")
    audit.save_freeze(modified)
    # No fake statistical report: the existing evaluator recomputes the entire
    # family and validates actual native-derived repository means.
    replacement = evaluate(modified, audit, lambda: tuple(observations), actor="forgery-regression")
    data = bundle.model_dump(mode="json")
    data["report"]["comparison"] = replacement.model_dump(mode="json")
    return data


@pytest.mark.parametrize(
    "mutation",
    ["missing_seed", "duplicate_trial", "foreign_trial", "wrong_seed", "foreign_checkpoint", "duplicate_checkpoint"],
)
def test_portable_frozen_learning_membership_is_bound_to_the_actual_plan(protected_evidence, tmp_path, mutation):
    payload = forged_learning_bundle(protected_evidence, mutation, tmp_path)
    with pytest.raises(ValueError):
        EvidenceBundle.model_validate(payload)


def test_protected_report_cannot_be_relabelled_as_descriptive(protected_evidence):
    payload = protected_evidence.model_dump(mode="json")
    payload["report"]["interpretation"] = "descriptive_only"
    with pytest.raises(ValueError, match="interpretation"):
        EvidenceBundle.model_validate(payload)


@pytest.mark.parametrize(
    "updates", [dict(run_id="another-run"), dict(experiment=2), dict(step=1), dict(step=3), dict(experiment=True)]
)
def test_native_payload_must_match_its_marker_before_restoring_global_rng(tmp_path, updates):
    store = WorkflowCheckpointStore(str(tmp_path), "sha256:" + "a" * 64)
    store.save(CheckpointPayload(run_id="owned", experiment=1, step=2))
    actual = LocalCheckpointStore(str(tmp_path)).load("owned", 1)
    assert actual is not None
    # Serialize only an owned native fixture; portable imports never supply or load these bytes.
    path = tmp_path / "owned" / "ckpt_ex1_step2.pkl"
    path.write_bytes(pickle.dumps(replace(actual, **updates)))
    policy_base._rng.random(3)
    before = copy.deepcopy(policy_base._rng.bit_generator.state)
    with pytest.raises(ValueError, match="payload"):
        store.load("owned", 1)
    assert policy_base._rng.bit_generator.state == before


def test_pending_source_exposure_cannot_overfill_protected_audit_capacity(tmp_path):
    service, descriptive = setup_workflow(tmp_path, seeds=(42,), ucb_exploration=(1.0,))
    job = service.execute(ExecutePlan(schema_version=1, request_id="prior-outcomes", plan_id=descriptive.plan_id))
    wait(service, job.job_id)
    protected = service.plan(
        PlanningRequest(
            schema_version=1,
            dataset_id="owned-ci-three-build-v1",
            evaluation="protected",
            seeds=(42,),
            ucb_exploration=(1.0,),
        )
    )
    job = service.execute(ExecutePlan(schema_version=1, request_id="scope-capacity", plan_id=protected.plan_id))
    wait(service, job.job_id, "awaiting_evaluation")
    root = tmp_path / "jobs" / job.job_id
    # Freeze without queueing so a retained coordinator cannot admit protected
    # work before the prior ledger is complete. Admission uses the real service.
    handoff = prepare_evaluation(tmp_path, root, service.saved_plan(protected.plan_id))
    audit = EvaluationAudit(tmp_path / "audit")
    for _ in range(511):
        audit.access(handoff.frozen, partition="test", purpose="development", actor="capacity-regression")
    service.evaluate(job.job_id)
    rejected = wait(service, job.job_id, "failed")
    assert rejected.result_digest is None
    assert rejected.attempts == 2
    assert len(audit.history(handoff.frozen.protocol.holdout_id)) == 512
    assert not list((root / "evidence").glob("*/evaluation/complete.json"))


def test_digest_suffix_cannot_replace_a_complete_owned_plan_identity(tmp_path):
    service, preview = setup_workflow(tmp_path)
    suffix = preview.plan_id[-1]
    with service.store.transaction() as connection:
        connection.execute("UPDATE plans SET id=?", (suffix,))
    with pytest.raises(ValueError, match="identity mismatch"):
        service.saved_plan(suffix)


@pytest.fixture(scope="module", params=["descriptive", "protected"])
def decision_evidence(request, tmp_path_factory):
    service, preview = setup_workflow(
        tmp_path_factory.mktemp("review-selection"),
        evaluation=request.param,
        seeds=(42,),
        ucb_exploration=(1.0,),
    )
    job = service.execute(ExecutePlan(schema_version=1, request_id="selection", plan_id=preview.plan_id))
    if request.param == "protected":
        wait(service, job.job_id, "awaiting_evaluation")
        service.evaluate(job.job_id)
    wait(service, job.job_id)
    return service.export(job.job_id)


@pytest.mark.parametrize("field", ["selected_configuration", "fallback_configuration"])
@pytest.mark.parametrize("replacement", ["null", "other_planned"])
def test_report_choices_must_match_the_native_winner_and_random_fallback(decision_evidence, field, replacement):
    payload = decision_evidence.model_dump(mode="json")
    original = payload["report"][field]
    payload["report"][field] = (
        None
        if replacement == "null"
        else next(
            trial.configuration_id for trial in decision_evidence.plan.trials if trial.configuration_id != original
        )
    )
    with pytest.raises(ValueError):
        EvidenceBundle.model_validate(payload)


def test_protected_planning_requires_two_invocations_before_expansion(tmp_path, monkeypatch):
    service, preview = setup_workflow(tmp_path)
    request = PlanningRequest(
        schema_version=1,
        dataset_id=service.saved_plan(preview.plan_id).dataset.dataset_id,
        evaluation="protected",
        limits=WorkflowLimits(max_attempts=1),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexecutable protected plans must reject before expansion")

    monkeypatch.setattr("coleman.workflows.planning.expand_sweep", forbidden)
    with pytest.raises(ValueError, match="two.*invocations"):
        service.plan(request)


def test_queued_protected_evaluation_uses_a_current_assembly_window(tmp_path, monkeypatch):
    service, preview = setup_workflow(tmp_path, evaluation="protected", seeds=(42,), ucb_exploration=(1.0,))
    job = service.execute(ExecutePlan(schema_version=1, request_id="archived-queue", plan_id=preview.plan_id))
    wait(service, job.job_id, "awaiting_evaluation")
    root = tmp_path / "jobs" / job.job_id
    original = archived_handoff(service, preview, root, monkeypatch)
    before = (root / "evaluation-handoff.json").read_bytes()
    service.evaluate(job.job_id)
    wait(service, job.job_id)
    report = service.report(job.job_id)
    assert report.comparison is not None
    assert report.comparison.frozen.plan.test_end > original.frozen.plan.test_end
    assert report.comparison.frozen.protocol == original.frozen.protocol
    assert report.comparison.access.holdout_id == original.frozen.protocol.holdout_id
    assert report.comparison.access.prior_test_accesses == 0
    assert (root / "evaluation-handoff.json").read_bytes() == before
    for candidate in report.comparison.frozen.candidates:
        previous = json.loads(original.frozen.candidate(candidate.candidate_id).configuration_json)
        current = json.loads(candidate.configuration_json)
        assert current["learning_state"] == previous["learning_state"]
        assert current["configuration_id"] == previous["configuration_id"]


def test_expired_direct_context_rejects_before_reserving_protected_access(tmp_path, monkeypatch):
    service, preview = setup_workflow(tmp_path, evaluation="protected", seeds=(42,), ucb_exploration=(1.0,))
    job = service.execute(ExecutePlan(schema_version=1, request_id="expired-context", plan_id=preview.plan_id))
    wait(service, job.job_id, "awaiting_evaluation")
    root = tmp_path / "jobs" / job.job_id
    initial = archived_handoff(service, preview, root, monkeypatch)
    plan = service.saved_plan(preview.plan_id)

    def forbidden():
        raise AssertionError("Expired analysis must reject before a protected outcome callback")

    with pytest.raises(ValueError, match="assembly window expired"):
        compare_continuation(tmp_path, root, plan, initial, forbidden)
    assert EvaluationAudit(tmp_path / "audit").history(initial.frozen.protocol.holdout_id) == ()


@pytest.fixture(scope="module")
def large_registered_history(tmp_path_factory):
    root = tmp_path_factory.mktemp("large-input-reservation")
    service = WorkflowService(
        root,
        policy=HostPolicy(
            schema_version=1,
            caps=WorkflowLimits(max_configurations=64, max_trials=128, max_output_bytes=67108864),
        ),
    )
    base = service.register(owned_fixture(root))
    frame = pl.read_parquet(root / "datasets" / f"{base.sha256.removeprefix('sha256:')}.parquet")
    payloads = [str(i) + "x" * 2097152 for i in range(frame.height)]
    path = root / "large-ci.parquet"
    frame.with_columns(pl.Series("auxiliary_log", payloads)).write_parquet(path, compression="uncompressed")
    manifest = snapshot_history(
        root,
        path,
        dataset_id="large-ci-history",
        source_fingerprint=base.source_fingerprint,
        projection_fingerprint=fingerprint([base.projection_fingerprint, "auxiliary-log-v1"]),
        evidence=base.evidence,
        assumptions=("Authored CI history with auxiliary log context.",),
    )
    assert manifest.size_bytes > 16777216
    service.register(manifest)
    return service, manifest


@pytest.mark.parametrize("allowance", ["default", "input_size"])
def test_registered_input_must_leave_output_allowance_before_expansion(
    large_registered_history, monkeypatch, allowance
):
    service, manifest = large_registered_history
    limits = WorkflowLimits() if allowance == "default" else WorkflowLimits(max_output_bytes=manifest.size_bytes)
    request = PlanningRequest(schema_version=1, dataset_id=manifest.dataset_id, limits=limits)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexecutable input reservations must reject before expansion")

    monkeypatch.setattr("coleman.workflows.planning.expand_sweep", forbidden)
    with pytest.raises(ValueError, match="input.*output"):
        service.plan(request)
