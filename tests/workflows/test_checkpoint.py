"""Actual native global-RNG checkpoint extension, legacy rejection and mathematical continuation."""

import json
import subprocess
import sys

import numpy as np
import pytest

from coleman.application.baseline_lock import baseline_lock
from coleman.application.workflow_checkpoint import WorkflowCheckpointStore, WorkflowRandomState
from coleman.application.workflow_store import JobRecord
from coleman.checkpoint.checkpoint_store import LocalCheckpointStore
from coleman.checkpoint.state import CheckpointPayload
from coleman.policy import base as policy_base
from coleman.workflows.lifecycle import ExecutePlan
from tests.workflows.test_execution import _CRASH_AT_REAL_CHECKPOINT, mathematics, setup_workflow, wait


def test_actual_global_generator_and_cached_uint32_restore_exactly(tmp_path, monkeypatch):
    monkeypatch.setattr(policy_base, "_rng", np.random.default_rng(42))
    # Consume one 32-bit sample so the generator can retain a cached half-word.
    policy_base._rng.integers(0, 2**31)
    store = WorkflowCheckpointStore(str(tmp_path), "sha256:" + "a" * 64)
    payload = CheckpointPayload(run_id="owned", experiment=1, step=2)
    store.save(payload)
    assert payload.extra == {}
    expected = policy_base._rng.permutation(31).tolist()
    policy_base._rng.permutation(31)
    restored = store.load("owned", 1)
    assert restored is not None
    assert policy_base._rng.permutation(31).tolist() == expected
    state = WorkflowRandomState.from_json(restored.extra["coleman_workflow_rng_v1"])
    assert state.to_json() == WorkflowRandomState.from_json(state.to_json()).to_json()
    assert state.increment > 2**64
    before = policy_base._rng.bit_generator.state
    with pytest.raises(ValueError, match="binding"):
        WorkflowCheckpointStore(str(tmp_path), "sha256:" + "b" * 64).load("owned", 1)
    assert policy_base._rng.bit_generator.state == before
    with pytest.raises(ValueError):
        WorkflowRandomState.model_validate({**state.model_dump(), "schema_version": 2})


def test_ordinary_native_checkpoints_are_not_implicitly_reinterpreted(tmp_path):
    LocalCheckpointStore(str(tmp_path)).save(CheckpointPayload(run_id="ordinary", experiment=1, step=1))
    with pytest.raises(ValueError, match="implicit reseeding"):
        WorkflowCheckpointStore(str(tmp_path), "sha256:" + "a" * 64).load("ordinary", 1)
    assert WorkflowCheckpointStore(str(tmp_path), "sha256:" + "a" * 64).load("not-started", 1) is None
    marker = tmp_path / "ordinary" / "progress_ex1.json"
    raw = json.loads(marker.read_text())
    raw["checkpoint_path"] = str(tmp_path / "elsewhere.pkl")
    marker.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="not owned"):
        WorkflowCheckpointStore(str(tmp_path), "sha256:" + "a" * 64).load("ordinary", 1)


def test_random_resume_after_consumed_rng_matches_uninterrupted_native_run(tmp_path, monkeypatch):
    service, preview = setup_workflow(tmp_path, policies=("Random",))
    with monkeypatch.context() as patch:
        patch.setattr(service, "_start", lambda: None)
        job = service.execute(ExecutePlan(schema_version=1, request_id="random-interrupted", plan_id=preview.plan_id))
    with baseline_lock(tmp_path / "locks" / f"{job.job_id}.lock", handoff=True) as descriptor:
        with service.store.transaction() as connection:
            record = service._job(connection, job.job_id)
            service.store.save_job(
                connection, JobRecord.model_validate({**record.model_dump(), "state": "running", "invocations": 1})
            )
        # Unlike the first build's agent-local shuffle, step 2 has consumed the
        # native policy module's global generator; resetting it changes future orders.
        code = _CRASH_AT_REAL_CHECKPOINT.replace("step==1", "step==2")
        process = subprocess.Popen(
            [sys.executable, "-c", code, str(tmp_path), job.job_id],
            pass_fds=(descriptor,),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    _, error = process.communicate(timeout=30)
    assert process.returncode == 87, error.decode()
    assert service.status(job.job_id).checkpoint_step == 2
    service.resume(job.job_id)
    wait(service, job.job_id)
    resumed = service.report(job.job_id)
    fresh = service.execute(ExecutePlan(schema_version=1, request_id="random-uninterrupted", plan_id=preview.plan_id))
    wait(service, fresh.job_id)
    assert mathematics(resumed) == mathematics(service.report(fresh.job_id))
    assert len(resumed.physical_attempts) == 3 and len(resumed.trials) == 2
