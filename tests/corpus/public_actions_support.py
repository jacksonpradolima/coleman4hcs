"""Owned public-API-shaped metadata fixtures; no foreign responses or network dependencies."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

EXAMPLE = Path(__file__).parents[2] / "examples/public-ci-feasibility/ci.py"
MODULE_NAME = "coleman_owned_public_feasibility_example"


def example_module():
    """Load the complete reference example without executing its CLI or fetching metadata."""
    if MODULE_NAME not in sys.modules:
        spec = importlib.util.spec_from_file_location(MODULE_NAME, EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[MODULE_NAME] = module
        spec.loader.exec_module(module)
    return sys.modules[MODULE_NAME]


def wire(*, completed=True, zero=False):
    """Return owned metadata with a successful job and intentionally unknown test-level outcomes/costs."""
    repo = {"id": 7, "full_name": "coleman-owned/ci-example", "private": False}
    start = "2026-01-01T10:00:01Z"
    end = start if zero else "2026-01-01T10:01:01Z"
    run = {
        "id": 42,
        "run_attempt": 1,
        "head_sha": "1" * 40,
        "head_branch": "main",
        "event": "push",
        "status": "completed" if completed else "in_progress",
        "conclusion": "success" if completed else None,
        "created_at": "2026-01-01T10:00:00Z",
        "run_started_at": "2026-01-01T10:00:00Z",
        "updated_at": end if completed else start,
        "repository": repo,
        "head_repository": dict(repo),
    }
    job = {
        "id": 99,
        "run_id": 42,
        "run_attempt": 1,
        "head_sha": "1" * 40,
        "name": "owned-unit-tests",
        "status": run["status"],
        "conclusion": run["conclusion"],
        "started_at": start,
        "completed_at": end if completed else None,
        "labels": ["owned-runner-label"],
        "steps": [
            {
                "number": 1,
                "name": "Run owned tests",
                "status": run["status"],
                "conclusion": run["conclusion"],
                "started_at": start,
                "completed_at": end if completed else None,
            }
        ],
    }
    return run, {"total_count": 1, "jobs": [job]}


def request(module, capture_id="owned"):
    """Declare fixture provenance explicitly; these bytes are not observed public CI."""
    return module.Request(
        schema_version=1, repository="coleman-owned/ci-example", run_id=42, capture_id=capture_id, basis="owned_fixture"
    )


def response(module, req, part, payload, *, received="2026-01-01T10:02:00Z"):
    """Bind exact owned bytes and explicit receipt to the correct fixed-origin request."""
    body = json.dumps(payload)
    url = (
        module._urls(req)
        if part == "run"
        else module._urls(
            req,
            module.Run.model_validate(
                {
                    **wire()[0],
                    "repository": module.Repository.model_validate(wire()[0]["repository"]),
                    "head_repository": module.Repository.model_validate(wire()[0]["head_repository"]),
                }
            ),
        )
    )
    return module.Response(
        schema_version=1,
        part=part,
        url=url,
        received_at=received,
        selected_api_version=module.API_VERSION,
        body=body,
        body_sha256="sha256:" + hashlib.sha256(body.encode()).hexdigest(),
    )


def snapshot(module, *, completed=True, zero=False):
    """Build a deterministic owned v1 fixture with declared fake runtime pins, not a producer attestation."""
    req = request(module)
    raw_run, raw_jobs = wire(completed=completed, zero=zero)
    run_response = response(module, req, "run", raw_run)
    jobs_response = response(module, req, "jobs", raw_jobs)
    run = module.parse_run(run_response, req)
    jobs = module.parse_jobs(jobs_response, run)
    runtime = module.InstalledRuntime(
        schema_version=1,
        package_fingerprint="sha256:" + "a" * 64,
        python_implementation="owned-fixture",
        python_version="3.14.0",
        operating_system="owned-fixture",
        machine="owned-fixture",
        distributions=(),
    )
    return module.Snapshot(
        schema_version=1,
        request=req,
        runtime=runtime,
        example_sha256="sha256:" + "b" * 64,
        created_at=datetime.fromisoformat("2026-01-01T10:02:01Z"),
        responses=(run_response.fingerprint, jobs_response.fingerprint),
        run=run,
        jobs=jobs,
        measurements=module.measurements(jobs),
        reality=module.reality(req, run, jobs),
    )
