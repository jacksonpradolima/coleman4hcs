"""Run artifact manifest generation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from coleman.artifacts import ArtifactWriter


def generate_manifest(run_dir: str | Path, *, run_id: str) -> Path:
    """Generate ``manifest.json`` with stable relative artifact metadata."""
    root = Path(run_dir)
    writer = ArtifactWriter(root.parent)

    artifacts: list[dict[str, Any]] = []

    spec_path = root / "spec.resolved.json"
    if spec_path.exists():
        artifacts.append(
            {
                "type": "spec",
                "relative_path": _rel(spec_path, root),
                "format": "json",
                "partition_keys": [],
            }
        )

    provenance_path = root / "provenance.json"
    if provenance_path.exists():
        artifacts.append(
            {
                "type": "provenance",
                "relative_path": _rel(provenance_path, root),
                "format": "json",
                "partition_keys": [],
            }
        )

    for file_path in sorted(root.rglob("*.duckdb")):
        artifacts.append(
            {
                "type": "results",
                "relative_path": _rel(file_path, root),
                "format": "duckdb",
                "partition_keys": ["execution_id"],
            }
        )

    for file_path in sorted(root.rglob("*.parquet")):
        artifacts.append(
            {
                "type": "results",
                "relative_path": _rel(file_path, root),
                "format": "parquet",
                "partition_keys": ["scenario", "policy", "reward_function", "budget_mode", "budget_value"],
            }
        )

    for file_path in sorted(root.rglob("domain-records/*.json")):
        artifacts.append(
            {
                "type": "domain_records",
                "relative_path": _rel(file_path, root),
                "format": "json",
                "partition_keys": ["namespace", "run_id", "execution_id", "decision_id"],
            }
        )

    versioned_patterns = {
        "evaluation_contract": "evaluation-contracts/*.json",
        "evaluation_access": "evaluation-access/*/*.json",
        "evaluation_report": "evaluation-reports/*.json",
        "coverage_result": "coverage-results/*.json",
        "allocation_result": "allocation-results/*.json",
        "benchmark_report": "benchmark-reports/*.json",
        "pilot_report": "pilot-reports/*.json",
        "corpus_fixture_report": "corpus-validations/*.json",
        "corpus_readiness": "corpus-readiness/*.json",
        "corpus_source_profile": "corpus-profiles/*.json",
        "corpus_profile_inventory": "corpus-profile-inventories/*.json",
        "corpus_materialization": "corpus-materializations/*.json",
        "observation_model": "observation-models/*.json",
        "decision_signal": "decision-signals/*.json",
        "resource_run": "resource-runs/*.json",
        "resource_report": "resource-reports/*.json",
        "resource_ledger": "resource-ledgers/*/checkpoints/*.json",
        "shadow_protocol": "shadow-journals/*/protocol.json",
        "shadow_event": "shadow-journals/*/frames/*.json",
        "shadow_checkpoint": "shadow-journals/*/checkpoints/*.json",
        "shadow_report": "shadow-reports/*.json",
        "shadow_source_batch": "shadow-source-batches/*.json",
    }
    for artifact_type, pattern in versioned_patterns.items():
        for file_path in sorted(root.rglob(pattern)):
            artifacts.append(
                {"type": artifact_type, "relative_path": _rel(file_path, root), "format": "json", "partition_keys": []}
            )

    payload = {
        "manifest_version": 1,
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "artifacts": artifacts,
    }

    output_path = root / "manifest.json"
    writer.write_json_atomic(output_path, payload, indent=2)
    return output_path


def _rel(path: Path, root: Path) -> str:
    """Return POSIX-style relative path from run root."""
    return path.relative_to(root).as_posix()
