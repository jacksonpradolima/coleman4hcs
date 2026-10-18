"""Immutable generated artifacts, independent regeneration, interruption and concurrency."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import pytest

from coleman.artifacts import ArtifactWriter
from coleman.corpus.stress_artifacts import export_stress, validate_stress
from coleman.corpus.stress_contracts import StressReceipt
from tests.corpus.stress_support import case


def exported(tmp_path, **factors):
    completion = export_stress(case(**factors), ArtifactWriter(tmp_path / "export"))
    return completion, StressReceipt.from_json(completion.read_bytes())


def test_export_round_trip_exact_native_truth_and_concurrent_idempotence(tmp_path):
    value = case(churn="collisions", variants="changing")
    with ThreadPoolExecutor(max_workers=3) as pool:
        paths = list(pool.map(lambda _: export_stress(value, ArtifactWriter(tmp_path)), range(3)))
    assert len(set(paths)) == 1
    root = paths[0].parent
    receipt = StressReceipt.from_json(paths[0].read_bytes())
    validate_stress(receipt, root)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert export_stress(value, ArtifactWriter(tmp_path)) == paths[0]
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    native = pl.read_parquet(root / receipt.files[-1].path)
    assert native.height == 32
    assert native["Name"].n_unique() == 24  # v0 recurs in the fourth build
    assert native["Duration"].null_count() == 0
    assert receipt.setup_seconds is None
    assert receipt.observed_wall_seconds is None


def test_interruption_after_first_complete_file_never_publishes_completion_and_retry_converges(tmp_path, monkeypatch):
    original = ArtifactWriter.write_immutable_stream
    calls = 0

    def interrupted(self, path, chunks, *, maximum_bytes):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("owned stop before second publication")
        return original(self, path, chunks, maximum_bytes=maximum_bytes)

    with monkeypatch.context() as fault:
        fault.setattr(ArtifactWriter, "write_immutable_stream", interrupted)
        with pytest.raises(OSError, match="second publication"):
            export_stress(case(), ArtifactWriter(tmp_path))
    assert len(list(tmp_path.rglob("*.jsonl"))) == 1
    assert not list(tmp_path.rglob("complete.json"))
    before = next(tmp_path.rglob("*.jsonl")).read_bytes()
    completion = export_stress(case(), ArtifactWriter(tmp_path))
    validate_stress(StressReceipt.from_json(completion.read_bytes()), completion.parent)
    assert next(tmp_path.rglob("build-00.jsonl")).read_bytes() == before


def test_modified_bytes_and_self_signed_receipts_cannot_certify_truth(tmp_path):
    completion, receipt = exported(tmp_path)
    item = receipt.files[0]
    path = completion.parent / item.path
    old = path.read_bytes()
    path.write_bytes(old[:-1])
    with pytest.raises(ValueError, match="artifact bytes"):
        validate_stress(receipt, completion.parent)
    with pytest.raises(ValueError, match="Conflicting|conflict"):
        export_stress(receipt.case, ArtifactWriter(tmp_path / "export"))
    path.write_bytes(old)
    forged = receipt.model_dump()
    forged["files"][0]["sha256"] = "sha256:" + "a" * 64
    with pytest.raises(ValueError, match="regenerated truth"):
        validate_stress(StressReceipt.model_validate(forged), completion.parent)
    for field in ["producer", "parquet_writer"]:
        forged = receipt.model_dump()
        forged[field] = "sha256:" + "b" * 64 if field == "producer" else "unknown-writer"
        with pytest.raises(ValueError, match="Unsupported stress"):
            validate_stress(StressReceipt.model_validate(forged), completion.parent)


def test_future_versions_file_roles_and_row_conservation_fail_closed(tmp_path):
    _, receipt = exported(tmp_path)
    for field, value in [("schema_version", 2), ("files", receipt.files[:-1])]:
        with pytest.raises(ValueError):
            StressReceipt.model_validate({**receipt.model_dump(), field: value})
    invalid = receipt.model_dump()
    invalid["files"][-1]["rows"] += 1
    with pytest.raises(ValueError, match="conservation"):
        StressReceipt.model_validate(invalid)
    with pytest.raises(ValueError, match="roles"):
        StressReceipt.model_validate({**receipt.model_dump(), "files": tuple(reversed(receipt.files))})


def test_output_must_be_external_and_fixed_roles_cannot_follow_escaping_symlinks(tmp_path):
    with pytest.raises(ValueError, match="outside a repository"):
        export_stress(case(), ArtifactWriter(Path(__file__).parent / "generated-must-not-exist"))
    output = tmp_path / "output"
    output.mkdir()
    (output / "stress").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        export_stress(case(), ArtifactWriter(output))
    assert not (tmp_path / "elsewhere").exists()
