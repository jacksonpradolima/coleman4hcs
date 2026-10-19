"""Actual cache/storage recovery and source verification for original CTest corroboration."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from coleman.artifacts import ArtifactWriter
from coleman.corpus.contracts import LicenseReview, SourceArtifact
from coleman.corpus.hcs_trace_artifacts import (
    HCSLogSpec,
    HCSTraceCompletion,
    publish_hcs_trace_evidence,
    validate_hcs_trace_completion,
)
from coleman.corpus.materialization_contracts import MaterializationReceipt
from tests.corpus.hcs_support import prepare_hcs
from tests.corpus.test_hcs_trace import OWNED, changed


def owned_log_spec():
    """Pin authored bytes while preserving a pending component review, without inventing licensing clearance."""
    return HCSLogSpec(
        schema_version=1,
        job_id=1,
        source=SourceArtifact(
            uri="urn:coleman:owned-ctest-log",
            version="owned-v1",
            citation="Authored tiny CTest-format fixture, not historical CI evidence",
            sha256="sha256:" + hashlib.sha256(OWNED).hexdigest(),
            size_bytes=len(OWNED),
            availability="verified_local",
        ),
        terms=LicenseReview(
            status="pending",
            license_id=None,
            terms_uri=None,
            terms_sha256=None,
            reviewed_at=None,
            redistribution="unverified",
        ),
    )


def test_complete_last_idempotent_concurrent_publication_and_full_actual_source_revalidation(tmp_path):
    recipe, cache = prepare_hcs(tmp_path)
    spec, writer = owned_log_spec(), ArtifactWriter(tmp_path / "output")
    with ThreadPoolExecutor(max_workers=3) as pool:
        paths = list(pool.map(lambda _: publish_hcs_trace_evidence(recipe, spec, OWNED, cache, writer), range(3)))
    assert len(set(paths)) == 1
    path = paths[0]
    before = {p: p.read_bytes() for p in path.parent.iterdir()}
    receipt = validate_hcs_trace_completion(path, recipe, OWNED, cache)
    assert receipt.log_spec.terms.status == "pending"
    assert receipt.admission == "not_granted"
    assert receipt.files[0].rows == 3
    assert HCSTraceCompletion.from_json(receipt.to_json()) == receipt
    assert publish_hcs_trace_evidence(recipe, spec, OWNED, cache, writer) == path
    assert {p: p.read_bytes() for p in path.parent.iterdir()} == before
    assert set(p.name for p in path.parent.iterdir()) == {"trace.json", "correspondence.json", "complete.json"}
    for member in [recipe.executions, recipe.jobs]:
        original = cache.member_path(MaterializationReceipt(schema_version=1, spec=member), member.members[0].name)
        data = original.read_bytes()
        original.write_bytes(data + b" ")
        with pytest.raises(ValueError):
            validate_hcs_trace_completion(path, recipe, OWNED, cache)
        original.write_bytes(data)


def test_interrupted_write_retains_partial_files_without_completion_and_retry_continues(tmp_path):
    recipe, cache = prepare_hcs(tmp_path)

    class Interrupted(ArtifactWriter):
        """Fail after the first durable evidence artifact, before publishing correspondence/completion."""

        def write_immutable(self, path, content):
            if path.name == "correspondence.json":
                raise OSError("owned write interruption")
            return super().write_immutable(path, content)

    root = tmp_path / "output"
    writer, spec = Interrupted(root), owned_log_spec()
    with pytest.raises(OSError, match="owned write interruption"):
        publish_hcs_trace_evidence(recipe, spec, OWNED, cache, writer)
    first = next(root.rglob("trace.json"))
    before = first.read_bytes()
    assert not list(root.rglob("complete.json"))
    path = publish_hcs_trace_evidence(recipe, spec, OWNED, cache, ArtifactWriter(root))
    assert first.read_bytes() == before
    validate_hcs_trace_completion(path, recipe, OWNED, cache)


@pytest.mark.parametrize("member", ["trace.json", "correspondence.json", "complete.json"])
def test_modified_output_or_forged_completion_cannot_validate_or_be_silently_overwritten(tmp_path, member):
    recipe, cache = prepare_hcs(tmp_path)
    spec, writer = owned_log_spec(), ArtifactWriter(tmp_path / "output")
    path = publish_hcs_trace_evidence(recipe, spec, OWNED, cache, writer)
    target = path.parent / member
    target.write_bytes(target.read_bytes() + b" ")
    if member == "complete.json":
        payload = json.loads(target.read_bytes())
        payload["producer_sha256"] = "sha256:" + "0" * 64
        target.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        validate_hcs_trace_completion(path, recipe, OWNED, cache)
    with pytest.raises(ValueError):
        publish_hcs_trace_evidence(recipe, spec, OWNED, cache, writer)


@pytest.mark.parametrize("kind", ["log_digest", "log_size", "job", "partial", "prohibited", "unknown_version"])
def test_invalid_or_unreviewed_claims_fail_before_false_completion(tmp_path, kind):
    recipe, cache = prepare_hcs(tmp_path)
    spec, payload = owned_log_spec(), OWNED
    if kind == "log_digest":
        payload = payload.replace(b"2.00 sec", b"2.01 sec")
    elif kind == "log_size":
        payload += b"\n"
    elif kind == "job":
        spec = changed(spec, job_id=999)
    elif kind == "partial":
        payload = b"Test project /owned\n"
        source = changed(spec.source, size_bytes=len(payload), sha256="sha256:" + hashlib.sha256(payload).hexdigest())
        spec = changed(spec, source=source)
    elif kind == "prohibited":
        terms = changed(spec.terms, status="prohibited")
        with pytest.raises(ValueError, match="prohibit"):
            changed(spec, terms=terms)
        return
    else:
        wire = json.loads(spec.to_json())
        wire["schema_version"] = 2
        with pytest.raises(ValueError):
            HCSLogSpec.from_json(json.dumps(wire))
        return
    root = tmp_path / "output"
    with pytest.raises(ValueError):
        publish_hcs_trace_evidence(recipe, spec, payload, cache, ArtifactWriter(root))
    assert not list(root.rglob("complete.json"))


def test_source_cache_overlap_and_output_symlinks_reject_before_writing(tmp_path):
    recipe, cache = prepare_hcs(tmp_path)
    spec = owned_log_spec()
    with pytest.raises(ValueError):
        publish_hcs_trace_evidence(recipe, spec, OWNED, cache, ArtifactWriter(cache.root))
    root, escape = tmp_path / "output", tmp_path / "escape"
    root.mkdir()
    escape.mkdir()
    (root / "hcs-trace").symlink_to(escape, target_is_directory=True)
    with pytest.raises(ValueError):
        publish_hcs_trace_evidence(recipe, spec, OWNED, cache, ArtifactWriter(root))
    assert not list(escape.iterdir())


def test_redistribution_prohibition_is_not_silently_reinterpreted_as_prohibited_local_analysis(tmp_path):
    recipe, cache = prepare_hcs(tmp_path)
    spec = owned_log_spec()
    terms = changed(spec.terms, redistribution="prohibited")
    reviewed = changed(spec, terms=terms)
    path = publish_hcs_trace_evidence(recipe, reviewed, OWNED, cache, ArtifactWriter(tmp_path / "output"))
    receipt = validate_hcs_trace_completion(path, recipe, OWNED, cache)
    assert receipt.log_spec.terms.redistribution == "prohibited"
    assert {p.name for p in path.parent.iterdir()} == {"trace.json", "correspondence.json", "complete.json"}
    assert receipt.admission == "not_granted"
