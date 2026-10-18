"""Owned source bindings and frozen explicit reviews for offline admission tests."""

from datetime import UTC, datetime
from pathlib import Path

from coleman.corpus.admission import SourceEvidence
from coleman.corpus.admission_contracts import SourceReview
from coleman.corpus.admission_sources import declared_history
from coleman.corpus.controlled import convert_controlled_source
from coleman.corpus.controlled_contracts import ControlledDataset
from coleman.corpus.materialization import SourceCache
from tests.corpus.test_controlled import source
from tests.corpus.test_lrts_etl import exported

FIXTURES = Path(__file__).with_name("fixtures")


def changed(model, **values):
    """Revalidate modifications; unchecked-copy cases are constructed explicitly in their individual tests."""
    return type(model).model_validate({**model.model_dump(), **values})


def review(payload, *, history=None, status="reviewed", **values):
    """Freeze owned metadata attestations; no real-project observation or curated fault is implied."""
    fields = {
        "schema_version": 1,
        "payload_fingerprint": payload.fingerprint,
        "status": status,
        "reviewed_at": datetime(2026, 10, 18, tzinfo=UTC) if status == "reviewed" else None,
        "history": history or declared_history(payload) or "unverified",
        "reference": "urn:coleman:owned-admission-contract-review-v1",
        "fault_reference": None,
        "subject_language": payload.spec.context.language if isinstance(payload, ControlledDataset) else None,
        "limitations": ("Owned schema/contract evidence; no observed historical CI or independent faults.",),
    }
    fields.update(values)
    return SourceReview.model_validate(fields)


def lrts_source(tmp_path):
    """Materialize actual owned nested source archives and deterministic analytical/native Parquet."""
    completion, payload, cache = exported(tmp_path)
    return SourceEvidence(payload, review(payload), completion.parent, cache), completion


def controlled_source(tmp_path, *, rows=None, context=None):
    """Materialize the existing owned controlled interchange without executing any foreign source."""
    cache = SourceCache(tmp_path / "cache")
    spec = source(cache, rows=rows, ctx=context)
    payload = convert_controlled_source(spec, cache)
    return SourceEvidence(payload, review(payload), tmp_path, cache)
