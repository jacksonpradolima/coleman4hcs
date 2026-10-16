"""Independent source semantics, conservation, missingness and immutable RTPTorrent conversion."""

import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest

from coleman.artifacts import ArtifactWriter
from coleman.corpus import MaterializationSpec, SourceCache, load_registry, materialize_source
from coleman.corpus.rtptorrent import (
    convert_rtptorrent_source,
    publish_rtptorrent_dataset,
    validate_rtptorrent_dataset,
)
from coleman.corpus.rtptorrent_contracts import REPORTED_SECONDS, ClassExecution, RTPTorrentDataset, RTPTorrentSpec
from coleman.selection.contracts import Quantity, UnitSemantics

HEADER = "travisJobId,testName,index,duration,count,failures,errors,skipped\n"
ROWS = (
    "101,owned.Repeated,0,0,1,0,0,0\n",
    "101,owned.Repeated,1,2.50,1,0,0,0\n",
    "102,owned.Failed,0,3,2,1,1,0\n",
    "103,owned.Unknown,0,,,,,\n",
    "104,owned.Skipped,0,0,2,0,0,2\n",
)
SOURCE = HEADER + "".join(ROWS)


def materialized(tmp_path, text=SOURCE):
    """Own the tiny source; use real licensed materialization and content-addressed cache."""
    payload = text.encode() if isinstance(text, str) else text
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    package = MaterializationSpec.model_validate(
        {
            "schema_version": 1,
            "package_id": "owned-rtptorrent-test-v1",
            "archive": "raw",
            "source": {
                "uri": "urn:coleman:owned:rtptorrent-test-v1",
                "version": "1.1",
                "citation": "Repository-authored CSV; not an external corpus",
                "sha256": digest,
                "size_bytes": len(payload),
                "availability": "not_verified",
            },
            "license": load_registry().card("ci-serial-boundary-v1").manifest.license,
            "members": [{"name": "class-results.csv", "sha256": digest, "size_bytes": len(payload)}],
        }
    )
    cache = SourceCache(tmp_path / "cache")
    receipt = materialize_source(package, cache, lambda: io.BytesIO(payload))
    spec = RTPTorrentSpec(
        schema_version=1, materialization=receipt, member="class-results.csv", project="owned/ci", history="constructed"
    )
    return spec, cache


def test_roundtrip_golden_and_independent_source_counts(tmp_path):
    spec, cache = materialized(tmp_path)
    dataset = convert_rtptorrent_source(spec, cache)
    assert dataset.input_rows == 5
    assert dataset.duplicate_rows_excluded == 0
    assert dataset.job_ids == (101, 102, 103, 104)
    assert dataset.repeated_class_executions == 1
    assert dataset.reported_zero_failure_jobs == 2
    assert dataset.job(102)[0].failures == 1
    assert dataset.job(102)[0].errors == 1
    assert dataset.job(102)[0].outcome == "failed"
    assert dataset.job(103)[0].outcome == "unknown"
    assert dataset.job(104)[0].outcome == "not_run"
    assert RTPTorrentDataset.from_json(dataset.to_json()) == dataset
    golden = Path(__file__).with_name("fixtures") / "rtptorrent-owned-slice-v1.json"
    assert dataset.to_json() + "\n" == golden.read_text()
    validate_rtptorrent_dataset(dataset, cache)
    with pytest.raises(ValueError, match="availability"):
        dataset.require_temporal_evaluation()


def test_zero_unavailable_and_signed_zero_are_distinct(tmp_path):
    spec, cache = materialized(tmp_path, HEADER + ROWS[0] + ROWS[3])
    dataset = convert_rtptorrent_source(spec, cache)
    assert dataset.job(101)[0].duration == Quantity(value=Decimal(0), semantics=REPORTED_SECONDS)
    assert dataset.job(103)[0].duration is None
    assert dataset.job(103)[0].duration_status == "unavailable"
    assert dataset.job(101)[0].duration_status == "observed"
    assert dataset.spec.duration_semantics == REPORTED_SECONDS
    assert dataset.job(103)[0].count is None
    positive = dataset.job(101)[0]
    signed = ClassExecution.model_validate(
        {**positive.model_dump(), "duration": {"value": "-0", "semantics": REPORTED_SECONDS}}
    )
    assert positive == signed
    assert signed.duration is not None
    assert str(signed.duration.value) == "0"
    unavailable = {**positive.model_dump(), "duration": None}
    with pytest.raises(ValueError, match="unavailable status"):
        ClassExecution.model_validate(unavailable)
    inapplicable = {**positive.model_dump(), "duration_status": "not_applicable"}
    with pytest.raises(ValueError):
        ClassExecution.model_validate(inapplicable)


@pytest.mark.parametrize(
    "duration", ["-1", "NaN", "nan", "Infinity", "-Infinity", "1e-13", "1e-999999999", "1000000000001", " 1"]
)
def test_invalid_duration_never_becomes_zero_or_unavailable(tmp_path, duration):
    spec, cache = materialized(tmp_path, HEADER + f"101,owned.Test,0,{duration},1,0,0,0\n")
    with pytest.raises(ValueError):
        convert_rtptorrent_source(spec, cache)


@pytest.mark.parametrize("counts", ["1,2,0,0", "1,0,1,1", "-1,0,0,0", "1.0,0,0,0", "01,0,0,0", "1,true,0,0"])
def test_invalid_method_counts_reject_without_repair(tmp_path, counts):
    spec, cache = materialized(tmp_path, HEADER + f"101,owned.Test,0,1,{counts}\n")
    with pytest.raises(ValueError):
        convert_rtptorrent_source(spec, cache)


def test_incompatible_duration_semantics_and_future_versions_reject(tmp_path):
    spec, cache = materialized(tmp_path)
    dataset = convert_rtptorrent_source(spec, cache)
    payload = json.loads(dataset.to_json())
    payload["executions"][0]["duration"]["semantics"] = UnitSemantics(unit="ms", meaning="milliseconds").model_dump()
    with pytest.raises(ValueError):
        RTPTorrentDataset.model_validate(payload)
    wrong_units = {**spec.model_dump(), "duration_semantics": {"unit": "ms", "meaning": "time"}}
    with pytest.raises(ValueError, match="units/meaning"):
        RTPTorrentSpec.model_validate(wrong_units)
    for changed in ({"schema_version": 2}, {"schema_version": True}, {"recipe": "rtptorrent-class-results-v2"}):
        incompatible = {**spec.model_dump(), **changed}
        with pytest.raises(ValueError):
            RTPTorrentSpec.model_validate(incompatible)
    for version in (None, 0, 2, True):
        incompatible = {**dataset.model_dump(), "schema_version": version}
        with pytest.raises(ValueError):
            RTPTorrentDataset.model_validate(incompatible)
    with pytest.raises(ValueError):
        RTPTorrentDataset.from_json('{"schema_version":1,"schema_version":2}')
    with pytest.raises(ValueError):
        RTPTorrentDataset.from_json('{"BuildId":1,"Duration":0,"Verdict":0}')


def test_row_permutation_and_decimal_spelling_preserve_logical_identities(tmp_path):
    first, cache = materialized(tmp_path / "first")
    second, other_cache = materialized(tmp_path / "second", HEADER + "".join(reversed(ROWS)).replace("2.50", "2.500"))
    a = convert_rtptorrent_source(first, cache)
    b = convert_rtptorrent_source(second, other_cache)
    assert a.executions == b.executions
    assert [row.execution_id("owned/ci") for row in a.executions] == [
        row.execution_id("owned/ci") for row in b.executions
    ]
    assert a.spec.materialization.spec.source.sha256 != b.spec.materialization.spec.source.sha256
    assert a.fingerprint != b.fingerprint  # Raw source provenance stays exact.
    assert a.executions[0].execution_id("different/project") != a.executions[0].execution_id("owned/ci")


def test_duplicate_conservation_conflicts_and_repeated_class_identity(tmp_path):
    spec, cache = materialized(tmp_path, HEADER + ROWS[0] + ROWS[0] + ROWS[1])
    dataset = convert_rtptorrent_source(spec, cache)
    assert dataset.input_rows == 3
    assert dataset.duplicate_rows_excluded == 1
    assert len(dataset.executions) == 2
    assert dataset.repeated_class_executions == 1
    conflict, other = materialized(tmp_path / "conflict", HEADER + ROWS[0] + ROWS[0].replace(",0,1,", ",1,1,"))
    with pytest.raises(ValueError, match="Conflicting"):
        convert_rtptorrent_source(conflict, other)
    wrong_count = {**dataset.model_dump(), "input_rows": 4}
    with pytest.raises(ValueError, match="conservation"):
        RTPTorrentDataset.model_validate(wrong_count)
    duplicated = {
        **dataset.model_dump(),
        "input_rows": 4,
        "executions": dataset.executions * 2,
        "duplicate_rows_excluded": 0,
    }
    with pytest.raises(ValueError, match="Duplicate"):
        RTPTorrentDataset.model_validate(duplicated)


@pytest.mark.parametrize(
    "source",
    [
        "",
        HEADER,
        HEADER.replace("duration", "Duration") + ROWS[0],
        HEADER + "\n",
        HEADER + '101,"unterminated\n',
        "travisBuildNumber,travisBuildId," + HEADER + "1,2," + ROWS[0],
    ],
)
def test_empty_malformed_and_filtered_baseline_inputs_reject(tmp_path, source):
    spec, cache = materialized(tmp_path, source)
    with pytest.raises(ValueError):
        convert_rtptorrent_source(spec, cache)


def test_declared_row_line_bounds_and_corrupt_cache(tmp_path):
    spec, cache = materialized(tmp_path)
    one = RTPTorrentSpec.model_validate({**spec.model_dump(), "maximum_rows": 1})
    with pytest.raises(ValueError, match="bounds"):
        convert_rtptorrent_source(one, cache)
    long, other = materialized(tmp_path / "long", HEADER + "101," + "A" * 200 + ",0,1,1,0,0,0\n")
    bounded = RTPTorrentSpec.model_validate({**long.model_dump(), "maximum_line_bytes": 128})
    with pytest.raises(ValueError, match="bounds"):
        convert_rtptorrent_source(bounded, other)
    path = cache.member_path(spec.materialization, spec.member)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum|byte count"):
        convert_rtptorrent_source(spec, cache)


def test_revalidation_unchecked_models_and_source_correspondence(tmp_path):
    spec, cache = materialized(tmp_path)
    dataset = convert_rtptorrent_source(spec, cache)
    payload = dataset.model_dump()
    payload["executions"][0]["duration"]["value"] = Decimal(99)
    changed = RTPTorrentDataset.model_validate(payload)
    with pytest.raises(ValueError, match="source bytes"):
        validate_rtptorrent_dataset(changed, cache)
    unchecked = dataset.model_copy(update={"input_rows": 99})
    with pytest.raises(ValueError):
        unchecked.to_json()
    writer = ArtifactWriter(tmp_path / "exports")
    with pytest.raises(ValueError):
        publish_rtptorrent_dataset(changed, cache, writer)
    assert not list((tmp_path / "exports").rglob("*.json"))


def test_idempotent_concurrent_publication_and_interrupted_write(tmp_path, monkeypatch):
    spec, cache = materialized(tmp_path)
    dataset = convert_rtptorrent_source(spec, cache)
    writer = ArtifactWriter(tmp_path / "exports")
    with ThreadPoolExecutor(max_workers=4) as executor:
        paths = list(executor.map(lambda _: publish_rtptorrent_dataset(dataset, cache, writer), range(4)))
    assert len(set(paths)) == 1
    assert RTPTorrentDataset.from_json(paths[0].read_bytes()) == dataset
    assert not list(writer.run_root.rglob("*.tmp"))
    failed_writer = ArtifactWriter(tmp_path / "interrupted")
    real_link = os.link

    def interrupted(src, dst, *args, **kwargs):
        if str(dst).endswith(".json"):
            raise OSError("interrupted publication")
        return real_link(src, dst, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "link", interrupted)
        with pytest.raises(OSError, match="interrupted"):
            publish_rtptorrent_dataset(dataset, cache, failed_writer)
    assert not list(failed_writer.run_root.rglob("*.json"))
    assert not list(failed_writer.run_root.rglob("*.tmp"))
    assert publish_rtptorrent_dataset(dataset, cache, failed_writer).exists()


def test_publication_confines_family_and_rejects_overlapping_cache(tmp_path):
    spec, cache = materialized(tmp_path)
    dataset = convert_rtptorrent_source(spec, cache)
    overlapping_writer = ArtifactWriter(cache.root)
    with pytest.raises(ValueError, match="separate"):
        publish_rtptorrent_dataset(dataset, cache, overlapping_writer)
    root = tmp_path / "exports"
    root.mkdir()
    (root / "corpus-rtptorrent").symlink_to(cache.root, target_is_directory=True)
    writer = ArtifactWriter(root)
    with pytest.raises(ValueError, match="output tree"):
        publish_rtptorrent_dataset(dataset, cache, writer)
    for label in (True, 0, -1, 101.0):
        invalid = cast(int, label)
        with pytest.raises(ValueError):
            dataset.job(invalid)
    with pytest.raises(KeyError):
        dataset.job(999)


def test_spec_member_version_and_payload_limits(tmp_path):
    spec, _ = materialized(tmp_path)
    missing = {**spec.model_dump(), "member": "missing.csv"}
    with pytest.raises(ValueError, match="member"):
        RTPTorrentSpec.model_validate(missing)
    payload = spec.model_dump()
    payload["materialization"]["spec"]["source"]["version"] = "2.0"
    with pytest.raises(ValueError, match="version"):
        RTPTorrentSpec.model_validate(payload)
    payload = spec.model_dump()
    payload["materialization"]["spec"]["source"]["size_bytes"] = 17 * 1024**2
    payload["materialization"]["spec"]["members"][0]["size_bytes"] = 17 * 1024**2
    with pytest.raises(ValueError, match="16 MiB"):
        RTPTorrentSpec.model_validate(payload)


@pytest.mark.parametrize(
    "source",
    [
        HEADER + "0,owned.Test,0,1,1,0,0,0\n",
        HEADER + "101,,0,1,1,0,0,0\n",
        HEADER + "101,owned.Test,0,1_0,1,0,0,0\n",
        HEADER + "101,owned.Test,0,١,1,0,0,0\n",
        HEADER + "101,owned.Test,0,1,1,0,0\n",
        HEADER + "101," + "X" * 1025 + ",0,1,1,0,0,0\n",
        HEADER.encode() + b"101,\xff,0,1,1,0,0,0\n",
    ],
)
def test_malformed_fields_are_not_coerced(tmp_path, source):
    spec, cache = materialized(tmp_path, source)
    with pytest.raises(ValueError):
        convert_rtptorrent_source(spec, cache)


def test_partial_counts_known_failure_and_not_run_are_preserved(tmp_path):
    spec, cache = materialized(tmp_path, HEADER + "101,owned.Test,0,0,,1,,\n102,owned.Nothing,0,,0,0,0,0\n")
    dataset = convert_rtptorrent_source(spec, cache)
    assert dataset.job(101)[0].outcome == "failed"
    assert dataset.job(101)[0].count is None
    assert dataset.job(102)[0].outcome == "not_run"
    assert dataset.job(102)[0].duration is None
    assert dataset.reported_zero_failure_jobs == 1


def test_parsed_bytes_are_pinned_even_if_cache_changes_after_preflight(tmp_path, monkeypatch):
    spec, cache = materialized(tmp_path)
    path = cache.member_path(spec.materialization, spec.member)
    real_open = Path.open
    reads = 0

    def changed_after_verification(self, *args, **kwargs):
        nonlocal reads
        if self == path and args and args[0] == "rb":
            reads += 1
            if reads == 3:  # Source verification, member verification, then parsing.
                with real_open(self, "wb") as stream:
                    stream.write(SOURCE.replace("2.50", "2.60").encode())
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", changed_after_verification)
    with pytest.raises(ValueError, match="Parsed RTPTorrent bytes"):
        convert_rtptorrent_source(spec, cache)


def test_existing_output_must_match_and_record_order_is_canonical(tmp_path):
    spec, cache = materialized(tmp_path)
    dataset = convert_rtptorrent_source(spec, cache)
    reordered = RTPTorrentDataset.model_validate(
        {**dataset.model_dump(), "executions": tuple(reversed(dataset.executions))}
    )
    assert reordered.fingerprint == dataset.fingerprint
    writer = ArtifactWriter(tmp_path / "exports")
    path = publish_rtptorrent_dataset(dataset, cache, writer)
    path.write_text("{}")
    with pytest.raises(ValueError, match="Conflicting immutable"):
        publish_rtptorrent_dataset(dataset, cache, writer)


@pytest.mark.parametrize(
    "payload",
    [
        HEADER + ROWS[0].rstrip("\n") + "\r" + ROWS[1],
        HEADER.rstrip("\n") + "\r" + ROWS[0] + ROWS[1],
        HEADER + ROWS[0].rstrip("\n") + "\r\n\r" + ROWS[1],
    ],
)
def test_internal_carriage_returns_cannot_hide_source_records(tmp_path, payload):
    spec, cache = materialized(tmp_path, payload)
    with pytest.raises(ValueError, match="Multiple CSV records"):
        convert_rtptorrent_source(spec, cache)


def test_crlf_framing_keeps_every_observation(tmp_path):
    first, cache = materialized(tmp_path / "lf")
    second, other = materialized(tmp_path / "crlf", SOURCE.replace("\n", "\r\n"))
    a = convert_rtptorrent_source(first, cache)
    b = convert_rtptorrent_source(second, other)
    assert a.executions == b.executions
    assert a.input_rows == b.input_rows == 5
    assert a.spec.materialization.spec.source.sha256 != b.spec.materialization.spec.source.sha256
