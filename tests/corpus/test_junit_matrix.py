"""Owned matrix contracts and real cache/capture publication; only external Docker is substituted."""

import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from coleman.artifacts import ArtifactWriter
from coleman.corpus.archives import deterministic_zip_bytes
from coleman.corpus.controlled import convert_controlled_source, publish_controlled_dataset, validate_controlled_dataset
from coleman.corpus.controlled_contracts import ControlledDataset, ControlledReport
from coleman.corpus.materialization import materialize_source
from coleman.corpus.materialization_contracts import PackageMember
from coleman.transports import junit_capture as capture
from coleman.transports import junit_matrix as matrix
from coleman.transports.junit_contracts import ProducerElapsed
from coleman.transports.junit_matrix_contracts import (
    JunitMatrixCell,
    JunitMatrixCellResult,
    JunitMatrixReceipt,
    JunitMatrixRecipe,
    JunitMatrixRole,
    JunitPairCounts,
)
from coleman.transports.junit_process import ProcessStreams
from tests.corpus.test_controlled import digest
from tests.corpus.test_junit_capture import prepared_inputs, substitute_docker
from tests.corpus.test_junit_protocol import changed, encoded, recipe, records


def owned_plan(spec=None, trials=("owned-trial",)):
    """Declare schema-only owned inputs; no foreign Java code or image is present."""
    return JunitMatrixRecipe(
        schema_version=1,
        matrix_id="owned-java-matrix",
        cells=tuple(
            JunitMatrixCell(
                cell_id="owned-" + str(index),
                trial_id=trial,
                source_reference="urn:coleman:owned:source-fix",
                fixture_selection="Owned schema-only common fixture, not executable Java",
                recipe=spec or recipe(),
            )
            for index, trial in enumerate(trials)
        ),
        limitations=("Owned schema-only evidence; no real fault or independence claim",),
    )


def owned_capture(tmp_path, monkeypatch, trials=("owned-trial",)):
    """Prepare actual source-cache contracts and substitute only external Docker process calls."""
    spec, cache, environment = prepared_inputs(tmp_path)
    calls = []

    def run(*args):
        role = args[-2]
        calls.append(role)
        rows = records()
        rows[1]["failures"] = int(role == "buggy")
        return ProcessStreams(encoded(rows), b"owned diagnostic", 0, 0, "exited")

    substitute_docker(monkeypatch, environment, run)
    return owned_plan(spec, trials), cache, ArtifactWriter(tmp_path / "results"), calls


def owned_receipt():
    """Construct stable schema-only index metadata; pins do not claim real datasets or execution."""
    plan = owned_plan()
    roles = tuple(
        JunitMatrixRole(
            role=role,
            packet_sha256="sha256:" + digit * 64,
            status="collected",
            producer_elapsed=ProducerElapsed(value=0, status="observed"),
        )
        for role, digit in (("buggy", "2"), ("fixed", "3"))
    )
    result = JunitMatrixCellResult(
        cell_id=plan.cells[0].cell_id,
        dataset_fingerprint="sha256:" + "0" * 64,
        context_fingerprint="sha256:" + "1" * 64,
        planned_pairs=1,
        unobserved_pairs=0,
        pairs=JunitPairCounts(expected=1),
        roles=roles,
    )
    return JunitMatrixReceipt(
        schema_version=1,
        plan=plan,
        producer_sha256="sha256:" + "4" * 64,
        cells=(result,),
        status="collected",
    )


def test_plan_roundtrip_golden_and_declaration_order_metamorphism():
    plan = owned_plan(trials=("trial-a", "trial-b"))
    assert JunitMatrixRecipe.from_json(plan.to_json()) == plan
    assert changed(plan, cells=tuple(reversed(plan.cells))).fingerprint == plan.fingerprint
    fixture = Path(__file__).with_name("fixtures") / "junit-matrix-owned-v1.json"
    assert plan.to_json() + "\n" == fixture.read_text()
    # Existing schema-1 JUnit recipes remain byte-identical, without a matrix wrapper.
    old = Path(__file__).with_name("fixtures") / "junit-recipe-owned-v1.json"
    assert recipe().to_json() + "\n" == old.read_text()


def test_receipt_roundtrip_versioned_golden_and_strict_accounting():
    receipt = owned_receipt()
    fixture = Path(__file__).with_name("fixtures") / "junit-matrix-receipt-owned-v1.json"
    assert receipt.to_json() + "\n" == fixture.read_text()
    assert JunitMatrixReceipt.from_json(fixture.read_bytes()) == receipt
    with pytest.raises(ValueError, match="collection status"):
        changed(receipt, status="incomplete")
    with pytest.raises(ValueError, match="conservation"):
        changed(receipt.cells[0], unobserved_pairs=1)
    with pytest.raises(ValueError, match="distinct source roles"):
        changed(receipt.cells[0], roles=(receipt.cells[0].roles[0], receipt.cells[0].roles[0]))
    unknown_cell = changed(receipt.cells[0], cell_id="unknown")
    with pytest.raises(ValueError, match="cover planned"):
        changed(receipt, cells=(unknown_cell,))
    with pytest.raises(ValueError, match="Duplicate Java matrix result"):
        changed(receipt, cells=(receipt.cells[0], receipt.cells[0]))


@pytest.mark.parametrize("location", ["receipt", "plan", "source_recipe"])
def test_unknown_versions_in_receipt_or_nested_contracts_reject(location):
    data = json.loads(owned_receipt().to_json())
    target = data if location == "receipt" else data["plan"]
    if location == "source_recipe":
        target = target["cells"][0]["recipe"]
    target["schema_version"] = 2
    payload = json.dumps(data)
    with pytest.raises(ValueError):
        JunitMatrixReceipt.from_json(payload)


@pytest.mark.parametrize(
    "update",
    [
        {"unit": "ms"},
        {"value": None},
        {"value": -1},
        {"value": float("inf")},
        {"value": float("nan")},
        {"status": "unavailable"},
    ],
)
def test_index_process_units_missingness_and_nonfinite_counters_reject(update):
    data = owned_receipt().model_dump(mode="json")
    data["cells"][0]["roles"][0]["producer_elapsed"].update(update)
    with pytest.raises(ValueError):
        JunitMatrixReceipt.model_validate(data)


@pytest.mark.parametrize("schema", [None, 0, 2, 99, True, 1.5])
def test_plan_unknown_or_missing_schema_rejects(schema):
    data = json.loads(owned_plan().to_json())
    if schema is None:
        del data["schema_version"]
    else:
        data["schema_version"] = schema
    payload = json.dumps(data)
    with pytest.raises(ValueError):
        JunitMatrixRecipe.from_json(payload)


def test_duplicate_ids_and_recipe_trial_collections_reject():
    plan = owned_plan(trials=("a", "b"))
    duplicate_id = changed(plan.cells[1], cell_id=plan.cells[0].cell_id)
    with pytest.raises(ValueError, match="cell identity"):
        changed(plan, cells=(plan.cells[0], duplicate_id))
    duplicate_trial = changed(plan.cells[1], trial_id=plan.cells[0].trial_id)
    with pytest.raises(ValueError, match="recipe/trial"):
        changed(plan, cells=(plan.cells[0], duplicate_trial))


@pytest.mark.parametrize("value", [-1, None, True, float("inf"), float("nan")])
def test_pair_counts_reject_negative_missing_and_nonfinite_values(value):
    with pytest.raises(ValueError):
        JunitPairCounts(expected=value)


def test_capture_roundtrip_raw_correspondence_replay_and_cli(tmp_path, monkeypatch, capsys):
    plan, cache, writer, calls = owned_capture(tmp_path, monkeypatch, ("a", "b"))
    first = matrix.capture_junit_matrix(plan, cache, writer)
    second = matrix.capture_junit_matrix(plan, cache, writer)
    assert first == second
    assert calls == ["buggy", "fixed", "buggy", "fixed"]
    receipt = JunitMatrixReceipt.from_json(first.receipt_path.read_bytes())
    assert receipt == first.receipt
    assert first.completion_path is not None
    assert first.completion_path.read_bytes() == first.receipt_path.read_bytes()
    assert receipt.status == "collected"
    assert receipt.declared_projects == 1
    assert receipt.curated_focal_fixes == 1
    assert receipt.admission == "not_evaluated"
    assert receipt.generalization == "insufficient_evidence"
    assert all(cell.pairs.expected == 1 and cell.unobserved_pairs == 0 for cell in receipt.cells)
    assert all(role.producer_elapsed.value == 0 for cell in receipt.cells for role in cell.roles)
    matrix.validate_junit_matrix(receipt, cache, writer.run_root)
    reversed_cells = tuple(changed(cell, roles=tuple(reversed(cell.roles))) for cell in reversed(receipt.cells))
    assert changed(receipt, cells=reversed_cells).fingerprint == receipt.fingerprint
    path = tmp_path / "plan.json"
    path.write_text(plan.to_json())
    assert matrix.main(["--recipe", str(path), "--cache", str(cache.root), "--output", str(writer.run_root)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["curated_focal_fixes"] == 1
    assert summary["declared_projects"] == 1
    monkeypatch.setattr("sys.stdin", SimpleNamespace(buffer=io.BytesIO(first.receipt_path.read_bytes())))
    assert matrix.main(["--validate", "--cache", str(cache.root), "--output", str(writer.run_root)]) == 0
    assert json.loads(capsys.readouterr().out)["integrity"] == "verified"
    assert len(calls) == 4


def test_interrupted_roles_remain_unobserved_without_false_completion(tmp_path, monkeypatch):
    plan, cache, writer, calls = owned_capture(tmp_path, monkeypatch)

    def interrupted(*args):
        calls.append(args[-2])
        raise KeyboardInterrupt

    monkeypatch.setattr(capture, "_run_role", interrupted)
    for _ in range(2):
        with pytest.raises(KeyboardInterrupt):
            matrix.capture_junit_matrix(plan, cache, writer)
        assert not list(writer.run_root.rglob("complete.json"))
        plans = list(writer.run_root.rglob("plan.json"))
        assert len(plans) == 1
        assert plans[0].read_text() == plan.to_json() + "\n"
    outcome = matrix.capture_junit_matrix(plan, cache, writer)
    assert calls == ["buggy", "fixed"]
    assert outcome.completion_path is None
    assert outcome.receipt.status == "incomplete"
    cell = outcome.receipt.cells[0]
    assert cell.unobserved_pairs == 1
    assert cell.pairs.total == 0
    assert all(role.status == "interrupted" for role in cell.roles)
    assert all(role.producer_elapsed.value is None for role in cell.roles)
    assert all(role.producer_elapsed.status == "unavailable" for role in cell.roles)
    assert matrix.capture_junit_matrix(plan, cache, writer) == outcome
    assert not list(writer.run_root.rglob("complete.json"))


def test_completion_failure_recovers_existing_captures_without_rerun(tmp_path, monkeypatch):
    plan, cache, writer, calls = owned_capture(tmp_path, monkeypatch)
    original = writer.write_immutable

    def interrupted(path, content):
        if path.name == "complete.json":
            raise OSError("owned final publication interruption")
        original(path, content)

    monkeypatch.setattr(writer, "write_immutable", interrupted)
    with pytest.raises(OSError, match="publication interruption"):
        matrix.capture_junit_matrix(plan, cache, writer)
    assert not list(writer.run_root.rglob("complete.json"))
    monkeypatch.setattr(writer, "write_immutable", original)
    recovered = matrix.capture_junit_matrix(plan, cache, writer)
    assert recovered.completion_path is not None
    assert recovered.completion_path.exists()
    assert calls == ["buggy", "fixed"]


def test_concurrent_contenders_preserve_existing_lock_rejection_and_replay(tmp_path, monkeypatch):
    plan, cache, writer, calls = owned_capture(tmp_path, monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(matrix.capture_junit_matrix, plan, cache, writer) for _ in range(2)]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except ValueError as error:
                assert "Another producer owns this context/trial" in str(error)
    assert outcomes
    replay = matrix.capture_junit_matrix(plan, cache, writer)
    assert all(outcome == replay for outcome in outcomes)
    assert calls == ["buggy", "fixed"]
    assert len(list(writer.run_root.rglob("complete.json"))) == 1


def test_forged_index_counts_and_unknown_producer_reject(tmp_path, monkeypatch):
    plan, cache, writer, _ = owned_capture(tmp_path, monkeypatch)
    receipt = matrix.capture_junit_matrix(plan, cache, writer).receipt
    altered_cell = changed(receipt.cells[0], pairs=JunitPairCounts(both_pass=1))
    forged = changed(receipt, cells=(altered_cell,))
    with pytest.raises(ValueError, match="actual source evidence"):
        matrix.validate_junit_matrix(forged, cache, writer.run_root)
    unknown = changed(receipt, producer_sha256="sha256:" + "f" * 64)
    with pytest.raises(ValueError, match="Unknown Java matrix producer"):
        matrix.validate_junit_matrix(unknown, cache, writer.run_root)


def test_nested_git_and_escaped_generation_reject_before_source_capture(tmp_path, monkeypatch):
    plan, cache, writer, calls = owned_capture(tmp_path, monkeypatch)
    base = writer.run_root / "corpus-junit-matrix"
    base.mkdir(parents=True)
    (base / ".git").mkdir()
    with pytest.raises(ValueError, match="outside a repository"):
        matrix.capture_junit_matrix(plan, cache, writer)
    assert not calls
    (base / ".git").rmdir()
    base.rmdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    base.symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="within its output tree"):
        matrix.capture_junit_matrix(plan, cache, writer)
    assert not calls


def test_metadata_bounds_fetch_type_and_validation_are_preflighted(tmp_path, monkeypatch):
    plan, cache, writer, calls = owned_capture(tmp_path, monkeypatch)
    invalid_fetch = json.loads("1")
    with pytest.raises(ValueError, match="boolean"):
        matrix.capture_junit_matrix(plan, cache, writer, fetch=invalid_fetch)
    monkeypatch.setattr(JunitMatrixRecipe, "MAXIMUM_BYTES", 1)
    with pytest.raises(ValueError, match="byte limit"):
        matrix.capture_junit_matrix(plan, cache, writer)
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"{}")
    with pytest.raises(ValueError, match="byte limit"):
        matrix.load_junit_matrix(oversized)
    assert not calls


def test_published_prior_recipe_and_new_frozen_presets_bind_actual_harness():
    plan = matrix.load_junit_matrix()
    assert len(plan.cells) == 3
    assert {cell.recipe.project for cell in plan.cells} == {"Apache Commons CLI", "Apache Commons Codec"}
    assert sorted(len(cell.recipe.test_ids) for cell in plan.cells) == [95, 96, 166]
    codec = next(cell.recipe for cell in plan.cells if cell.recipe.fault_id == "CODEC-65")
    assert len(codec.buggy.inputs) == len(codec.fixed.inputs) == 27
    assert len(codec.fixture.inputs) == 17
    names = {member.name for member in codec.fixture.output.members}
    excluded = {"DoubleMetaphoneTest.java", "SoundexTest.java", "AllTests.java"}
    assert not any(Path(name).name in excluded for name in names)
    assert {"LICENSE.txt", "NOTICE.txt"}.issubset(names)


def test_uncontrolled_upstream_fixture_inputs_are_disclosed_without_dropping_tests():
    plan = matrix.load_junit_matrix()
    codec = next(cell.recipe for cell in plan.cells if cell.recipe.fault_id == "CODEC-65")
    names = {Path(member.name).name for member in codec.fixture.output.members}
    assert {"Base64Test.java", "HexTest.java", "DigestUtilsTest.java"}.issubset(names)
    for classname in (
        "org.apache.commons.codec.binary.Base64Test",
        "org.apache.commons.codec.binary.HexTest",
        "org.apache.commons.codec.digest.DigestUtilsTest",
    ):
        assert any(test.startswith(classname + "::") for test in codec.test_ids)
    assert len(codec.test_ids) == 166
    assert any("unseeded java.util.Random" in text and "not captured" in text for text in codec.limitations)
    assert any("randomness is uncontrolled" in text for text in plan.limitations)


@pytest.mark.parametrize(
    "status,elapsed",
    [
        ("interrupted", ProducerElapsed(value=0, status="observed")),
        ("collected", ProducerElapsed(value=None, status="unavailable")),
        ("timeout", ProducerElapsed(value=None, status="unavailable")),
        ("output_limit", ProducerElapsed(value=None, status="unavailable")),
        ("protocol_error", ProducerElapsed(value=None, status="unavailable")),
    ],
)
def test_role_status_cannot_invent_or_discard_elapsed_observations(status, elapsed):
    with pytest.raises(ValueError, match="elapsed"):
        JunitMatrixRole(role="buggy", packet_sha256="sha256:" + "a" * 64, status=status, producer_elapsed=elapsed)


def test_completed_matrix_replay_is_offline_without_runtime_preflight(tmp_path, monkeypatch):
    plan, cache, writer, calls = owned_capture(tmp_path, monkeypatch)
    first = matrix.capture_junit_matrix(plan, cache, writer)

    def unavailable(*args, **kwargs):
        raise AssertionError("Completed matrix replay must not start runtime preflight or source capture")

    monkeypatch.setattr(matrix, "capture_junit_source", unavailable)
    assert matrix.capture_junit_matrix(plan, cache, writer) == first
    assert calls == ["buggy", "fixed"]


def test_corrupt_completion_rejects_before_source_capture(tmp_path, monkeypatch):
    plan, cache, writer, calls = owned_capture(tmp_path, monkeypatch)
    first = matrix.capture_junit_matrix(plan, cache, writer)
    assert first.completion_path is not None
    first.completion_path.write_text('{"schema_version":99}')
    with pytest.raises(ValueError):
        matrix.capture_junit_matrix(plan, cache, writer)
    assert calls == ["buggy", "fixed"]


@pytest.mark.parametrize(
    "buggy,fixed,classification",
    [
        ((0, 0), (0, 0), "both_pass"),
        ((1, 0), (1, 0), "both_fail"),
        ((0, 0), (1, 0), "reversed"),
        ((0, 1), (0, 0), "inconclusive"),
    ],
)
def test_negative_reversed_and_error_results_remain_evidence(tmp_path, monkeypatch, buggy, fixed, classification):
    plan, cache, writer, _ = owned_capture(tmp_path, monkeypatch)

    def observed(*args):
        rows = records()
        rows[1]["failures"], rows[1]["errors"] = buggy if args[-2] == "buggy" else fixed
        return ProcessStreams(encoded(rows), b"owned compiler diagnostic", 0, 0, "exited")

    monkeypatch.setattr(capture, "_run_role", observed)
    outcome = matrix.capture_junit_matrix(plan, cache, writer)
    assert outcome.receipt.status == "collected"
    assert getattr(outcome.receipt.cells[0].pairs, classification) == 1
    assert outcome.receipt.cells[0].pairs.total == 1
    assert outcome.receipt.admission == "not_evaluated"


def repinned_owned_forgery(dataset, cache, kind):
    """Create coherent generic source envelopes that still violate the Java capture protocol.

    Recomputing all physical digests must not turn a rewritten normalized outcome,
    context, missing packet or raw-case annotation into accepted Java evidence.
    """
    package = dataset.spec.report_materialization
    entries = {member.name: cache.member_path(package, member.name).read_bytes() for member in package.spec.members}
    report = ControlledReport.from_json(entries["report.json"])
    context = dataset.spec.context
    observations = list(report.observations)
    if kind == "normalized_outcome":
        observations = [changed(item, outcome="passed") if item.role == "buggy" else item for item in observations]
    elif kind == "context":
        context = changed(context, project="rewritten-owned-project")
    elif kind == "missing_packet":
        del entries["fixed-capture.json"]
        observations = [item for item in observations if item.role != "fixed"]
    else:
        raw = json.loads(entries["buggy-capture.json"])
        if kind == "packet_identity":
            raw["trial_id"] = "different-trial"
        else:
            raw["cases"][0]["elapsed_ns"] = 1
        entries["buggy-capture.json"] = json.dumps(raw, sort_keys=True).encode()
        pin = digest(entries["buggy-capture.json"])
        observations = [
            changed(item, diagnostics_sha256=pin) if item.role == "buggy" else item for item in observations
        ]
    entries["report.json"] = (
        changed(report, context_fingerprint=context.fingerprint, observations=tuple(observations)).to_json().encode()
    )
    payload = deterministic_zip_bytes(
        tuple(sorted(entries.items())), maximum_member_bytes=16 * 1024**2, maximum_total_bytes=40 * 1024**2
    )
    spec = changed(
        package.spec,
        package_id="owned-repinned-forgery",
        source=changed(
            package.spec.source,
            uri="urn:coleman:owned:repinned-forgery",
            sha256=digest(payload),
            size_bytes=len(payload),
        ),
        members=tuple(
            PackageMember(name=name, sha256=digest(data), size_bytes=len(data))
            for name, data in sorted(entries.items())
        ),
    )
    receipt = materialize_source(spec, cache, lambda: io.BytesIO(payload))
    return convert_controlled_source(changed(dataset.spec, context=context, report_materialization=receipt), cache)


@pytest.mark.parametrize(
    "kind,problem",
    [
        ("normalized_outcome", "normalized observations"),
        ("context", "context disagrees"),
        ("missing_packet", "raw role packet"),
        ("packet_identity", "raw role identity"),
        ("raw_annotation", "disagree with raw streams"),
    ],
)
def test_repinning_all_digests_does_not_certify_forged_java_evidence(tmp_path, monkeypatch, kind, problem):
    plan, cache, writer, _ = owned_capture(tmp_path, monkeypatch)
    outcome = matrix.capture_junit_matrix(plan, cache, writer)
    reference = outcome.receipt.cells[0]
    path = writer.run_root / "corpus-controlled" / (reference.dataset_fingerprint.removeprefix("sha256:") + ".json")
    dataset = ControlledDataset.from_json(path.read_bytes())
    forged = repinned_owned_forgery(dataset, cache, kind)
    validate_controlled_dataset(forged, cache)
    publish_controlled_dataset(forged, cache, writer)
    index = changed(outcome.receipt, cells=(changed(reference, dataset_fingerprint=forged.fingerprint),))
    with pytest.raises(ValueError, match=problem):
        matrix.validate_junit_matrix(index, cache, writer.run_root)


def test_validation_enforces_plan_receipt_and_dataset_budgets(tmp_path, monkeypatch):
    plan, cache, writer, _ = owned_capture(tmp_path, monkeypatch)
    receipt = matrix.capture_junit_matrix(plan, cache, writer).receipt
    with monkeypatch.context() as limits:
        limits.setattr(JunitMatrixRecipe, "MAXIMUM_BYTES", 1)
        with pytest.raises(ValueError, match="plan exceeds"):
            matrix.validate_junit_matrix(receipt, cache, writer.run_root)
    with monkeypatch.context() as limits:
        limits.setattr(JunitMatrixReceipt, "MAXIMUM_BYTES", 1)
        with pytest.raises(ValueError, match="receipt exceeds"):
            matrix.validate_junit_matrix(receipt, cache, writer.run_root)
    monkeypatch.setattr(matrix, "_DATASET_BYTES", 1)
    with pytest.raises(ValueError, match="metadata exceeds"):
        matrix.validate_junit_matrix(receipt, cache, writer.run_root)


@pytest.mark.parametrize(
    "package,module",
    [
        ("coleman.transports", "junit_protocol.py"),
        ("coleman.corpus", "source_views.py"),
        ("coleman.corpus", "registry.py"),
        ("coleman.corpus", "contracts.py"),
        ("coleman.corpus", "resources/registry-v1.json"),
        ("coleman.corpus", "zip_metadata.py"),
        ("coleman.evaluation.protected", "temporal.py"),
    ],
)
def test_changing_reused_producer_dependency_changes_matrix_generation(tmp_path, monkeypatch, package, module):
    before = matrix.junit_matrix_producer()
    original = matrix.files
    target = package.replace(".", "/") + "/" + module

    def altered_resource(resource, path):
        def read_bytes():
            payload = resource.read_bytes()
            if path == target:
                payload += b"\n# Owned provenance metamorphism\n"
            return payload

        return SimpleNamespace(
            name=resource.name,
            is_dir=resource.is_dir,
            is_file=resource.is_file,
            read_bytes=read_bytes,
            iterdir=lambda: (altered_resource(child, path + "/" + child.name) for child in resource.iterdir()),
        )

    monkeypatch.setattr(matrix, "files", lambda namespace: altered_resource(original(namespace), namespace))
    after = matrix.junit_matrix_producer()
    assert after != before
    assert matrix._generation(owned_plan(), tmp_path, after) != matrix._generation(owned_plan(), tmp_path, before)


def test_new_package_resources_change_identity_but_installation_caches_do_not(tmp_path, monkeypatch):
    (tmp_path / "owned.py").write_bytes(b"# Owned package fixture\n")
    monkeypatch.setattr(matrix, "files", lambda namespace: tmp_path)
    before = matrix.junit_matrix_producer()
    resources = tmp_path / "resources"
    resources.mkdir()
    (resources / "future-format.bin").write_bytes(b"owned future resource")
    after = matrix.junit_matrix_producer()
    assert after != before
    (tmp_path / "owned.pyc").write_bytes(b"incidental bytecode")
    (tmp_path / "owned.pyo").write_bytes(b"incidental optimized bytecode")
    (tmp_path / ".installation-state").write_bytes(b"incidental hidden file")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "owned.cpython.pyc").write_bytes(b"incidental interpreter cache")
    assert matrix.junit_matrix_producer() == after


@pytest.mark.parametrize("alias", ["parent", "symlink"])
def test_cli_validation_rejects_receipt_path_arguments_without_opening_files(tmp_path, monkeypatch, alias):
    output = tmp_path / "output"
    output.mkdir()
    external = tmp_path / "external.json"
    external.write_text(owned_receipt().to_json())
    candidate = output / ".." / external.name
    if alias == "symlink":
        candidate = output / "receipt.json"
        candidate.symlink_to(external)
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path.resolve() == external.resolve():
            raise AssertionError("Out-of-output artifact was read before containment validation")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    arguments = [
        "--validate",
        str(candidate),
        "--cache",
        str(tmp_path / "cache"),
        "--output",
        str(output),
    ]
    with pytest.raises(SystemExit) as error:
        matrix.main(arguments)
    assert error.value.code == 2


def test_cli_validation_bounds_stdin_before_parsing(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin", SimpleNamespace(buffer=io.BytesIO(b"{}")))
    monkeypatch.setattr(JunitMatrixReceipt, "MAXIMUM_BYTES", 1)
    arguments = ["--validate", "--cache", str(tmp_path / "cache"), "--output", str(tmp_path / "output")]
    with pytest.raises(ValueError, match="stdin exceeds"):
        matrix.main(arguments)
