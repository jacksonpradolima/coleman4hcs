"""Optional Python source preparation; acquisition is explicit, pinned and never installs code on the host."""

from __future__ import annotations

import io
from importlib.resources import files
from typing import NamedTuple

from coleman.corpus import (
    LicenseReview,
    MaterializationReceipt,
    MaterializationSpec,
    SourceArtifact,
    SourceCache,
    load_registry,
)
from coleman.corpus.controlled_contracts import ControlledContext, ControlledInput, ControlledRuntime
from coleman.corpus.materialization import materialize_source, validate_materialization
from coleman.corpus.materialization_contracts import PackageMember
from coleman.corpus.source_views import materialize_source_view
from coleman.transports.junit_capture import _member_bytes
from coleman.transports.junit_source import digest, public_reader
from coleman.transports.producer_identity import installed_producer_identity
from coleman.transports.python_contracts import (
    PythonCaptureRecipe,
    PythonEnvironment,
    PythonHarnessPlan,
)
from coleman.transports.python_wheels import validate_python_wheel

HARNESS_NAME = "ColemanPytestCapture.py.source"


def python_harness_bytes() -> bytes:
    """Read COLEMAN-authored Python 3.8-compatible source as data, without importing or executing it."""
    return files("coleman.corpus.resources").joinpath(HARNESS_NAME).read_bytes()


def owned_python_package(cache: SourceCache, name: str, payload: bytes, version: str) -> MaterializationReceipt:
    """Retain newly authored producer metadata in the existing cache with recipe-only COLEMAN terms.

    This helper never labels foreign source, wheels or runtime components under
    COLEMAN's license. It performs local immutable cache writes, no network.
    Invalid packages or conflicting bytes raise; filesystem errors propagate.
    """
    terms = load_registry().card("ci-serial-boundary-v1").manifest.license
    terms = LicenseReview.model_validate({**terms.model_dump(), "redistribution": "recipe_only"})
    pin = digest(payload)
    spec = MaterializationSpec(
        schema_version=1,
        package_id=name,
        source=SourceArtifact(
            uri="urn:coleman:controlled-python-owned:" + name,
            version=version,
            citation="COLEMAN-authored metadata; foreign inputs retain independently reviewed terms and receipts",
            sha256=pin,
            size_bytes=len(payload),
            availability="verified_local",
        ),
        license=terms,
        archive="raw",
        members=(PackageMember(name=name, sha256=pin, size_bytes=len(payload)),),
    )
    return materialize_source(spec, cache, lambda: io.BytesIO(payload))


class PreparedPythonInputs(NamedTuple):
    """Verified receipts and frozen context; preparation neither executes a subject nor pulls its runtime."""

    context: ControlledContext
    buggy: MaterializationReceipt
    fixed: MaterializationReceipt
    fixture: MaterializationReceipt
    wheels: tuple[MaterializationReceipt, ...]
    harness: bytes


def python_harness_plan(recipe: PythonCaptureRecipe, context: str, trial_id: str, role: str) -> PythonHarnessPlan:
    """Freeze fixed container roots, sorted explicit tests and actual-version checks for one physical role.

    This pure constructor never accepts recipe shell commands. Exact serialized
    bytes plus LF are hashed by the harness and revalidated on replay.
    """
    return PythonHarnessPlan.model_validate(
        {
            "schema_version": 1,
            "context_fingerprint": context,
            "trial_id": trial_id,
            "role": role,
            "fixture_root": "/fixture",
            "production_root": "/source",
            "dependency_storage": "private-runtime-tempdir-v1",
            "wheel_root": "/wheels",
            "wheels": tuple(wheel.package.members[0].name for wheel in recipe.wheels),
            "distributions": (
                *recipe.runtime.provided_distributions,
                *(wheel.distribution for wheel in recipe.wheels),
            ),
            "test_ids": recipe.test_ids,
        }
    )


def prepare_python_inputs(
    recipe: PythonCaptureRecipe, cache: SourceCache, environment: MaterializationReceipt, *, fetch: bool = False
) -> PreparedPythonInputs:
    """Verify source views, whole wheels, metadata and installed harness, then freeze generic domain provenance.

    Default is offline. Explicit fetch uses the existing pinned public HTTPS
    adapter. Full wheels remain unmodified; their selected notices and metadata
    are validated before any container can install them. Runtime facts must be a
    typed producer record, not an arbitrary dictionary. No project setup scripts,
    wheel imports, image pulls or subject processes run. Missing/changed bytes,
    unknown schemas and incompatible plans fail closed; I/O errors propagate.
    """
    recipe = PythonCaptureRecipe.model_validate(recipe)
    if type(fetch) is not bool:
        raise ValueError("Python fetch flag must be boolean")
    validate_materialization(environment, cache)
    if len(environment.spec.members) != 1:
        raise ValueError("Python execution environment requires one typed metadata member")
    environment_bytes = _member_bytes(cache, environment, environment.spec.members[0].name)
    PythonEnvironment.from_json(environment_bytes)
    harness = python_harness_bytes()
    if digest(harness) != recipe.harness_sha256:
        raise ValueError("Installed Python harness disagrees with the frozen recipe")
    reader = public_reader if fetch else None
    views = tuple(materialize_source_view(view, cache, reader) for view in (recipe.buggy, recipe.fixed, recipe.fixture))
    wheels = []
    for wheel in recipe.wheels:
        receipt = materialize_source(wheel.package, cache, public_reader(wheel.package) if fetch else None)
        validate_python_wheel(wheel, cache.member_path(receipt, wheel.package.members[0].name))
        wheels.append(receipt)
    harness_receipt = owned_python_package(cache, HARNESS_NAME, harness, "coleman-pytest-harness-v1")
    plan = owned_python_package(cache, "producer-recipe.json", recipe.to_json().encode(), recipe.recipe)
    producer = owned_python_package(
        cache, "producer-identity.txt", (installed_producer_identity() + "\n").encode(), "coleman-package-identity-v1"
    )
    inputs = [
        ControlledInput.model_validate({"name": name, "kind": kind, "materialization": receipt})
        for name, kind, receipt in (
            ("buggy-source", "buggy_source", views[0]),
            ("fixed-source", "fixed_source", views[1]),
            ("common-fixture", "test_fixture", views[2]),
            ("observation-harness", "harness", harness_receipt),
            ("producer-recipe", "dependency", plan),
            ("producer-identity", "dependency", producer),
            ("execution-environment", "dependency", environment),
        )
    ]
    inputs.extend(
        ControlledInput(name="wheel-" + wheel.distribution.name, kind="dependency", materialization=receipt)
        for wheel, receipt in zip(recipe.wheels, wheels, strict=True)
    )
    notices = _materialize_notices(recipe, cache, fetch=fetch)
    inputs.extend(
        ControlledInput(name="notice-" + str(index), kind="dependency", materialization=receipt)
        for index, (_, receipt) in enumerate(sorted(notices.items()))
    )
    runtime = ControlledRuntime(
        implementation=recipe.runtime.implementation,
        version=recipe.runtime.version,
        platform=recipe.runtime.platform,
        execution_model="container",
        image_digest=recipe.runtime.image_digest,
        emulated=recipe.runtime.emulated,
        timezone=recipe.runtime.timezone,
        setup=(
            "Unmodified common fixed-tree fixture and separately pinned buggy/fixed production trees",
            "Pure wheels installed locally with --no-index --no-deps --no-compile; no project setup.py/build",
            "Dependency storage: private-runtime-tempdir-v1; atomic mode0700 allocation in bounded container tmpfs",
            "One interpreter per role; explicit lexical test order; order independence is not established",
            "Per-case monotonic integer ns include setup/call/teardown and are not removable CI runner work",
            "Collection clocks are actual UTC microseconds, never historical CI event or feature clocks",
            "LC_ALL=" + recipe.runtime.locale + "; LANG=" + recipe.runtime.locale,
            "PYTHONUTF8=0; PYTHONCOERCECLOCALE=0; PYTHONIOENCODING=utf-8; PYTHONDONTWRITEBYTECODE=1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; PYTHONHASHSEED=0; HOME=/tmp; no inherited host environment",
            "Read-only non-root container: network=none; CPUs=2; PIDs=64; tmpfs_bytes=134217728; memory_bytes="
            + str(recipe.limits.memory_bytes),
            f"Parent guards: timeout_seconds={recipe.limits.timeout_seconds}; "
            f"output_bytes={recipe.limits.maximum_output_bytes}",
            "Producer identity conservatively hashes all COLEMAN packaged files/version, excluding caches/bytecode",
        ),
        commands=("python -B -s /harness/ColemanPytestCapture.py /harness/plan.json",),
        interventions=(
            "Use the same declared fixed-tree test fixture against both source revisions",
            "Use a pinned Python-3.8-compatible pytest/wheel environment instead of the unpinned original bootstrap",
        ),
    )
    context = ControlledContext(
        schema_version=1,
        project=recipe.project,
        fault_id=recipe.fault_id,
        fault_origin=recipe.fault_origin,
        language="Python",
        buggy_revision=recipe.buggy.output.source.version,
        fixed_revision=recipe.fixed.output.source.version,
        fixture_revision=recipe.fixture.output.source.version,
        dependence_group=recipe.dependence_group,
        test_ids=recipe.test_ids,
        inputs=tuple(inputs),
        runtime=runtime,
        limitations=recipe.limitations,
    )
    return PreparedPythonInputs(context, views[0], views[1], views[2], tuple(wheels), harness)


def _materialize_notices(
    recipe: PythonCaptureRecipe, cache: SourceCache, *, fetch: bool
) -> dict[str, MaterializationReceipt]:
    """Retain each already validated notice once through the existing cache; no source code is executed."""
    notices = {}
    for package in recipe.component_notices():
        if package.package_id not in notices:
            notices[package.package_id] = materialize_source(package, cache, public_reader(package) if fetch else None)
    return notices
