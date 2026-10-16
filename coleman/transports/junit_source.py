"""Optional JUnit producer input preparation; source acquisition stays explicit and verified."""

from __future__ import annotations

import hashlib
import io
from functools import partial
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
from coleman.corpus.materialization import ReaderFactory, materialize_source, validate_materialization
from coleman.corpus.materialization_contracts import PackageMember
from coleman.corpus.source_views import materialize_source_view
from coleman.transports.corpus_http import public_https_source
from coleman.transports.junit_contracts import JunitCaptureRecipe

HARNESS_NAME = "ColemanJunit3Capture.java"


def digest(payload: bytes) -> str:
    """Return the explicit SHA-256 identity of actual bytes, without file/network access."""
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def harness_bytes() -> bytes:
    """Load the packaged COLEMAN-authored source harness; no subject/JVM code is executed."""
    return files("coleman.corpus.resources").joinpath(HARNESS_NAME).read_bytes()


def owned_package(cache: SourceCache, name: str, payload: bytes, version: str) -> MaterializationReceipt:
    """Cache new producer-owned bytes with explicit recipe-only terms, not a foreign-code blanket license.

    Foreign subject and dependency inputs retain their independent reviewed terms.
    This helper only describes newly authored harness/recipe/code/evidence records.
    It uses the existing source cache and never publishes the data to Git/network.
    """
    terms = load_registry().card("ci-serial-boundary-v1").manifest.license
    terms = LicenseReview.model_validate({**terms.model_dump(), "redistribution": "recipe_only"})
    pin = digest(payload)
    spec = MaterializationSpec(
        schema_version=1,
        package_id=name,
        source=SourceArtifact(
            uri="urn:coleman:controlled-junit-owned:" + name,
            version=version,
            citation="COLEMAN-authored producer record; foreign inputs have separate source/terms receipts",
            sha256=pin,
            size_bytes=len(payload),
            availability="verified_local",
        ),
        license=terms,
        archive="raw",
        members=(PackageMember(name=name, sha256=pin, size_bytes=len(payload)),),
    )
    return materialize_source(spec, cache, lambda: io.BytesIO(payload))


def public_reader(package: MaterializationSpec) -> ReaderFactory:
    """Return an unopened explicit public reader; host/size/TLS/redirect checks remain in the existing transport."""
    if package.source.size_bytes is None:
        raise ValueError("JUnit acquisition requires a pinned source byte count")
    return partial(public_https_source, package.source.uri, package.source.size_bytes)


def classpath(count: int) -> str:
    """Return only producer-generated numeric dependency paths; no user text becomes shell syntax."""
    if not 1 <= count <= 16:
        raise ValueError("JUnit classpath requires 1..16 dependencies")
    return ":".join(f"/deps/{index}.jar" for index in range(count))


def java_commands(count: int) -> tuple[str, str]:
    """Return the exact fixed in-container compile/observe commands; subject paths live in argument files."""
    jars = classpath(count)
    compile_command = (
        "javac -encoding UTF-8 -cp " + jars + " -d /tmp/classes @/harness/production-sources.txt "
        "@/harness/fixture-sources.txt /harness/" + HARNESS_NAME
    )
    observe_command = "java -Dfile.encoding=UTF-8 -cp /tmp/classes:" + jars + " ColemanJunit3Capture"
    return compile_command, observe_command


class PreparedJunitInputs(NamedTuple):
    """Verified source-view/dependency receipts and frozen context; no process is started here."""

    context: ControlledContext
    buggy: MaterializationReceipt
    fixed: MaterializationReceipt
    fixture: MaterializationReceipt
    dependencies: tuple[MaterializationReceipt, ...]
    harness: bytes


def prepare_junit_inputs(
    recipe: JunitCaptureRecipe, cache: SourceCache, environment: MaterializationReceipt, *, fetch: bool = False
) -> PreparedJunitInputs:
    """Materialize a frozen source recipe offline by default and bind actual producer implementation bytes.

    Explicit fetch uses only the reviewed public transport. Source views preserve
    all raw bytes and their homogeneous terms. No image is pulled or code executed.
    Changed harness pins, invalid inputs and missing offline bytes fail closed;
    I/O failures propagate. Recipe/controller/harness receipts retain provenance.
    """
    recipe = JunitCaptureRecipe.model_validate(recipe)
    if type(fetch) is not bool:
        raise ValueError("JUnit fetch flag must be boolean")
    validate_materialization(environment, cache)
    harness = harness_bytes()
    if digest(harness) != recipe.harness_sha256:
        raise ValueError("Installed JUnit harness disagrees with frozen recipe; a reviewed recipe update is required")
    reader = public_reader if fetch else None
    views = tuple(materialize_source_view(view, cache, reader) for view in (recipe.buggy, recipe.fixed, recipe.fixture))
    dependencies = tuple(
        materialize_source(package, cache, public_reader(package) if fetch else None) for package in recipe.dependencies
    )
    harness_receipt = owned_package(cache, HARNESS_NAME, harness, "coleman-junit3-harness-v1")
    plan = owned_package(cache, "producer-recipe.json", recipe.to_json().encode(), recipe.recipe)
    inputs = [
        ControlledInput.model_validate({"name": name, "kind": kind, "materialization": receipt})
        for name, kind, receipt in (
            ("buggy-source", "buggy_source", views[0]),
            ("fixed-source", "fixed_source", views[1]),
            ("common-fixture", "test_fixture", views[2]),
            ("observation-harness", "harness", harness_receipt),
            ("producer-recipe", "dependency", plan),
            ("execution-environment", "dependency", environment),
        )
    ]
    inputs.extend(
        ControlledInput(name=f"classpath-{index}", kind="dependency", materialization=receipt)
        for index, receipt in enumerate(dependencies)
    )
    for module in (
        "junit_contracts.py",
        "junit_process.py",
        "junit_source.py",
        "junit_protocol.py",
        "junit_capture.py",
    ):
        payload = files("coleman.transports").joinpath(module).read_bytes()
        receipt = owned_package(cache, module, payload, "coleman-junit3-producer-code-v1")
        inputs.append(ControlledInput(name=module, kind="dependency", materialization=receipt))
    view_code = files("coleman.corpus").joinpath("source_views.py").read_bytes()
    inputs.append(
        ControlledInput(
            name="source-view-code",
            kind="dependency",
            materialization=owned_package(cache, "source_views.py", view_code, "coleman-source-view-code-v1"),
        )
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
            "Pinned source views and a common fixed-revision test fixture; no original build bootstrap",
            "Shared JVM per role; lexical test order; order independence is not established",
            "Per-case elapsed ticks include JUnit setup/teardown; not removable CI runner work",
            "Producer UTC capture instants explicitly use microsecond resolution; monotonic counters retain integer ns",
            f"Container: memory_bytes={recipe.limits.memory_bytes}; "
            "CPUs=2; PIDs=64; tmpfs_bytes=134217728; network=none",
            f"Parent guards: timeout_seconds={recipe.limits.timeout_seconds}; "
            f"output_bytes={recipe.limits.maximum_output_bytes}",
        ),
        commands=java_commands(len(dependencies)),
        interventions=("Use the same declared fixed-tree test fixture against both source revisions",),
    )
    context = ControlledContext(
        schema_version=1,
        project=recipe.project,
        fault_id=recipe.fault_id,
        fault_origin=recipe.fault_origin,
        language="Java",
        buggy_revision=recipe.buggy.output.source.version,
        fixed_revision=recipe.fixed.output.source.version,
        fixture_revision=recipe.fixture.output.source.version,
        dependence_group=recipe.dependence_group,
        test_ids=recipe.test_ids,
        inputs=tuple(inputs),
        runtime=runtime,
        limitations=recipe.limitations,
    )
    return PreparedJunitInputs(context, views[0], views[1], views[2], dependencies, harness)
