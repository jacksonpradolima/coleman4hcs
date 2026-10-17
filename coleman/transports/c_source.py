"""Pinned C source preparation using existing raw views, cache and generic controlled contexts."""

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
from coleman.transports.c_contracts import CCaptureRecipe, CEnvironment, CHarnessPlan
from coleman.transports.junit_capture import _member_bytes
from coleman.transports.junit_source import digest, public_reader
from coleman.transports.producer_identity import installed_producer_identity

HARNESS_NAME = "ColemanCCapture.py.source"


def c_harness_bytes() -> bytes:
    """Read authored Python 3.13-compatible C observer source as data, without executing it."""
    return files("coleman.corpus.resources").joinpath(HARNESS_NAME).read_bytes()


def c_collector_bytes() -> bytes:
    """Reuse the exact standalone bounded process collector; no copied foreign runner is introduced."""
    return files("coleman.transports").joinpath("junit_process.py").read_bytes()


def owned_c_package(cache: SourceCache, name: str, payload: bytes, version: str) -> MaterializationReceipt:
    """Store authored producer metadata under COLEMAN recipe-only terms, never relabeling foreign inputs."""
    terms = load_registry().card("ci-serial-boundary-v1").manifest.license
    terms = LicenseReview.model_validate({**terms.model_dump(), "redistribution": "recipe_only"})
    pin = digest(payload)
    spec = MaterializationSpec(
        schema_version=1,
        package_id=name,
        source=SourceArtifact(
            uri="urn:coleman:controlled-c-owned:" + name,
            version=version,
            citation="COLEMAN-authored observer metadata; foreign inputs retain independent terms",
            sha256=pin,
            size_bytes=len(payload),
            availability="verified_local",
        ),
        license=terms,
        archive="raw",
        members=(PackageMember(name=name, sha256=pin, size_bytes=len(payload)),),
    )
    return materialize_source(spec, cache, lambda: io.BytesIO(payload))


class PreparedCInputs(NamedTuple):
    """Verified inputs/context only; preparation neither compiles subjects nor acquires runtime images."""

    context: ControlledContext
    buggy: MaterializationReceipt
    fixed: MaterializationReceipt
    fixture: MaterializationReceipt
    harness: bytes
    collector: bytes


def c_harness_plan(recipe: CCaptureRecipe, context: str, trial_id: str, role: str) -> CHarnessPlan:
    """Bind the entire validated recipe and logical role to one bounded producer-written bootstrap."""
    plan = CHarnessPlan.model_validate(
        dict(schema_version=1, context_fingerprint=context, trial_id=trial_id, role=role, recipe=recipe)
    )
    if len((plan.to_json() + "\n").encode()) > plan.MAXIMUM_BYTES:
        raise ValueError("C observer bootstrap exceeds its 1MiB byte limit")
    return plan


def prepare_c_inputs(
    recipe: CCaptureRecipe, cache: SourceCache, environment: MaterializationReceipt, *, fetch: bool = False
) -> PreparedCInputs:
    """Verify unmodified C inputs and owned observer, then freeze explicit synthetic function-selection provenance.

    Offline by default. Explicit fetch uses the existing pinned HTTPS adapter.
    The two roles may share original source bytes; their version labels identify
    different entry functions, not fabricated upstream commits. No compilation,
    shell, runtime acquisition or subject execution occurs here. Unknown schemas,
    changed bytes or malformed typed environment records reject before execution.
    """
    recipe = CCaptureRecipe.model_validate(recipe)
    if type(fetch) is not bool:
        raise ValueError("C fetch flag must be boolean")
    validate_materialization(environment, cache)
    if len(environment.spec.members) != 1:
        raise ValueError("C environment requires one typed metadata member")
    CEnvironment.from_json(_member_bytes(cache, environment, environment.spec.members[0].name))
    harness, collector = c_harness_bytes(), c_collector_bytes()
    if (digest(harness), digest(collector)) != (recipe.harness_sha256, recipe.collector_sha256):
        raise ValueError("Installed C observer/collector disagrees with frozen recipe")
    views = tuple(
        materialize_source_view(view, cache, public_reader if fetch else None)
        for view in (recipe.buggy, recipe.fixed, recipe.fixture)
    )
    metadata = (
        ("buggy-source", "buggy_source", views[0]),
        ("fixed-source", "fixed_source", views[1]),
        ("common-fixture", "test_fixture", views[2]),
        ("observation-harness", "harness", owned_c_package(cache, HARNESS_NAME, harness, "coleman-c-harness-v1")),
        (
            "bounded-process-collector",
            "dependency",
            owned_c_package(cache, "process.py", collector, "coleman-bounded-process-v1"),
        ),
        (
            "producer-recipe",
            "dependency",
            owned_c_package(cache, "producer-recipe.json", recipe.to_json().encode(), recipe.recipe),
        ),
        (
            "producer-identity",
            "dependency",
            owned_c_package(
                cache,
                "producer-identity.txt",
                (installed_producer_identity() + "\n").encode(),
                "coleman-package-identity-v1",
            ),
        ),
        ("execution-environment", "dependency", environment),
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
            "Unchanged source and common support files; explicit original function selection per role",
            "GCC -std=c11 -O0 -Wall -Wextra; owned wrapper supplies canonical decimal stdin and checks uint32 ABI",
            "No INCLUDEMAIN, random seeding, upstream build scripts, package installation or subject network",
            "Private mode0700 build tree in bounded executable tmpfs; no host compiler execution",
            "Per-case integer monotonic ns include child startup and are not CI scheduling costs",
            "UTC capture timestamps describe collection, never historical decision/feature availability",
            "Non-root read-only container, network=none, CPUs=2, PIDs=64, executable_tmpfs_bytes=134217728",
            f"Parent timeout_seconds={recipe.limits.timeout_seconds}; "
            f"combined_output_bytes={recipe.limits.maximum_output_bytes}; memory_bytes={recipe.limits.memory_bytes}",
            "Build deadline20s/output1MiB; case deadline5s/output64KiB; existing bounded process collector",
        ),
        commands=("python3 -I -B /harness/ColemanCCapture.py /harness/plan.json",),
        interventions=(
            "Role revisions explicitly select original bad versus goodB2G functions in the same unchanged source",
            "Owned main redirects canonical decimal input into private temporary stdin; "
            "original INCLUDEMAIN and goodG2B are excluded",
            "Mathematical checked uint32 increment oracle is frozen independently of either subject's observed outputs",
        ),
    )
    context = ControlledContext(
        schema_version=1,
        project=recipe.project,
        fault_id=recipe.fault_id,
        fault_origin=recipe.fault_origin,
        language="C",
        buggy_revision=recipe.buggy.output.source.version,
        fixed_revision=recipe.fixed.output.source.version,
        fixture_revision=recipe.fixture.output.source.version,
        dependence_group=recipe.dependence_group,
        test_ids=recipe.test_ids,
        inputs=tuple(ControlledInput.model_validate(dict(name=n, kind=k, materialization=r)) for n, k, r in metadata),
        runtime=runtime,
        limitations=recipe.limitations,
    )
    return PreparedCInputs(context, views[0], views[1], views[2], harness, collector)
