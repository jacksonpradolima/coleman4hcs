"""Execute the packaged Java harness against owned discovery side effects in the pinned offline JVM."""

import io
import os
from uuid import uuid4

import pytest

from coleman.corpus import SourceCache
from coleman.corpus.source_views import SourceViewSpec, materialize_source_view, source_view_bytes
from coleman.transports.junit import load_junit_recipe
from coleman.transports.junit_capture import _run_role
from coleman.transports.junit_protocol import parsed_junit_stream
from coleman.transports.junit_source import owned_package, prepare_junit_inputs
from tests.corpus.test_source_views import changed, source_spec


@pytest.mark.parametrize("newline", [False, True])
def test_real_harness_keeps_discovery_output_separate_from_protocol(tmp_path, newline):
    location = os.environ.get("COLEMAN_JUNIT_REAL_CACHE")
    if location is None:
        pytest.skip("Explicit offline source cache and pinned local JVM image required")
    cache = SourceCache(location)
    recipe = load_junit_recipe()
    environment = owned_package(cache, "owned-harness-environment.json", b"Owned harness regression", "owned-v1")
    prepared = prepare_junit_inputs(recipe, cache, environment)
    method = "println" if newline else "print"
    case = (
        "package owned; public final class NoisyCase extends junit.framework.TestCase {"
        "static { System.out." + method + '("static-out"); System.err.print("static-err"); }'
        'public NoisyCase(String name) { super(name); System.out.print("constructor-out"); }'
        'public void test_ci() { System.out.print("case-out"); System.err.print("case-err"); }'
        "}"
    ).encode()
    excluded = b'package owned; public final class NotACase {static { System.out.print("excluded-out"); } }'
    entries = (("src/test/owned/NoisyCase.java", case), ("src/test/owned/NotACase.java", excluded))
    inputs = tuple(source_spec(name, payload) for name, payload in entries)
    output = source_spec(
        "owned-noisy-fixture",
        source_view_bytes(entries),
        "zip",
        [item.members[0].model_dump() for item in sorted(inputs, key=lambda item: item.members[0].name)],
    )
    view = SourceViewSpec(schema_version=1, inputs=inputs, output=output)
    data = dict(entries)
    fixture = materialize_source_view(view, cache, lambda item: lambda: io.BytesIO(data[item.members[0].name]))
    prepared = prepared._replace(fixture=fixture)
    recipe = changed(recipe, test_ids=("owned.NoisyCase::test_ci",))
    config = tmp_path / "docker-config"
    config.mkdir()
    streams = _run_role(
        ("docker", "--config", str(config)),
        recipe,
        prepared,
        cache,
        "buggy",
        "coleman-junit-owned-noisy-" + uuid4().hex,
    )
    assert streams.termination == "exited"
    assert streams.exit_code == 0
    header, cases = parsed_junit_stream(streams.stdout, recipe)
    assert header.excluded_classes == ("owned.NotACase",)
    assert len(cases) == 1
    assert cases[0].failures == cases[0].errors == 0
    assert cases[0].diagnostics == "case-outcase-err"
    assert b"static-out" in streams.stderr
    assert b"static-err" in streams.stderr
    assert b"constructor-out" in streams.stderr
    assert b"excluded-out" not in streams.stdout + streams.stderr
