"""Exercise authored observer helpers with real bounded owned children; no foreign sources or Docker."""

import json
import runpy
import sys
from pathlib import Path

import pytest

from coleman.transports.c_source import c_collector_bytes, c_harness_bytes
from coleman.transports.junit_process import ProcessStreams, collect_process
from tests.corpus.test_source_views import digest


def harness(tmp_path):
    """Load authored source as a standalone helper, preserving its actual shared collector bytes."""
    source = tmp_path / "ColemanCCapture.py"
    source.write_bytes(c_harness_bytes())
    (tmp_path / "process.py").write_bytes(c_collector_bytes())
    return runpy.run_path(str(source), run_name="owned_c_observer")


def test_shared_collector_is_exact_pinned_and_rejects_changes(tmp_path):
    module = harness(tmp_path)
    recipe = {"collector_sha256": digest(c_collector_bytes())}
    actual = module["collector"](recipe)
    result = actual((sys.executable, "-c", "print('owned')"), timeout_seconds=5, maximum_output_bytes=1024)
    assert result.stdout == b"owned\n"
    (tmp_path / "process.py").write_bytes(b"raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="digest mismatch"):
        module["collector"](recipe)


@pytest.mark.parametrize("value", [0, 1, 2, 2**31, 2**32 - 2, 2**32 - 1])
def test_observer_oracle_matches_independent_mathematical_expectation(tmp_path, value):
    from coleman.transports.c_contracts import CInputCase

    module = harness(tmp_path)
    assert module["expected"](value) == CInputCase(test_id="owned", value=value).expected_stdout


@pytest.mark.parametrize("mode", ["passed", "assertion_failed", "test_error", "output_limit", "timeout"])
def test_owned_bounded_child_preserves_physical_outcome_and_full_prefix(tmp_path, capsys, mode):
    module = harness(tmp_path)
    executable = tmp_path / "owned-child"
    code = {
        "passed": "print(1)",
        "assertion_failed": "print(0)",
        "test_error": "raise RuntimeError('owned error')",
        "output_limit": "print('x'*100000)",
        "timeout": "import time;time.sleep(15)",
    }[mode]
    executable.write_text("#!" + sys.executable + "\n" + code + "\n")
    executable.chmod(0o700)
    failed = module["observe"](collect_process, executable, {"test_id": "owned", "value": 0})
    raw = capsys.readouterr().out
    prefix, payload = raw.rstrip("\n").split("\t", 1)
    assert prefix == "COLEMAN-C-V1"
    record = json.loads(payload)
    assert failed == (mode != "passed")
    assert record["outcome"] == ("test_error" if mode == "output_limit" else mode)
    assert record["elapsed"]["value"] >= 0
    if mode == "timeout":
        assert record["elapsed"]["value"] >= 5 * 10**9
        assert record["termination"] == "timeout"
    if mode == "output_limit":
        assert len(record["stdout"]) + len(record["stderr"]) == 65536


def test_undecodable_child_bytes_do_not_silently_become_an_assertion(tmp_path):
    module = harness(tmp_path)

    def raw(*args, **kwargs):
        return ProcessStreams(b"\xff", b"", 0, 0, "exited")

    with pytest.raises(UnicodeDecodeError):
        module["observe"](raw, Path("unused"), {"test_id": "owned", "value": 0})


@pytest.mark.parametrize("termination,exit_code", [("exited", 1), ("timeout", -9), ("output_limit", 0)])
def test_incomplete_build_or_probe_never_promotes_success(tmp_path, termination, exit_code):
    module = harness(tmp_path)

    def raw(*args, **kwargs):
        return ProcessStreams(b"owned compile output", b"owned diagnostic", exit_code, 1, termination)

    with pytest.raises(RuntimeError, match="probe failed"):
        module["checked_process"](raw, ("owned-compiler",))


def test_wrapper_selects_only_declared_function_and_no_original_main(tmp_path):
    module = harness(tmp_path)
    recipe = {"source_member": "subject.c", "buggy_function": "bad", "fixed_function": "goodB2G"}
    buggy, fixed = (module["wrapper"](recipe, role) for role in ("buggy", "fixed"))
    assert "bad();" in buggy and "goodB2G();" in fixed
    assert "INCLUDEMAIN" not in buggy + fixed
    assert "tmpfile()" in fixed and "UINT_MAX != 4294967295U" in fixed
    assert module["instant"]().endswith("Z")
