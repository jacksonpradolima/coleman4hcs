"""Real owned child processes verify bounded collection without Docker or external network."""

import sys

import pytest

from coleman.transports.junit_process import collect_process


def command(code):
    return (sys.executable, "-I", "-c", code)


def test_real_child_exit_streams_and_environment_are_explicit(monkeypatch):
    monkeypatch.setenv("COLEMAN_OWNED_PARENT_VALUE", "must not reach child")
    result = collect_process(
        command(
            "import os,sys; assert 'COLEMAN_OWNED_PARENT_VALUE' not in os.environ; "
            "sys.stdout.buffer.write(b'owned CI output'); "
            "sys.stderr.buffer.write(b'owned diagnostic'); sys.exit(7)"
        ),
        timeout_seconds=5,
        maximum_output_bytes=1024,
    )
    assert result.stdout == b"owned CI output"
    assert result.stderr == b"owned diagnostic"
    assert result.exit_code == 7
    assert result.termination == "exited"
    assert result.elapsed_ns > 0


def test_exact_byte_limit_succeeds_and_overflow_preserves_a_bounded_prefix():
    exact = collect_process(
        command("import sys; sys.stdout.buffer.write(b'x'*1024)"), timeout_seconds=5, maximum_output_bytes=1024
    )
    assert exact.termination == "exited"
    assert len(exact.stdout) == 1024
    exceeded = collect_process(
        command("import sys,time; sys.stdout.buffer.write(b'x'*1000000); sys.stdout.flush(); time.sleep(10)"),
        timeout_seconds=5,
        maximum_output_bytes=1024,
    )
    assert exceeded.termination == "output_limit"
    assert len(exceeded.stdout) + len(exceeded.stderr) == 1024


def test_timeout_kills_the_actual_child_process_group():
    result = collect_process(
        command("import time; print('partial evidence',flush=True); time.sleep(10)"),
        timeout_seconds=1,
        maximum_output_bytes=1024,
    )
    assert result.termination == "timeout"
    assert result.stdout == b"partial evidence\n"
    assert result.exit_code < 0


def test_a_child_that_closes_pipes_cannot_escape_the_deadline():
    result = collect_process(
        command("import os,time; os.close(1); os.close(2); time.sleep(10)"),
        timeout_seconds=1,
        maximum_output_bytes=1024,
    )
    assert result.termination == "timeout"


@pytest.mark.parametrize(
    "seconds,maximum", [(0, 1024), (True, 1024), (121, 1024), (5, 0), (5, True), (5, 4 * 1024**2 + 1)]
)
def test_invalid_guards_reject_before_execution(seconds, maximum):
    args = command('raise RuntimeError("must not execute")')
    with pytest.raises(ValueError):
        collect_process(args, timeout_seconds=seconds, maximum_output_bytes=maximum)


def test_invalid_command_and_missing_executable_reject(tmp_path):
    with pytest.raises(ValueError):
        collect_process((), timeout_seconds=1, maximum_output_bytes=1)
    with pytest.raises(ValueError):
        collect_process(("bad\x00command",), timeout_seconds=1, maximum_output_bytes=1)
    with pytest.raises(OSError):
        collect_process((str(tmp_path / "not-created" / "command"),), timeout_seconds=1, maximum_output_bytes=1)
