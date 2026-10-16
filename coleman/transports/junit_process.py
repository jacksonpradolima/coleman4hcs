"""Optional POSIX bounded process stream collector used only by the JUnit source producer."""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time
from contextlib import suppress
from typing import Literal, NamedTuple


class ProcessStreams(NamedTuple):
    """Exact bounded stream prefixes, actual exit code and monotonic producer-process nanoseconds.

    A limit/timeout identifies incomplete collection. These bytes must never be
    parsed as a successfully completed test experiment. The caller owns cleanup
    of an external container; this helper owns only its child process group.
    """

    stdout: bytes
    stderr: bytes
    exit_code: int
    elapsed_ns: int
    termination: Literal["exited", "timeout", "output_limit"]


def _drain(selector, streams, deadline, maximum):
    """Read ready bounded pipe prefixes until EOF, deadline or the first excess byte."""
    total = 0
    while selector.get_map():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "timeout"
        for key, _ in selector.select(min(remaining, 0.1)):
            chunk = os.read(key.fd, min(65536, maximum - total + 1))
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            accepted = min(len(chunk), maximum - total)
            streams[key.data].extend(chunk[:accepted])
            total += accepted
            if accepted != len(chunk):
                return "output_limit"
    return "exited"


def collect_process(args: tuple[str, ...], *, timeout_seconds: int, maximum_output_bytes: int) -> ProcessStreams:
    """Collect a trusted command without a shell, with a combined byte cap and parent execution timeout.

    Only POSIX is supported. No ambient authentication, proxy or Python settings
    are inherited; PATH is retained solely for executable/helper resolution.
    The process gets a new group, killed on timeout/oversized output/error so
    inherited pipe writers cannot keep collection alive indefinitely. Output
    prefixes remain available for an explicit incomplete evidence packet. Exit
    codes alone are never interpreted as assertion failures. OS failures propagate.
    """
    if os.name != "posix":
        raise OSError("Bounded JUnit process collection requires POSIX")
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 120:
        raise ValueError("Process timeout must be 1..120 seconds")
    if type(maximum_output_bytes) is not int or not 1 <= maximum_output_bytes <= 4 * 1024**2:
        raise ValueError("Process output limit must be 1..4 MiB")
    if not args or any(not isinstance(arg, str) or "\x00" in arg for arg in args):
        raise ValueError("Process requires nonempty NUL-free string arguments")
    begin = time.monotonic_ns()
    deadline = time.monotonic() + timeout_seconds
    streams = {"stdout": bytearray(), "stderr": bytearray()}
    termination: Literal["exited", "timeout", "output_limit"] = "exited"
    env = {"PATH": os.environ.get("PATH", os.defpath)}
    with subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, env=env
    ) as process:
        try:
            with selectors.DefaultSelector() as selector:
                for pipe, name in ((process.stdout, "stdout"), (process.stderr, "stderr")):
                    if pipe is None:
                        raise OSError("Process stream unavailable")
                    os.set_blocking(pipe.fileno(), False)
                    selector.register(pipe, selectors.EVENT_READ, name)
                termination = _drain(selector, streams, deadline, maximum_output_bytes)
                if termination == "exited":
                    try:
                        process.wait(timeout=max(0.001, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        termination = "timeout"
        finally:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            exit_code = process.wait(timeout=10)
    return ProcessStreams(
        bytes(streams["stdout"]), bytes(streams["stderr"]), exit_code, time.monotonic_ns() - begin, termination
    )
