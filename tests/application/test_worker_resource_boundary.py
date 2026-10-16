"""Actual Linux worker memory belongs to its executable, independently of a large parent process."""

import io
import subprocess
import sys
from types import SimpleNamespace

import pytest

from coleman.application import worker
from coleman.application.worker import _worker_peak_rss_mib


@pytest.mark.parametrize(
    "metadata,expected",
    [
        ("Name:\towned\nVmHWM:\t126976 kB\n", 124.0),
        ("VmHWM:\t0 kB\n", 0.0),
        ("VmRSS:\t126976 kB\n", None),
        ("VmHWM:\tunknown kB\n", None),
        ("VmHWM:\t-1 kB\n", None),
        ("VmHWM:\t128 MB\n", None),
        ("VmHWM:\t128\n", None),
    ],
)
def test_linux_own_executable_peak_units_and_unavailable_kernel_metadata(monkeypatch, metadata, expected):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(worker, "Path", lambda _: SimpleNamespace(open=lambda **_kwargs: io.StringIO(metadata)))
    assert _worker_peak_rss_mib() == expected


def test_unreadable_linux_kernel_peak_is_unavailable(monkeypatch):
    def unreadable(*_args, **_kwargs):
        raise OSError("Owned kernel metadata unavailable")

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(worker, "Path", lambda _: SimpleNamespace(open=unreadable))
    assert _worker_peak_rss_mib() is None


def test_other_platforms_keep_existing_resource_sampler(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("coleman.telemetry.resources._get_peak_rss_mib", lambda: 124.0)
    assert _worker_peak_rss_mib() == 124.0


@pytest.mark.skipif(sys.platform != "linux", reason="Linux pre-exec inherited ru_maxrss regression")
def test_large_parent_does_not_consume_worker_memory_and_actual_worker_cap_still_applies(tmp_path):
    script = """
from pathlib import Path
from coleman.application import ApplicationService, Caller, Limits, Submit, default_registry

# Keep these real resident bytes until both worker types finish. Linux ru_maxrss
# can inherit this pre-exec parent footprint; /proc/self/status VmHWM cannot.
parent_memory = bytearray(600 * 1024**2)
caller = Caller(subject='owned-linux-resource-regression')
with ApplicationService(default_registry(), root=Path('service')) as service:
    for index, spec in enumerate(('weighted-overlap-3', 'native-ci-smoke-v1')):
        initial = service.submit(Submit(schema_version=1, request_id=str(index), spec_id=spec), caller=caller)
        final = service.wait(initial.job_id, caller=caller)
        assert final.state == 'completed', (spec, final.state, final.error)
    initial = service.submit(Submit(schema_version=1, request_id='actual-cap', spec_id='weighted-overlap-3',
                                    limits=Limits(max_memory_bytes=1024**2)), caller=caller)
    final = service.wait(initial.job_id, caller=caller)
    assert (final.state, final.error) == ('failed', 'resource_limit'), (final.state, final.error)
assert len(parent_memory) == 600 * 1024**2
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
