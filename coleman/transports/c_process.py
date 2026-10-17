"""Optional bounded C source execution; compiler and interpreter run only in a pinned offline container."""

from __future__ import annotations

import os
import platform
import tempfile
from importlib.metadata import version
from pathlib import Path

from coleman.corpus import MaterializationReceipt, SourceCache
from coleman.transports.c_contracts import CCaptureRecipe, CEnvironment, CHarnessPlan
from coleman.transports.c_source import PreparedCInputs, owned_c_package
from coleman.transports.junit_capture import _cleanup, _docker_query, _mount, _stage_view
from coleman.transports.junit_process import ProcessStreams, collect_process


def c_runtime_environment(base: tuple[str, ...], recipe: CCaptureRecipe, cache: SourceCache) -> MaterializationReceipt:
    """Verify an already present official image and filtered Docker/host facts; never pull or probe a subject.

    Version/locale facts from the actual interpreter are separately checked in
    the harness header. Architecture translation must match the declared profile.
    No tags, unrelated containers or ambient authentication are inspected.
    Missing runtime or metadata mismatch raises; cache/I/O errors propagate.
    """
    image = "docker.io/library/gcc@" + recipe.runtime.image_digest
    info = _docker_query(
        base,
        ("image", "inspect", image, "--format", "[{{json .Os}},{{json .Architecture}},{{json .Size}},{{json .Id}}]"),
    )
    versions = _docker_query(
        base, ("version", "--format", "[{{json .Client.Version}},{{json .Server.Version}},{{json .Server.Arch}}]")
    )
    kernel = _docker_query(base, ("info", "--format", "{{json .KernelVersion}}"))
    if not isinstance(info, list) or len(info) != 4 or not isinstance(versions, list) or len(versions) != 3:
        raise ValueError("Unexpected filtered C Docker runtime metadata")
    target = recipe.runtime.platform.split("/")[1]
    aliases = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "amd64", "amd64": "amd64", "AMD64": "amd64"}
    host = aliases.get(platform.machine())
    if not all(isinstance(value, str) for value in versions):
        raise ValueError("Unexpected C Docker version/architecture scalar")
    server = aliases.get(versions[2])
    if info[0] != "linux" or info[1] != target or host is None or server is None:
        raise ValueError("C image/host architecture does not satisfy the reviewed recipe")
    if recipe.runtime.emulated != (len({host, server, target}) != 1):
        raise ValueError("C visible translation disagrees with the declared emulation profile")
    environment = CEnvironment.model_validate(
        {
            "schema_version": 1,
            "host_os": platform.system(),
            "host_architecture": platform.machine(),
            "host_release": platform.release(),
            "producer_python_version": platform.python_version(),
            "producer_package_version": version("coleman"),
            "docker_client_version": versions[0],
            "docker_server_version": versions[1],
            "docker_server_architecture": versions[2],
            "container_kernel": kernel,
            "image_os": info[0],
            "image_architecture": info[1],
            "image_size_bytes": info[2],
            "image_config_digest": info[3],
        }
    )
    return owned_c_package(
        cache, "execution-environment.json", environment.to_json().encode(), "filtered-local-c-runtime-v1"
    )


def run_c_role(
    base: tuple[str, ...],
    recipe: CCaptureRecipe,
    prepared: PreparedCInputs,
    cache: SourceCache,
    plan: CHarnessPlan,
    container_name: str,
) -> ProcessStreams:
    """Run one offline, non-root, read-only role and always clean only its owned container.

    All staged bytes are rehashed against existing cache receipts. Compilation
    uses a private temporary directory inside bounded tmpfs, never the host. Parent time/output
    guards use the existing process collector. Cleanup failure propagates so its
    durable start marker remains recoverable; no completed packet is fabricated.
    """
    source = prepared.buggy if plan.role == "buggy" else prepared.fixed
    with tempfile.TemporaryDirectory(prefix="coleman-c-stage-") as temporary:
        stage = Path(temporary)
        _stage_view(cache, source, stage / "source")
        _stage_view(cache, prepared.fixture, stage / "fixture")
        harness = stage / "harness"
        harness.mkdir(mode=0o755)
        harness.chmod(0o755)
        (harness / "ColemanCCapture.py").write_bytes(prepared.harness)
        (harness / "process.py").write_bytes(prepared.collector)
        (harness / "plan.json").write_text(plan.to_json() + "\n", encoding="utf-8")
        for path in harness.iterdir():
            path.chmod(0o644)
        uid, gid = (os.getuid(), os.getgid()) if os.getuid() else (65534, 65534)
        args = base + (
            "run",
            "--pull=never",
            "--rm",
            "--name",
            container_name,
            "--platform",
            recipe.runtime.platform,
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--cpus",
            "2",
            "--memory",
            str(recipe.limits.memory_bytes),
            "--read-only",
            "--tmpfs",
            "/tmp:rw,exec,nosuid,nodev,size=134217728,mode=1777",
            "--user",
            f"{uid}:{gid}",
            "--workdir",
            "/fixture",
        )
        for setting in (
            "TZ=" + recipe.runtime.timezone,
            "LANG=" + recipe.runtime.locale,
            "LC_ALL=" + recipe.runtime.locale,
            "PYTHONUTF8=0",
            "PYTHONCOERCECLOCALE=0",
            "PYTHONIOENCODING=utf-8",
            "PYTHONDONTWRITEBYTECODE=1",
            "PYTHONHASHSEED=0",
            "HOME=/tmp",
        ):
            args += ("--env", setting)
        for name in ("source", "fixture", "harness"):
            args += ("--mount", _mount(stage / name, "/" + name))
        try:
            return collect_process(
                args
                + (
                    "docker.io/library/gcc@" + recipe.runtime.image_digest,
                    "python3",
                    "-I",
                    "-B",
                    "/harness/ColemanCCapture.py",
                    "/harness/plan.json",
                ),
                timeout_seconds=recipe.limits.timeout_seconds,
                maximum_output_bytes=recipe.limits.maximum_output_bytes,
            )
        finally:
            _cleanup(base, container_name)
