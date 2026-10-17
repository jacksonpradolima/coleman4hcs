"""Owned Docker-boundary tests inspect actual staging; installed tests separately execute the real subject."""

import csv
import io
from pathlib import Path

import pytest

from coleman.transports import c_process as process
from coleman.transports.c_contracts import CHarnessPlan
from coleman.transports.c_source import c_harness_plan, prepare_c_inputs
from coleman.transports.junit_process import ProcessStreams
from tests.corpus.test_c_capture import prepared_inputs


@pytest.mark.parametrize("failure", [None, OSError, KeyboardInterrupt])
def test_staging_uses_exact_original_bytes_fixed_guards_and_owned_cleanup(tmp_path, monkeypatch, failure):
    spec, cache, environment = prepared_inputs(tmp_path)
    prepared = prepare_c_inputs(spec, cache, environment)
    plan = c_harness_plan(spec, prepared.context.fingerprint, "owned-process", "buggy")
    container = "coleman-c-" + "a" * 24
    cleanup = []
    staged = []

    def observe(args, *, timeout_seconds, maximum_output_bytes):
        assert timeout_seconds == 60
        assert maximum_output_bytes == 1024**2
        assert "--pull=never" in args
        assert args[args.index("--network") + 1] == "none"
        assert args[args.index("--name") + 1] == container
        assert "--read-only" in args
        assert args[args.index("--tmpfs") + 1] == "/tmp:rw,exec,nosuid,nodev,size=134217728,mode=1777"
        assert "no-new-privileges" in args
        assert "PYTHONCOERCECLOCALE=0" in args
        assert "PYTHONUTF8=0" in args
        assert "-B" in args
        assert "-I" in args
        mounts = {}
        for index, argument in enumerate(args):
            if argument == "--mount":
                parts = next(csv.reader(io.StringIO(args[index + 1])))
                assert "readonly" in parts
                options = dict(part.split("=", 1) for part in parts if "=" in part)
                mounts[options["target"]] = Path(options["source"])
        assert set(mounts) == {"/source", "/fixture", "/harness"}
        staged.extend(mounts.values())
        assert (mounts["/source"] / "subject.c").read_bytes() == b"Owned schema-only source fixture.\n"
        assert (mounts["/harness"] / "ColemanCCapture.py").read_bytes() == prepared.harness
        assert CHarnessPlan.from_json((mounts["/harness"] / "plan.json").read_bytes()) == plan
        assert (mounts["/harness"] / "process.py").read_bytes() == prepared.collector
        if failure is not None:
            raise failure("Owned process boundary failure")
        return ProcessStreams(b"owned-observer-stream", b"", 0, 0, "exited")

    monkeypatch.setattr(process, "collect_process", observe)
    monkeypatch.setattr(process, "_cleanup", lambda base, name: cleanup.append(name))
    if failure is not None:
        with pytest.raises(failure):
            process.run_c_role(("owned-docker",), spec, prepared, cache, plan, container)
    else:
        actual = process.run_c_role(("owned-docker",), spec, prepared, cache, plan, container)
        assert actual.stdout == b"owned-observer-stream"
    assert cleanup == [container]
    assert staged
    assert all(not path.exists() for path in staged)
