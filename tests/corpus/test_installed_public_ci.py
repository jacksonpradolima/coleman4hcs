"""Installed base SDK and standalone feasibility example outside checkout, without network or MCP."""

import json
import shutil
import subprocess

from tests.corpus.installed_source_support import installed_source_environment
from tests.corpus.public_actions_support import EXAMPLE, example_module, request, response, wire


def test_installed_public_feasibility_capture_outside_checkout(tmp_path, monkeypatch):
    """Use actual wheel contracts, artifact writer, receipt checkpoints and reader in an isolated process."""
    monkeypatch.setenv("UV_OFFLINE", "1")
    python, outside, env = installed_source_environment(tmp_path)
    shutil.copyfile(EXAMPLE, outside / "ci.py")
    module = example_module()
    req = request(module)
    (outside / "request-owned.json").write_text(req.to_json())
    for part, payload in zip(("run", "jobs"), wire(), strict=True):
        (outside / (part + "-owned.json")).write_text(response(module, req, part, payload).to_json())
    script = """
import hashlib, importlib.abc, importlib.util, sys
from pathlib import Path
assert importlib.util.find_spec("mcp") is None
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("coleman.cli", "coleman.transports", "coleman.application", "mcp")):
            raise AssertionError("Forbidden domain import: " + fullname)
guard = Guard()
sys.meta_path.insert(0, guard)
import coleman.observations, coleman.results.domain, coleman.corpus.admission_contracts
sys.meta_path.remove(guard)
spec = importlib.util.spec_from_file_location("reference", "ci.py")
ci = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ci
spec.loader.exec_module(ci)
request = ci.Request.from_json(Path("request-owned.json").read_bytes())
output = Path("evidence")
try:
    ci.capture(request, output)
except FileNotFoundError:
    pass
else:
    raise AssertionError("Missing receipts must require explicit fetch")
root = next(output.rglob("request.json")).parent
writer = ci.ArtifactWriter(root)
for part in ("run", "jobs"):
    record = ci.Response.from_json(Path(part + "-owned.json").read_bytes())
    writer.write_immutable(root / (part + ".json"), record.to_json() + "\\n")
root, snapshot = ci.capture(request, output)
assert snapshot.runtime == ci.InstalledRuntime.capture()
assert snapshot.example_sha256 == "sha256:" + hashlib.sha256(Path("ci.py").read_bytes()).hexdigest()
assert snapshot.conclusion == "insufficient_evidence_for_forward_tcpci"
assert snapshot.history().revisions[0].measurement.value == 60.0
assert ci.read_capture(root) == snapshot
before = {p: p.read_bytes() for p in root.iterdir() if p.is_file()}
assert ci.capture(request, output) == (root, snapshot)
assert before == {p: p.read_bytes() for p in root.iterdir() if p.is_file()}
print(snapshot.to_json())
"""
    completed = subprocess.run(
        [str(python), "-I", "-c", script], cwd=outside, env=env, check=True, capture_output=True, text=True, timeout=60
    )
    result = json.loads(completed.stdout)
    assert result["scope"] == "public_metadata_feasibility_only"
    assert result["upstream_first_available_at"] is None
