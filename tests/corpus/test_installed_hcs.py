"""Fresh base wheel outside checkout: HCS Parquet, actual strategies/checkpoints and import boundary."""

import json
import os
import shutil
import sys
from pathlib import Path

from tests.corpus.hcs_support import hcs_recipe
from tests.corpus.test_installed_controlled import ROOT, command


def test_installed_hcs_sources_native_recovery_and_domain_import_boundary(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update(UV_LINK_MODE="copy", UV_NO_PROGRESS="1")
    dist = tmp_path / "dist"
    command([uv, "build", "--wheel", "--out-dir", str(dist)], ROOT, env)
    virtual = tmp_path / "installed"
    command([uv, "venv", str(virtual), "--python", sys.executable], tmp_path, env)
    python = virtual / "bin/python"
    command([uv, "pip", "install", "--offline", "--python", str(python), str(next(dist.glob("*.whl")))], tmp_path, env)
    outside = tmp_path / "outside"
    outside.mkdir()
    recipe, csv_bytes, jobs_bytes = hcs_recipe()
    (outside / "recipe.json").write_text(recipe.to_json())
    (outside / "executions.csv").write_bytes(csv_bytes)
    (outside / "jobs.json").write_bytes(jobs_bytes)
    script = """
import importlib.abc,importlib.util,json,sys
from pathlib import Path
assert importlib.util.find_spec('mcp') is None
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='coleman.cli' or fullname.startswith(('coleman.transports','mcp')):
            raise AssertionError('HCS domain imported transport: '+fullname)
boundary=Boundary();sys.meta_path.insert(0,boundary)
from coleman.corpus.hcs_etl import export_hcs_parquet,validate_hcs_etl
from coleman.corpus.hcs_contracts import HCSReceipt,HCSRecipe
assert Path(sys.modules['coleman.corpus.hcs_etl'].__file__).resolve().is_relative_to(Path(sys.prefix))
sys.meta_path.remove(boundary)
from coleman.application.hcs import run_hcs_replay
from coleman.artifacts import ArtifactWriter
from coleman.corpus.materialization import SourceCache,materialize_source
from coleman.transports.hcs_etl import load_hcs_recipe
from coleman.corpus.lrts_etl_contracts import LRTSETLReceipt
import polars as pl
old=LRTSETLReceipt.from_json(Path('legacy-completion.json').read_bytes())
assert old.schema_version==1
assert old.recipe.feedback.value=='budget-censored-native-RNFail-zero-nondetection'
recipe=HCSRecipe.from_json(Path('recipe.json').read_bytes());cache=SourceCache('cache')
for package,name in [(recipe.executions,'executions.csv'),(recipe.jobs,'jobs.json')]:
    materialize_source(package,cache,lambda name=name:Path(name).open('rb'))
p=export_hcs_parquet(recipe,cache,ArtifactWriter('etl'))
assert export_hcs_parquet(recipe,cache,ArtifactWriter('etl'))==p
r=HCSReceipt.from_json(p.read_bytes());validate_hcs_etl(r,cache,p.parent)
assert r.accounting.replay_rows==10 and r.accounting.selected_rows==17
assert r.evidence.setup_seconds is None and r.admission=='blocked_source_evidence'
proof=[]
for strategy in ['wts','vts']:
    a=run_hcs_replay(p,cache,Path('native')/strategy,strategy=strategy)
    b=run_hcs_replay(p,cache,Path('native')/strategy,strategy=strategy)
    assert a.run_id==b.run_id
    rows=pl.read_parquet(str(Path(b.artifacts_dir)/'results/**/*.parquet'),hive_partitioning=True)
    assert rows.height=={'wts':24,'vts':16}[strategy]
    assert set(rows['budget_value'])=={0.5,1.0}
    identities={(x['scenario'],x['variant'],x['policy'],x['budget_value'],x['step']) for x in rows.to_dicts()}
    assert len(identities)==rows.height
    markers=list(Path(b.artifacts_dir).rglob('progress_ex1.json'))
    assert markers and all('budget-v1' in str(m) for m in markers)
    proof.append(dict(strategy=strategy,rows=rows.height,run_id=b.run_id))
real=[]
if sys.argv[1]!='-':
    recipe=load_hcs_recipe();cache=SourceCache(sys.argv[1]);p=export_hcs_parquet(recipe,cache,ArtifactWriter('real-etl'))
    assert export_hcs_parquet(recipe,cache,ArtifactWriter('real-etl'))==p
    r=HCSReceipt.from_json(p.read_bytes());validate_hcs_etl(r,cache,p.parent)
    counts=r.accounting
    assert (counts.input_rows,counts.selected_rows,counts.replay_rows,counts.other_attempt_rows)==(278950,885,856,29)
    assert len(r.variants)==7
    cells=pl.read_parquet(p.parent/'cells.parquet');assert cells.height==112
    assert cells.filter(pl.col('cell_status')=='eligible').height==44
    for strategy in ['wts','vts']:
        a=run_hcs_replay(p,cache,Path('real-native')/strategy,strategy=strategy)
        b=run_hcs_replay(p,cache,Path('real-native')/strategy,strategy=strategy)
        assert a.run_id==b.run_id
        rows=pl.read_parquet(str(Path(b.artifacts_dir)/'results/**/*.parquet'),hive_partitioning=True)
        assert rows.height=={'wts':240,'vts':176}[strategy]
        assert set(rows['budget_value'])=={0.5,1.0}
        real.append(dict(strategy=strategy,run_id=b.run_id,rows=rows.height,artifacts_dir=b.artifacts_dir))
print(json.dumps(dict(installed=True,owned=proof,real=real,real_completion=str(p) if real else None),sort_keys=True))
"""
    shutil.copyfile(
        ROOT / "tests/corpus/fixtures/lrts-etl-owned-completion-v1.json", outside / "legacy-completion.json"
    )
    result = command([str(python), "-c", script, env.get("COLEMAN_HCS_REAL_CACHE", "-")], outside, env)
    proof = json.loads(result.stdout.strip().splitlines()[-1])
    assert proof["installed"]
    assert len(proof["real"]) == (2 if "COLEMAN_HCS_REAL_CACHE" in env else 0)
    (tmp_path / "installed-hcs-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    args = [
        str(python),
        "-m",
        "coleman.transports.hcs_etl",
        "--recipe",
        "recipe.json",
        "--cache",
        "cli-cache",
        "--output",
        "cli-output",
        "--run",
    ]
    first = command(args + ["--executions", "executions.csv", "--jobs", "jobs.json"], outside, env)
    second = command(args, outside, env)
    assert json.loads(first.stdout.strip().splitlines()[-1]) == json.loads(second.stdout.strip().splitlines()[-1])
    assert Path(virtual).resolve() != ROOT.resolve()
