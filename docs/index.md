# Coleman

**Multi-Armed Bandit based Test Case Prioritization for Continuous Integration**

Coleman is a framework that applies Multi-Armed Bandit (MAB) algorithms to
solve the Test Case Prioritization problem in Continuous Integration (CI)
environments.

## Features

- Adaptive learning from test execution feedback
- Multiple MAB policies: baseline, Bayesian/stochastic, adversarial, and non-stationary variants
- Contextual bandits: LinUCB/SWLinUCB plus LinTS and contextual epsilon-greedy variants
- HCS support with WTS and VTS strategies
- Cost-effective prioritization under time budgets
- **Typed configuration** — Pydantic v2 models with YAML configs and composable config packs
- **Library-first API** — `run()`, `run_many()`, `sweep()`, `load_spec()`
- **`coleman` CLI** — thin wrapper: `coleman run`, `coleman sweep`, `coleman validate`
- **Sweep engine** — grid (Cartesian) and zip (paired) parameter expansion with seed replication
- **Deterministic `run_id`** — `sha256(canonical_json(spec))[:12]` for exact replication
- **Provenance tracking** — `spec.resolved.json` + `provenance.json` per run

## New in 4.2

Reproduce what a CI decision knew with [point-in-time observations](point-in-time-observations.md):
late corrections cannot change an earlier snapshot, stale signals fall back conservatively,
and versioned evidence survives checkpoint continuation. The
[benchmark corpus foundation](benchmark-corpus.md) provides two known-answer fixtures
and source registrations; all seven required diversity regimes remain uncovered.

The [operational CI pilot](operational-pilot.md) continues to audit native selections,
keep missed failures in recall, and distinguish runner work from elapsed time.
Use the [application API](application-api.md) and [independent benchmarks](decision-benchmarks.md)
to run experiments and check finite-model mathematics. See the
[release notes](releases/4.2.0.md) for complete examples, compatibility and limits.

## Quick Start

With Python 3.14 or later in an activated environment, install the GitHub release wheel.
For SHA-256 verification before installation, use the [complete setup command](setup.md#default-run-no-docker-required):

```bash
python -m pip install "https://github.com/jacksonpradolima/coleman/releases/download/v4.2.0/coleman-4.2.0-py3-none-any.whl#sha256=8e929093fc2103d0de141c3e64dccb099001dfd6216f9051cfe4ceee120ebef8"
coleman run --config my-experiment.yaml
```

Or use the library API:

```python
from coleman.api import run, load_spec

spec = load_spec("my-experiment.yaml")
result = run(spec)
```

See the [Getting Started](getting-started.md) guide for full instructions
and the [Configuration](configuration.md) guide for the YAML schema,
config packs, sweep engine, and determinism contract.
For deep post-run analytics, see the [Analysis Playbook](analysis-playbook.md).
For advanced customization and parallel extension contracts, see
[Extensibility & Parallelism](extensibility.md).
