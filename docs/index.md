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

## New in 4.3

Audit CI budgets with [exact resource accounting](resource-replay.md): estimates,
reservations, actual runner work, release and overhead remain separate. Run
[frozen shadow evaluation](prospective-shadow.md) through the native runner,
sealing recommendations before feedback and recovering committed evidence after
interruption. Required tests keep their full base order. The included one-day
synthetic fixture remains inconclusive against its seven-day evidence requirement.

[Point-in-time observations](point-in-time-observations.md) preserve what each
decision knew, including late/corrected outcomes and conservative freshness.
The [corpus foundation](benchmark-corpus.md) retains two known-answer fixtures and
visible source-admission gaps; all seven diversity regimes remain uncovered.
Use the [operational CI pilot](operational-pilot.md), [application API](application-api.md)
and [independent benchmarks](decision-benchmarks.md) for complete experiments and
correctness checks. See the [4.3 release notes](releases/4.3.0.md) for examples,
compatibility and limits.

## Quick Start

With Python 3.14 or later in an activated environment, install the GitHub release wheel.
For SHA-256 verification before installation, use the [complete setup command](setup.md#default-run-no-docker-required):

```bash
python -m pip install "https://github.com/jacksonpradolima/coleman/releases/download/v4.3.0/coleman-4.3.0-py3-none-any.whl#sha256=1e324fd6d9e82e4ab84ca17cc38d814fd407fcb1ed7e04ab6826a5dea356739e"
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
