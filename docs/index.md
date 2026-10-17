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

## New in 4.9

[Run larger LRTS histories with explicit stage views](lrts-stage-views.md).
The published wheel preserves **18,901 ActiveMQ method observations** and
**30,066 Log4j observations**, with separate Ubuntu/Windows native views.
Exact source-context/node bindings keep stages distinct; all raw methods and
processed stages remain analytical evidence. Each four-build view produces
eight Random/UCB result rows and resumes committed checkpoints.

The Log4j views share source project/build groups. Missing executor, outcome and
feature clocks, expected test universe, verified continuity and concurrency
remain explicit blockers. The workflow is a post-hoc serial simulation, and
broader corpus admission and confirmation remain open.
See the [4.9 release notes](releases/4.9.0.md).

This revision also adds a [controlled Java fault matrix](java-fault-matrix.md):
three curated upstream fixes across CLI and Codec, raw/source validation and
offline recovery. Its guide builds this revision's wheel. The initial 357
same-test pairs retain both-fail and inconclusive results; this controlled layer
does not establish historical CI chronology, independence or policy efficacy.

## Existing real-data workflows

[Turn real RTPTorrent CI history into Parquet and native TCPCI replays](rtptorrent-parquet-etl.md).
The published wheel verifies the complete ZIP64 source, preserves **46,995 class
executions / 1,662 source jobs**, and projects 18 executions from eight declared
jobs into Random/UCB replay with the existing checkpoints. All exclusions, units
and source/producer/file identities remain inspectable.

The [LRTS download-to-Parquet workflow](lrts-parquet-etl.md) remains available in
the same wheel: eight Karaf builds, 6,876 method observations, 6,567 compatible
native executions and 309 not-run exclusions. Each workflow produces 16 result
rows and resumes without duplicating committed rounds. Their post-hoc serial
and budget-censored feedback assumptions do not establish observed chronology,
policy efficacy or prospective savings. Source admission and confirmation remain open.

The 4.8 release also corrected Linux worker memory accounting while retaining its
limits and historical general telemetry. See the [4.8 release notes](releases/4.8.0.md).

[Controlled observations](controlled-faults.md), the
[pinned JUnit3 source recipe](junit-source-capture.md),
[RTPTorrent](rtptorrent.md) and [LRTS](lrts.md) readers,
[corpus profiles](benchmark-corpus.md) and [source cache](corpus-materialization.md)
retain their workflows. All seven required diversity regimes remain uncovered.

[Exact resource accounting](resource-replay.md) and [frozen shadow evaluation](prospective-shadow.md)
retain their 4.3 behavior. The included synthetic shadow fixture remains inconclusive.

[Point-in-time observations](point-in-time-observations.md) preserve what each
decision knew, including late/corrected outcomes and conservative freshness.
The [corpus foundation](benchmark-corpus.md) retains two known-answer fixtures and
visible source-admission gaps; all seven diversity regimes remain uncovered.
Use the [operational CI pilot](operational-pilot.md), [application API](application-api.md)
and [independent benchmarks](decision-benchmarks.md) for complete experiments and
correctness checks. See the [4.9 release notes](releases/4.9.0.md) for examples,
compatibility and limits.

## Quick Start

With Python 3.14 or later in an activated environment, install the GitHub release wheel.
For SHA-256 verification before installation, use the [complete setup command](setup.md#default-run-no-docker-required):

```bash
python -m pip install "https://github.com/jacksonpradolima/coleman/releases/download/v4.9.0/coleman-4.9.0-py3-none-any.whl#sha256=eb59ae45bc280a3d7cfa33bc1c360f68fc69859d9632c864bd996b29ac81a9bd"
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
