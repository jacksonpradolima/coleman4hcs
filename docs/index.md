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

## New in 4.12

[Inspect a constructed C weakness](c-seeded-source-capture.md) with the published
wheel: unchanged NIST Juliet functions, pinned GCC and six frozen uint32 boundary
inputs checked against an independent arithmetic oracle. The installed
qualification retains five both-pass pairs and one assertion-failure/pass pair.
Raw streams, compiler/ABI facts, offline replay and actual interruption recovery
remain inspectable. This is one synthetic template, not six independent faults
or historical CI.

The guide includes the verified wheel, explicit native arm64 Docker preparation,
the typed API, component terms and recovery. Shared transport isolation keeps
runtime policies consistent; JUnit v2 markers retain original container ownership
after moving an output tree, while legacy v1 markers retain their reader and
cleanup rule. See the [4.12 release notes](releases/4.12.0.md) for compatibility.

## Controlled Python source

[Reproduce a controlled Python fault](python-source-capture.md) with the same
published wheel: unchanged PySnooper source/tests, pinned dependencies and two
frozen locale contexts. The ASCII exception-to-pass comparison remains
**inconclusive**; the UTF-8 contrast is **both-pass**. Component terms, raw evidence,
offline validation and real interruption recovery stay inspectable. These
contexts share one curated fix and do not establish independent observations or
policy efficacy. The producer was introduced in
[4.11](releases/4.11.0.md); its guide now uses the qualified current wheel.

## Controlled Java source matrix

[Collect a controlled Java fault matrix](java-fault-matrix.md) with the published
wheel: three curated upstream fixes across Apache Commons CLI and Codec, frozen
source/runtime recipes, typed indices and offline recovery. Source/context/raw
validation keeps every selected outcome inspectable. The initial qualification
retained **714 role observations / 357 same-test pairs**, including both-fail and
inconclusive results. Incomplete collection never acquires a completion marker.

The complete guide includes the verified wheel, explicit native arm64 Docker
preparation, collection, offline validation and recovery. Controlled observations
retain their admission and generalization limits. See the
[4.10 release notes](releases/4.10.0.md) for compatibility and scientific limits.

## LRTS stage views

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
correctness checks. See the [4.12 release notes](releases/4.12.0.md) for examples,
compatibility and limits.

## Quick Start

With Python 3.14 or later in an activated environment, install the GitHub release wheel.
For SHA-256 verification before installation, use the [complete setup command](setup.md#default-run-no-docker-required):

```bash
python -m pip install "https://github.com/jacksonpradolima/coleman/releases/download/v4.12.0/coleman-4.12.0-py3-none-any.whl#sha256=5802f83dfa4608a979f132e6e00ea3d8a128c88f4b7bbde07afcd416c33df55e"
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
