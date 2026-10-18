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

## New in 4.15

[Audit what a CI source actually supports](corpus-admission.md). Reuse existing
source completions and the registry, verify original bytes and produce versioned
JSON plus a human report covering 35 production-reality axes. Controlled
functional evidence, constructed mechanism checks and protected historical
evaluation retain separate gates. Missing evidence stays visible. The guide
includes complete published-wheel installation, audit and offline recovery examples.
The actual qualification verifies 61 source bindings, original raw producers and
two CI report structures. Three functional/mechanism regimes are covered; four
historical/HCS regimes remain uncovered. See the [4.15 release notes](releases/4.15.0.md).

## New in 4.14

[Reproduce difficult CI decisions in the deterministic stress laboratory](ci-stress-laboratory.md):
32 cases cover every pair of eleven declared factors, from rare failures and
name collisions to drift, late feedback and 50,000-test suites. Export known
truth and native Parquet, run the existing Random/UCB agents and resume their
checkpoints. Inspect exact work, modeled elapsed time and feedback availability.
The published-wheel qualification includes a real interrupted process and an
actual native 50,000-test history. Constructed cases test mechanisms; source
admission and empirical policy comparisons retain their own gates. See the
[4.14 release notes](releases/4.14.0.md) for compatibility and limits.

## New in 4.13

[Follow original LibSSH CI data through WTS/VTS Parquet and native replay](hcs-parquet-etl.md)
with the published wheel. Inspect all 208 sparse cells across thirteen known
job labels, including six without parsed tests. The qualification independently
checks 416 native result rows against the original CSV/JSON and verifies offline
continuation. Exact budget namespaces and result/checkpoint ownership preserve
committed results during interrupted runs. Missing evidence remains explicit;
source reproduction does not grant scientific admission or policy superiority.
See the [4.13 release notes](releases/4.13.0.md) for compatibility and limits.

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
retain their workflows. The original fixture-readiness view keeps all seven
regimes uncovered; the separate [4.15 source audit](corpus-admission.md) qualifies
three narrow purposes and retains four historical/HCS gaps.

[Exact resource accounting](resource-replay.md) and [frozen shadow evaluation](prospective-shadow.md)
retain their 4.3 behavior. The included synthetic shadow fixture remains inconclusive.

[Point-in-time observations](point-in-time-observations.md) preserve what each
decision knew, including late/corrected outcomes and conservative freshness.
The [corpus foundation](benchmark-corpus.md) retains two known-answer fixtures and
visible source-admission gaps. Its original phase-one readiness remains unchanged.
Use the [operational CI pilot](operational-pilot.md), [application API](application-api.md)
and [independent benchmarks](decision-benchmarks.md) for complete experiments and
correctness checks. See the [4.13 release notes](releases/4.13.0.md) for examples,
compatibility and limits.

## Quick Start

With Python 3.14 or later in an activated environment, install the GitHub release wheel.
For SHA-256 verification before installation, use the [complete setup command](setup.md#default-run-no-docker-required):

```bash
python -m pip install "https://github.com/jacksonpradolima/coleman/releases/download/v4.15.0/coleman-4.15.0-py3-none-any.whl#sha256=ad2daedb4c251d5ac001db7d73910d912d439629aea4e0a04d84dcc66e279fde"
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
