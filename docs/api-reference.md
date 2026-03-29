# API Reference

## Experiment system

| Module | Description |
|--------|-------------|
| [`coleman4hcs.api`](api/experiment-api.md) | Library-first API: `run`, `run_many`, `sweep`, `load_spec`, `save_resolved` |
| [`coleman4hcs.spec.models`](api/spec-models.md) | Typed `RunSpec` and sub-spec Pydantic v2 models |
| [`coleman4hcs.spec.sweep`](api/spec-sweep.md) | Sweep engine: `SweepSpec`, `SweepAxis`, `expand_sweep` |
| [`coleman4hcs.spec.packs`](api/spec-packs.md) | Config pack resolution: `resolve_packs` |
| [`coleman4hcs.spec.run_id`](api/spec-run-id.md) | Deterministic `run_id`: `compute_run_id` |
| [`coleman4hcs.spec.io`](api/spec-io.md) | YAML I/O: `load_spec`, `save_resolved` |
| [`coleman4hcs.spec.provenance`](api/spec-provenance.md) | Provenance capture: `save_provenance` |
| [`coleman4hcs.cli`](api/cli.md) | CLI entry-point: `coleman run`, `coleman sweep`, `coleman validate` |

## Core framework

The core API is documented by module:

- [Agent](api/agent.md)
- [Bandit](api/bandit.md)
- [Environment](api/environment.md)
- [Evaluation](api/evaluation.md)
- [Policy](api/policy.md)
- [Reward](api/reward.md)
- [Scenarios](api/scenarios.md)
