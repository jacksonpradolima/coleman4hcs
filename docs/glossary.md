# Terms and contracts

Use the linked guide for the definition's implementation context and limits.

| Term | Meaning here | Read more |
| --- | --- | --- |
| TCPCI | Test case prioritization in continuous integration | [Concepts](concepts.md) |
| Prioritization | Ordering candidate tests | [Policies](policies.md) |
| Selection | Choosing scheduled work under a constraint | [Experiments](experiments.md), [finite coverage](finite-coverage.md) |
| MAB | Multi-armed bandit; a family of adaptive decision methods | [Research](research.md) |
| Context | Features available before the decision | [Point-in-time observations](point-in-time-observations.md) |
| Reward | A transformed learning signal | [Policies and rewards](policies.md) |
| NAPFD / APFDc | Prioritization-quality metrics exposed as native `fitness` / `cost` | [Evaluation](evaluation-guide.md#read-native-results) |
| HCS | Highly configurable system | [Configurable systems](highly-configurable.md) |
| WTS / VTS | Whole test set / variant test set strategies | [WTS and VTS](highly-configurable.md) |
| RunSpec | Typed, resolved configuration for a native experiment | [Spec API](api/spec-models.md) |
| Run ID | Compatibility-aware hash prefix of canonical configuration | [Run identity](api/spec-run-id.md) |
| Provenance | Retained source, producer, configuration and environment identities | [Architecture](setup.md), [versioned results](domain-results.md) |
| Native projection | Explicit interpretation of analytical observations for replay | [Input format](input-data.md) |
| Checkpoint | Persisted state for continuing an experiment | [First run](getting-started.md#4-run-the-same-configuration-through-the-cli) |
| Completion receipt | Contract-specific marker binding a fully published set of verified artifacts | [Source materialization](corpus-materialization.md) |
| Outcome availability | When an outcome is available under the declared source meaning | [Observations](point-in-time-observations.md) |
| Receipt time | When the application first received an observation | [Observations](point-in-time-observations.md) |
| Repository family | Reviewed grouping for potentially shared source evidence | [Repeated observations](corpus-evaluation.md) |
| Protected comparison | Evaluation with frozen candidates, access rules and declared assumptions | [Protected evaluation](protected-evaluation.md) |
| Insufficient evidence | An explicit result when the protocol cannot justify a decision | [Evaluation guide](evaluation-guide.md) |
| Shadow evaluation | Recording recommendations while required CI continues | [Shadow workflow](prospective-shadow.md) |
| MCP | Optional transport for bounded application tools | [Application API](application-api.md) |
