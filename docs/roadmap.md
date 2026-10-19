# Evidence status and future work

COLEMAN has a reusable native TCPCI runner and working source, replay, measurement
and evaluation paths. This page distinguishes available capabilities from
scientific conclusions that have not been established. Use the
[documentation map](index.md) to learn the platform and the
[compatibility archive](release-history.md) to investigate individual deliveries.

## Available foundations

| Capability | Where to use it |
| --- | --- |
| Typed native experiments, sweeps and checkpoints | [First experiment](getting-started.md), [configuration](configuration.md) |
| Original-source analytical/native data paths | [Dataset workflows](datasets.md) |
| Controlled Java/Python observations and a constructed C weakness | [Controlled observations](controlled-faults.md) |
| Known-answer difficult CI cases | [Stress laboratory](ci-stress-laboratory.md) |
| Source integrity and production-reality assessment | [Source admission](corpus-admission.md) |
| Comparable candidates and repeated observations | [Baselines](comparable-baselines.md), [corpus evaluation](corpus-evaluation.md) |
| Protected normal or bounded paired comparisons | [Protected evaluation](protected-evaluation.md), [bounded means](bounded-paired-evaluation.md) |
| Time-aware observations and prospective integration studies | [Integration guide](integration-guide.md) |
| Finite selection/allocation contracts and correctness checks | [Coverage](finite-coverage.md), [allocation](scenario-payoffs.md), [benchmarks](decision-benchmarks.md) |

These are implemented capabilities with their own documented contracts and limits.
They are not a declaration that every dataset is scientifically admitted or that
one policy is universally superior.

## Historical corpus status

The diversified corpus phase delivered twelve native subtasks and several source
qualification increments through the verified 4.19.0 release. The maintainer chose
to close [#261](https://github.com/jacksonpradolima/coleman/issues/261#issuecomment-6271942313)
and [#313](https://github.com/jacksonpradolima/coleman/issues/313#issuecomment-6271940884)
as **not planned**, with unmet criteria preserved. This is administrative closure,
not completed historical cross-corpus confirmation.

Four required regimes remain unqualified:

- Small volatile historical CI.
- Medium historical CI.
- Long-running or large-suite historical CI.
- Highly configurable historical systems.

The missing source-specific evidence includes availability clocks, expected
pre-execution test membership, continuity, execution grouping/topology and
independence. A final eligible frozen multi-corpus execution, its raw/per-project/
cross-corpus report and full confirmatory reproduction tutorial were not completed.
Existing installed workflows remain useful at their declared scopes.

The [original-source investigation](https://github.com/jacksonpradolima/coleman/issues/313#issuecomment-6270541363)
recovers numeric build links for 1,578 of 1,662 HikariCP jobs from three verified
TravisTorrent snapshots, with 84 unlinked and version-dependent number labels for
279 jobs. This research is not a released admission contract. The inspected LRTS
processed release did not resolve the requested original metadata/console evidence.
The public [upstream data request](https://github.com/lrtsuser/LRTS/issues/4) records
that dependency; a later response requires renewed verification.

## Conditions for resuming that work

Reopen the existing issues if permitted, version-pinned original evidence resolves
the admission prerequisites, or if the maintainer explicitly approves a revised
scientific scope. Revalidate source meaning, terms and identities before freezing
the candidate grid, split, metrics, inference units and resource protocol. Preserve
negative and inconclusive findings. More conversions, seeds or favorable averages
alone do not satisfy these gates.

## Current improvement priorities

The [maintainability review](maintainability.md) records the architectural,
complexity and docstring findings. The immediate direction is to make working
capabilities easier to learn and compose, preserve bounded domain boundaries,
and improve public contract documentation. A wholesale rewrite is not justified
by the current structural measurements.
