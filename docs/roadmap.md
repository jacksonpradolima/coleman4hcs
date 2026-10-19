# What the current evidence supports

Choose a workflow by the question you need to answer. A reproducible source
conversion, a controlled fault observation, a mathematical correctness check and
a historical comparison provide different kinds of evidence.

## What you can use today

| Your question | Available workflow | Interpretation |
| --- | --- | --- |
| Can I execute and inspect a CI prioritization experiment? | [Native experiments, policies, sweeps and checkpoints](experiments.md) | Reproducible execution under your declared inputs and budgets |
| Can I replay original public CI observations? | [LRTS, RTPTorrent and LibSSH data paths](datasets.md) | Source-specific analytical/native projections with explicit grouping, exclusions and qualification limits |
| Can I reproduce a source fault? | [Controlled Java and Python observations](controlled-faults.md) | Behavior under a frozen source, observer and runtime; the constructed C workflow has its own narrower scope |
| Can I test drift, churn or delayed feedback? | [CI stress laboratory](ci-stress-laboratory.md) | Known-answer mechanism checks under authored conditions |
| Can I audit source suitability? | [Source admission](corpus-admission.md) | Byte/provenance checks and explicit scientific eligibility; missing evidence remains visible |
| Can I compare methods? | [Baselines](comparable-baselines.md), [repeated observations](corpus-evaluation.md), [protected comparisons](protected-evaluation.md) | A declared comparison, with valid outcomes including benefit, harm, equivalence and insufficient evidence |
| Can I check operational costs before adoption? | [Operational pilot](operational-pilot.md), [resource replay](resource-replay.md), [shadow evaluation](prospective-shadow.md) | Explicit observed versus modeled costs and declared deployment assumptions |

The [dataset guide](datasets.md) connects each available source to its executable
workflow, original terms and evidence type. The [research guide](research.md)
connects the underlying publications to their original questions and subjects.

## Historical corpus status

A final admitted historical cross-corpus confirmation has not been completed.
The runnable source paths do not yet establish a broad comparison across all of
these historical settings:

| Setting | Evidence needed for a defensible comparison |
| --- | --- |
| Small volatile CI | Pre-decision membership, appearance/removal of tests, outcome availability and independent project/group identity |
| Medium CI | Continuous source coverage, stable testcase identity, feedback timing and comparable budgets |
| Long-running or large suites | Original execution grouping, late results, expected test inventory and observed versus modeled physical costs |
| Highly configurable software | Variant membership, shared executions, sparse cells, attempt identity and independent-family accounting |

A processed dataset's build number alone does not establish a source clock or
prove which tests were known before execution. A complete archive can still lack
those facts. The [RTPTorrent](rtptorrent.md) and [LRTS](lrts.md) source assessments
explain the specific boundaries; the [LibSSH workflow](hcs-parquet-etl.md) retains
variant evidence and [original CTest correspondence](hcs-trace-evidence.md).

No final eligible frozen multi-corpus execution, complete confirmatory report and
end-to-end confirmatory reproduction tutorial are currently established. The
existing installed replay and evaluation examples remain usable at their declared
scopes. Controlled and synthetic observations cannot fill missing historical facts.

## What a stronger comparison would require

1. **Verify source meaning.** Retain original bytes, component terms, units,
   grouping, expected test membership and information availability.
2. **Establish eligible observations.** Apply source-specific admission and
   [point-in-time selection](point-in-time-observations.md); preserve exclusions.
3. **Freeze a comparable experiment.** Fix candidate configurations, data/splits,
   metrics, budget/resource protocol and the full hypothesis family before access.
4. **Use the correct inference unit.** Aggregate dependent builds, variants and
   seeds within the reviewed family; require justified independent units.
5. **Publish the evidence.** Retain raw records, per-project/family results,
   comparison reports, runtime/resource observations and reproduction commands.

More seeds do not create additional independent projects. A favorable average
on one subject does not establish a generally better policy. Negative and
inconclusive findings remain valid outcomes, including a conservative fallback
when the required evidence is unavailable.

## Engineering improvements with a clear boundary

The [maintainability review](maintainability.md) records measured complexity,
public-contract documentation and identity/continuation priorities. The direction
is to make available capabilities easier to understand and compose while
preserving the reusable native engine and its compatibility contracts.
