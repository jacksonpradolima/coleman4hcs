# Prioritize tests across configurations

A highly configurable system can execute overlapping tests in several variants.
Variants may appear or disappear as the system evolves. COLEMAN supports two
strategies for organizing learning state across those variants.

The strategies originate in the
[SPLC study](https://doi.org/10.1145/3382025.3414967) and the expanded
[Empirical Software Engineering article](https://doi.org/10.1007/s10664-021-10093-3).
The papers describe their evaluated conditions; a new source still needs its
own unit, grouping and availability assessment.

## WTS: whole test set

Prioritize the union of tests represented across variants, using a whole-system
history plus a variant execution table. Enable `hcs_configuration.wts_strategy`:

```yaml
experiment:
  datasets_dir: data
  datasets: [whole-system]
hcs_configuration:
  wts_strategy: true
```

The dataset folder contains `features-engineered.parquet` and
`data-variants.parquet`. The variant file retains a `Variant` field and source
build/test bindings. The whole-system interpretation and cost projection must be
declared; a union is not automatically an additional set of independent executions.

## VTS: variant test sets

Treat each selected variant as a separate history with its own policy state:

```yaml
experiment:
  datasets_dir: data/variants
  datasets: [linux-debug, linux-release]
hcs_configuration:
  wts_strategy: false
```

Each named directory contains a native `features-engineered.parquet` file. Separate
learning state does not establish independent evidence: variants can share commits,
source tests, pipelines and infrastructure.

## Work from original data

The [LibSSH workflow](hcs-parquet-etl.md) produces whole-system and variant-aware
Parquet from pinned original CSV/JSON. It retains sparse pipeline/variant cells,
other-attempt exclusions and missing test evidence. The optional
[CTest corroboration](hcs-trace-evidence.md) binds reported process outcomes and
exact decimal seconds back to original logs and archived observations.

The Dune examples in the checkout remain useful for exploring the layouts. Their
source qualification limits stay explicit in the [dataset guide](datasets.md).

## Evaluate the strategy

Use equal budgets and source bindings when comparing WTS and VTS. Retain shared
pipeline/build groups and variant identity in the raw results. Distinguish
serial test-work cost from observed physical execution, and do not count
configuration variants as additional independent repositories.

Continue with [comparable baselines](comparable-baselines.md),
[source admission](corpus-admission.md) and [repeated observations](corpus-evaluation.md).
