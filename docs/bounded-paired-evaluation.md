# Compare bounded CI metrics without assuming normal differences

Use `BoundedMeanProtocol` when a metric has known limits and your paired
inference units are independent. For example, a correctly defined NAPFD score
and its equally weighted repository-family mean lie in `[0, 1]`. The method
compares paired means, controls the full declared family, and reports benefit,
harm, practical equivalence or insufficient/inconclusive evidence.

Use the pinned published wheel below for this protocol. Old frozen contracts
and reports retain their exact serialized bytes and declared method.

## Run a complete installed native CI example

With Python 3.14+ and uv, from any working directory:

```bash
task_bounded_work="$(mktemp -d /tmp/coleman-bounded-XXXXXX)"
uv venv "$task_bounded_work/installed" --python 3.14
uv pip install --python "$task_bounded_work/installed/bin/python" \
  "https://github.com/jacksonpradolima/coleman/releases/download/v4.19.0/coleman-4.19.0-py3-none-any.whl#sha256=070e21c19fb7e3f2e392bd68706c354ed5ab7f76b86db80f5f1bdc5721b767bd"
curl --fail --location \
  https://raw.githubusercontent.com/jacksonpradolima/coleman/v4.19.0/examples/corpus-evaluation/ci.py \
  --output "$task_bounded_work/ci.py"
cd "$task_bounded_work"

# Check the example bytes before execution.
"$task_bounded_work/installed/bin/python" -c \
  'import hashlib, sys; from pathlib import Path; sys.exit(0 if hashlib.sha256(Path("ci.py").read_bytes()).hexdigest() == "80fba6db84da73e4e006058452e312e10cdbfb2a21c4cfdc15ec0195dd942dcb" else "Example SHA-256 mismatch")' &&
  "$task_bounded_work/installed/bin/python" -I ci.py ./output --bounded &&
  "$task_bounded_work/installed/bin/python" -I ci.py ./output --bounded
```

This authored CI example uses the actual Random/UCB runner, independent rational
metric checks, immutable raw artifacts, native checkpoints and the existing
protected access audit. It retains sixteen build/seed results and aggregates the
two related histories into **one family**, yielding `insufficient_evidence`.
Retry preserves native files and checkpoints; protected reuse remains recorded
and cannot produce a new confirmatory claim. These are mechanism observations,
not independent empirical projects. The default command without `--bounded`
preserves the original Student-t workflow and works with the published 4.17 SDK.

## Freeze the protocol before outcomes

```python
from coleman.evaluation.protected import BoundedMeanProtocol

protocol = BoundedMeanProtocol(
    schema_version=1,
    protocol_id="ci-napfd-family-means-bounded-v1",
    holdout_id="reviewed-ci-corpus-protected-holdout-v1",
    metric_name="verdict_napfd",
    unit="1",
    meaning="verdict-napfd-full-universe-v1:equal-build-seed-mean-binary64-v1",
    direction="maximize",
    independent_unit="reviewed_repository_family",
    dependence="independent",
    outcome_lower=0,
    outcome_upper=1,
    practical_effect=0.02,
    alpha=0.05,
    minimum_units=20,
)
print(protocol.to_json())
```

Pass this typed protocol to the same `freeze(...)` and `evaluate(...)` APIs
described in [protected evaluation](protected-evaluation.md). Preserve complete
candidate configurations, data/provenance identities, temporal plans, the full
hypothesis family and a stable holdout ID. The [repetition contract](corpus-evaluation.md)
continues to aggregate seeds/builds within reviewed repository families before
inference. No second runner, access ledger or result format is introduced.

`outcome_lower` and `outcome_upper` are inclusive limits of **each independent
unit's metric**, in the declared unit/meaning. Freeze limits justified by the
metric definition before access. Do not estimate limits from protected outcomes,
clip offending values, substitute missing values or interpret a ranking/reward
as a calibrated failure probability. Observations outside the frozen range reject;
the failed access remains in the audit. Signed bounded metrics and observed zero
are valid. Any missing pair invalidates that comparison's mean and uncertainty.

## Assumptions and the estimand

The estimand is the average expected paired unit difference. Candidate minus
baseline is the raw effect; minimization reverses its sign for improvement.
The units must be independent, while their distributions may differ. Neither
normality nor a variance estimate is required. Bounds must hold for the random
metric, not merely the particular values seen in a sample.

If both candidate measurements lie in `[L, U]`, each paired difference lies in
`[-S, S]`, where `S = U - L`. For `n` independent units, the two-sided
[Hoeffding bound](https://doi.org/10.1080/01621459.1963.10500830)
is `min(1, 2 exp(-n t² / (2 S²)))` for deviation `t` of the sample mean from its
expectation. The original paper's Theorem 1, inequality (2.3), gives the bounded
independent-sum result; the two-sided statement follows by applying both tails.

Independence is an assumption to justify, not a certificate inferred from IDs.
The new protocol rejects `serial`/`clustered` unit declarations. Aggregating
dependent builds/seeds within a genuinely independent family is supported by
the existing repetition design; inventing additional family names is not.
The inference does not establish a sampling model, empirical source admission,
operational savings or generalization to unseen repositories.

## Multiplicity, practical decisions and conservative arithmetic

The complete frozen family contributes to Holm-adjusted conservative p-values.
Missing hypotheses remain present with adjustment input one. Simultaneous
confidence bounds use Bonferroni `alpha / m`, where `m` is the whole family size;
independence between hypotheses is not required.

For a first eligible protected access:

- Benefit/harm requires corrected evidence and an entire simultaneous interval
  beyond the corresponding practical threshold.
- Practical equivalence requires the whole interval inside the closed frozen
  region `[-practical_effect, practical_effect]`. Rejecting zero is unnecessary.
- Other complete comparisons remain inconclusive. Too few units, incomplete
  pairs or unrepresentable uncertainty retain insufficient evidence explicitly.

The v1 arithmetic profile uses exact rational differences of input binary64
measurements. A positive degree-32 Taylor lower sum for `exp(x)` supplies a
conservative rational upper tail bound; it is **not an exact test p-value**.
A fixed 32-step dyadic search keeps an upper confidence radius, followed by
outward rounding of endpoints/probabilities to binary64. When the bound is too
weak, the full possible parameter range remains the interval. There is no
platform `exp`/`log` approximation or data-dependent precision/stopping choice.
Holm products use rational arithmetic and upward output rounding as well; ordinary
floating multiplication could narrow a conservative probability bound. The public
`holm_adjust(..., outward=True)` selects this profile, while its default remains
the legacy arithmetic. Mean effects round once; nonfinite data reject, and unrepresentable uncertainty
stays null. Finite bounds can be much wider than normal-theory intervals.

The method, estimand, arithmetic profile, metric limits, practical threshold and
family are frozen into identity. Report reads recompute every result and require
identical binary64 bounded-method outputs; a one-ULP narrowed endpoint rejects.
Legacy Student-t reports retain their existing numerical-library tolerance.
Both methods retain signed effects, explicit null uncertainty and #251 provenance.

## Compatibility and limits

`BoundedMeanProtocol` requires its explicit nested schema version 1. Missing/future
versions, unknown arithmetic/method identifiers and contradictory assumptions
reject. Older SDKs reject the new method; they do not reinterpret it as Student-t.
No old artifact is migrated or silently assigned bounds. Existing Student-t
normality requirements, golden fixtures, result meanings and conclusions stay
unchanged. Practical equivalence is an explicit new-method conclusion.

This is a conservative paired-mean alternative. It does not implement rank tests,
bootstrap/block/hierarchical inference, sequential testing or alpha spending.
An analyst must freeze the appropriate method before outcomes; trying methods
until one is favorable is holdout reuse. The existing audit records reuse and
removes confirmatory eligibility. Review each project's raw results, missingness,
effect/uncertainty, runtime and source limitations alongside aggregate comparisons.

## Continue the workflow

Prepare the paired inference units with [repeated observations](corpus-evaluation.md). Use the
[evaluation guide](evaluation-guide.md) to check what claim the resulting uncertainty can support.
