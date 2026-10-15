# CI operational pilot: synthetic-native-ci-v1

Matrix: completed; missing pairs: 0.

Evidence: synthetic_replay. Source: urn:coleman:fixture:operational-ci-v1.

License: MIT (COLEMAN repository LICENSE). Source digest: sha256:67368dc46ee55b76cee8a0c717b73c0f99158120714fdd9ada7be1c7cce41627.

Frozen manifest: sha256:e52cc5f344820cd6a0243e6f2f7b7e1957d4d55eb310b8ad7bf928ebc9baecff.

Runner work/elapsed values are replay accounting from declared durations; prioritization overhead is separately measured. No queue/setup/fixed cost is claimed saved.

| Candidate | Build | Recall | First failure s | Runner s | Elapsed s | Modeled credits | Overhead s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base-order | 2 | 0.5 | 3.000000000000 | 3.000000000000 | 3.000000000000 | 0.030000000000000000000000 | 0.000251055 |
| base-order | 3 | 0.5 | 3.000000000000 | 3.000000000000 | 3.000000000000 | 0.030000000000000000000000 | 0.000262023 |
| native-greedy | 2 | 0.5 | 2.000000000000 | 3.000000000000 | 3.000000000000 | 0.030000000000000000000000 | 0.000605822 |
| native-greedy | 3 | 0.5 | 2.000000000000 | 3.000000000000 | 3.000000000000 | 0.030000000000000000000000 | 0.000592709 |
| native-random | 2 | 0.0 | null (censored) | 0 | 0 | 0 | 0.00027895 |
| native-random | 3 | 0.0 | null (censored) | 1.000000000000 | 1.000000000000 | 0.010000000000000000000000 | 0.000245095 |
| shortest-first | 2 | 0.5 | 3.000000000000 | 3.000000000000 | 3.000000000000 | 0.030000000000000000000000 | 0.000251294 |
| shortest-first | 3 | 0.5 | 3.000000000000 | 3.000000000000 | 3.000000000000 | 0.030000000000000000000000 | 0.000242711 |

Candidate native-greedy: descriptive gate **inconclusive**; reasons: synthetic_replay_is_not_operational_evidence. Recall range (0.5, 0.5); runner saving fraction range (0.0, 0.0).

Candidate native-random: descriptive gate **fail**; reasons: detection_regression_or_threshold_failure, synthetic_replay_is_not_operational_evidence. Recall range (0.0, 0.0); runner saving fraction range (0.6666666666666666, 1.0).

Candidate shortest-first: descriptive gate **inconclusive**; reasons: synthetic_replay_is_not_operational_evidence. Recall range (0.5, 0.5); runner saving fraction range (0.0, 0.0).

Adoption: **insufficient evidence**. Commercial demand: **unmeasured**.

Ranges describe observed heterogeneity, not confidence intervals. Repeated seeds do not add independent builds. No IID confidence interval or savings/profitability claim is made.

Before adoption: verify source/coverage availability, representative independent projects or justified temporal blocks, protected comparisons, real end-to-end overhead, setup/queue/retry costs, required suites and conservative full-suite fallback when evidence or inputs are insufficient.
