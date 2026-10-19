# Choose policies and rewards

A policy determines how accumulated feedback becomes a ranking. A reward defines
the learning signal. Keep the source, feedback visibility and budget fixed when
comparing policies; [comparable baseline matrices](comparable-baselines.md)
provide an explicit candidate-grid workflow.

## Start with simple comparators

Use `Random` as an uninformed ranking comparator and `UCB` as a small adaptive
starting point. Add history-based heuristics through the baseline matrix before
interpreting the value of a more elaborate method. The
[original COLEMAN study](https://doi.org/10.1109/TSE.2020.2992428)
provides the research context for bandit-based TCPCI.

| Family | Examples implemented in COLEMAN | Question to investigate |
| --- | --- | --- |
| Random and greedy | `Random`, `Greedy`, `EpsilonGreedy`, `OptimisticGreedy` | Does adaptation improve on simple rankings under the same feedback? |
| Confidence-based | `UCB`, `UCB1`, `UCB2`, `KLUCB`, `UCBTuned`, `UCBV`, `MOSSUCB` | How does exploration respond to limited observations? |
| Stochastic and adversarial | `ThompsonSampling`, `BayesianUCB`, `Softmax`, `EXP3`, `EXP3IX` | How do different selection mechanisms behave on this history? |
| Changing-history | `SlidingWindowUCB`, `DiscountedUCB`, `ChangeDetectionUCB`, `FRRMAB`, `SlMAB` | What happens when old feedback becomes less relevant? |
| Contextual | `LinUCB`, `SWLinUCB`, `LinTS`, contextual epsilon-greedy variants | Do features available before the decision add useful information? |
| Structured and meta-policy | Combinatorial, dueling/ranking and `PortfolioUCB` policies | Does the additional structure justify its assumptions and cost? |

This is a guide to families, not a claim that each implementation inherits every
theoretical guarantee of the original algorithm. Exact names and constructors
are in the [policy API](api/policy.md); configuration examples are in the
[configuration reference](configuration.md).

## Rewards and quality metrics serve different purposes

`RNFail` and `TimeRank` are native reward choices. RNFail uses failure feedback;
TimeRank additionally incorporates ranking information. Other reward classes
are described in the [reward API](api/reward.md). Check the runner's supported
configuration names before using a class name as a config value.

A reward transforms feedback for learning. NAPFD and APFDc describe prioritization
quality under their own definitions. Neither a reward nor an internal score is a
calibrated failure probability or a statistical comparison.

See [native result interpretation](evaluation-guide.md#read-native-results) and
[versioned measurements](domain-results.md) for the corresponding output contracts.

## Contextual policies

Contextual policies consume configured feature columns, with previous-build
features declared under `contextual_information.config.previous_build`. The
current execution's eventual duration and verdict must not leak into the
pre-decision feature vector. A previous-row transformation alone does not prove
source availability; retained timestamps and source meaning still matter.

[Li et al.'s contextual-bandit paper](https://arxiv.org/abs/1003.0146) introduces
LinUCB in a recommendation setting. Its presence here motivates a policy family;
CI effectiveness needs its own source-appropriate evaluation.

Continue with [point-in-time observations](point-in-time-observations.md),
[protected evaluation](protected-evaluation.md) and the
[research reading path](research.md).

## Extend a policy

Use the native policy/agent interfaces and lifecycle hooks described in
[extensions and parallelism](extensibility.md). Reuse orchestration, sinks and
checkpoints. Document update state, randomness, feedback assumptions and the
conditions under which a score is meaningful.
