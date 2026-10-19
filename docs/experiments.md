# Design and run an experiment

An experiment combines a test history, candidate policies, a reward, a budget
and execution settings. Start with a small run you can inspect, then expand one
question at a time.

## Follow the experiment path

```mermaid
flowchart LR
    D[Choose and inspect data] --> S[Resolve a RunSpec]
    S --> R[Run candidates]
    R --> Q[Inspect results and recovery]
    Q --> E[Evaluate the comparison]
```

| Step | Guide | What you decide |
| --- | --- | --- |
| Prepare input | [Native input format](input-data.md), [dataset workflows](datasets.md) | Test identity, build order, units and projection assumptions |
| Choose a method | [Policies and rewards](policies.md) | Candidate families and the feedback each receives |
| Fix the experiment | [Configuration](configuration.md) | Budgets, seeds, packs, output and checkpoint settings |
| Handle variants | [Highly configurable systems](highly-configurable.md) | Whole-set or per-variant state and shared execution groups |
| Run and inspect | [First experiment](getting-started.md), [workflow notebook](workflow.md) | Result paths, queries and continuation |
| Extend behavior | [Extensions and parallelism](extensibility.md) | Hooks or custom policies with the existing runner |
| Compare fairly | [Evaluation guide](evaluation-guide.md) | Source eligibility, pairing and inference units |

## Three choices that affect interpretation

**Budget.** A ratio uses a fraction of the source build's total test duration;
`fixed_time` uses seconds; `subset_size` limits the test count. A ratio derived
from the recorded full suite is a post-hoc simulation assumption. Its feasibility
as a pre-decision operational budget requires separate justification.

**Repetition.** Repeated seeds assess algorithmic randomness for the same source.
Keep build, job, stage and repository-family identity; more seeds do not create
more independent repositories.

**Parallelism.** Runner workers parallelize experiment execution. They do not
turn a serial test-cost simulation into an observed parallel CI execution. See
[resource replay](resource-replay.md) when you need explicit occupancy and topology.

For a runnable first configuration, use [getting started](getting-started.md).
For a fair candidate grid, use [comparable baselines](comparable-baselines.md).
