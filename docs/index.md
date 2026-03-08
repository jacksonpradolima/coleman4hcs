# Coleman4HCS

**Multi-Armed Bandit based Test Case Prioritization for Continuous Integration**

Coleman4HCS is a framework that applies Multi-Armed Bandit (MAB) algorithms to
solve the Test Case Prioritization problem in Continuous Integration (CI)
environments.

## Features

- Adaptive learning from test execution feedback
- Multiple MAB policies: Random, Greedy, EpsilonGreedy, UCB, FRRMAB
- Contextual bandits: LinUCB, SWLinUCB
- HCS support with WTS and VTS strategies
- Cost-effective prioritization under time budgets

## Quick Start

```bash
uv sync
uv run python main.py
```

See the [Getting Started](getting-started.md) guide for full instructions.
