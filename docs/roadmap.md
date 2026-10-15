# Public CI Intelligence roadmap

COLEMAN 4.0 established versioned domain results, protected temporal evaluation,
finite coverage and scenario-payoff decisions, independent solver benchmarks and
a bounded application API. These foundations supplement its native TCPCI runner.
A completed foundation is not evidence that the remaining roadmap is implemented.

**4.1.0** delivers the bounded operational CI
pilot in [#257](https://github.com/jacksonpradolima/coleman/issues/257). Its
claim is auditable accounting and honest evidence, including negative/inconclusive
findings. It does not promise production savings from a synthetic example.

The next compatible feature target is **4.2.0**. Group related reviewed corpus
and point-in-time adapter slices on a release branch, preserving independently
shippable security fixes. The diversified corpus in
[#261](https://github.com/jacksonpradolima/coleman/issues/261) follows in sequential,
reviewable vertical slices: manifest/registry and correctness fixtures; existing
data inventory and diversity cards; bounded secure fetch/conversion; admitted
historical and controlled-fault sources; synthetic stress cases; baseline and
protected statistical profiles; frozen cross-corpus evidence. Deliveries can span
4.2 and later 4.x minor releases. The issue remains open until its admission,
diversity, execution and evidence gates are satisfied. An inaccessible or
unlicensed source is a visible blocker, not an invented successful experiment.

The existing public adapter issues now have consciously revised generic CI
specifications in **M4 — Point-in-time CI adapters and prospective evaluation**:

- [#258](https://github.com/jacksonpradolima/coleman/issues/258): point-in-time CI
  observations and versioned decision signals; reuse existing temporal contracts.
- [#259](https://github.com/jacksonpradolima/coleman/issues/259): deterministic CI
  resource accounting and cost-aware replay; depends on #258 and existing
  allocation/evaluation/benchmark foundations.
- [#260](https://github.com/jacksonpradolima/coleman/issues/260): prospective shadow
  evaluation of frozen CI candidates; depends on #259 and protected evaluation.

Implement these sequentially in later 4.x feature releases, with reusable core
contracts and CI-centered fixtures. Provider dependencies stay outside the core.
They are specified, not implemented. Generic mechanisms, native CI integration,
functional recovery and actual prospective evidence each need their own explicit
validation; an offline replay must never masquerade as a forward observation.

Compatible features use subsequent **4.x minor** versions; compatible fixes use
**4.x patch** versions. A major version requires an actual breaking public contract,
not a large issue count. Release branches may group related reviewed PRs where
needed, while security fixes remain independently shippable. Only implemented,
reviewed, tested behavior belongs in published release notes.

Every slice reuses existing runner, results, artifact and checkpoint interfaces;
includes installed-package examples and independent correctness evidence; passes
local review and repository checks; and receives PR review before integration.
No source admission, performance result, prospective observation, independence
assumption or commercial value is inferred from a planned feature.
