# Public CI Intelligence roadmap

COLEMAN 4.0 established versioned domain results, protected temporal evaluation,
finite coverage and scenario-payoff decisions, independent solver benchmarks and
a bounded application API. These foundations supplement its native TCPCI runner.
A completed foundation is not evidence that the remaining roadmap is implemented.

**4.1.0** delivers the bounded operational CI
pilot in [#257](https://github.com/jacksonpradolima/coleman/issues/257). Its
claim is auditable accounting and honest evidence, including negative/inconclusive
findings. It does not promise production savings from a synthetic example.

**4.2.0** groups two reviewed public increments: point-in-time observations and
versioned decision signals in [#258](https://github.com/jacksonpradolima/coleman/issues/258),
and the first corpus foundation in [#261](https://github.com/jacksonpradolima/coleman/issues/261).
The foundation includes typed manifests/data cards, two tiny known-answer sources,
metadata inventory and installed native workflows. It does not complete source
admission, corpus diversity or the full benchmark program; all seven required
diversity regimes remain uncovered.

The next compatible feature target is **4.3.0**, starting with #259's exact CI
resource accounting and cost-aware replay, followed sequentially by #260. Further
#261 slices cover full profiles, secure fetch/conversion, verified historical and
controlled-fault sources, stress cases, baselines/statistics and frozen cross-corpus
evidence. Deliveries can span later 4.x releases. The issue remains open until its
admission, diversity, execution and evidence gates pass. Inaccessible or unlicensed
sources remain visible blockers. Security fixes remain independently shippable.

The existing public adapter issues now have consciously revised generic CI
specifications in **M4 — Point-in-time CI adapters and prospective evaluation**:

- [#258](https://github.com/jacksonpradolima/coleman/issues/258): point-in-time CI
  observations and versioned decision signals; implemented for 4.2 through existing
  temporal contracts and native extensions.
- [#259](https://github.com/jacksonpradolima/coleman/issues/259): deterministic CI
  resource accounting and cost-aware replay; depends on #258 and existing
  allocation/evaluation/benchmark foundations.
- [#260](https://github.com/jacksonpradolima/coleman/issues/260): prospective shadow
  evaluation of frozen CI candidates; depends on #259 and protected evaluation.

Implement these sequentially in later 4.x feature releases, with reusable core
contracts and CI-centered fixtures. Provider dependencies stay outside the core.
#259 is implemented on the reviewed 4.3 integration branch. #260 adds frozen
shadow evaluation in the same release; actual prospective observations remain
uncollected. Generic mechanisms, native CI
integration, functional recovery and actual prospective evidence each need their own explicit
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
