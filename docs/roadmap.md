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

**4.3.0** groups #259's exact CI resource accounting and cost-aware replay with
#260's frozen shadow evaluation. Both reuse the native runner and checkpoints.
The shadow fixture demonstrates functional recovery, with zero collected forward
builds and insufficient evidence for adoption.

**4.4.0** groups the reviewed next two #261 slices: bounded native profiles and a
reproducible inventory in [#284](https://github.com/jacksonpradolima/coleman/pull/284),
and licensed source-package materialization in
[#285](https://github.com/jacksonpradolima/coleman/pull/285). Typed integrity receipts,
external caches, bounded archive validation and installed native recovery workflows
reuse the existing runner/artifacts/checkpoints. All ten existing native registrations
retain their admission blockers; all seven required diversity regimes remain
uncovered.

**4.5.0** publishes the reviewed RTPTorrent and LRTS source-assessment adapters,
with explicit missing temporal evidence and reproducible admission blockers.

**4.6.0** publishes [controlled fault observations](controlled-faults.md) and
[pinned JUnit3 source capture](junit-source-capture.md), with raw evidence,
interruption recovery and negative outcomes. The published wheel reproduces the
Java pair and reopens its committed observations outside the checkout. This is
source reproduction, not completed benchmark admission.

**4.7.0** publishes the [real LRTS ETL](lrts-parquet-etl.md):
explicit download, pinned nested-member conversion, analytical/native Parquet,
conserved exclusions and actual native replay/checkpoint continuation. Native
execution declares its post-hoc serial assumptions; protected historical
admission and cross-corpus confirmation remain separate unfinished gates.
Later steps cover admitted real-fault matrices, HCS/variant expansion
and synthetic stress, comparable baselines/statistics, frozen confirmatory execution
and the final evidence report. Compatible deliveries continue in later **4.x**
releases. #261 stays open until its admission, diversity, execution and evidence
gates pass. Inaccessible or unlicensed sources remain visible blockers. Security
fixes remain independently shippable.

## Remaining corpus subtasks

The public #261 umbrella has native GitHub sub-issues. #301 and #302 shipped in
4.8. This revision adds #303's [larger LRTS stage views](lrts-stage-views.md);
#304–#313 track the remaining work. Source conversion does not establish admission.

| Stage | Subtasks | Dependency |
| --- | --- | --- |
| RTPTorrent source path — delivered in 4.8 | [#301 ZIP64 integrity](https://github.com/jacksonpradolima/coleman/issues/301), [#302 real Parquet ETL/replay](https://github.com/jacksonpradolima/coleman/issues/302) | #302 requires #301 |
| LRTS source expansion — this revision | [#303 larger histories and explicit stage views](https://github.com/jacksonpradolima/coleman/issues/303) | Existing reviewed main; temporal blockers remain explicit |
| Other source expansion | [#304 Java faults](https://github.com/jacksonpradolima/coleman/issues/304), [#305 non-Java faults](https://github.com/jacksonpradolima/coleman/issues/305), [#306 seeded faults](https://github.com/jacksonpradolima/coleman/issues/306), [#307 HCS](https://github.com/jacksonpradolima/coleman/issues/307), [#308 stress generator](https://github.com/jacksonpradolima/coleman/issues/308) | Existing reviewed main |
| Evidence integration | [#309 admission and production reality](https://github.com/jacksonpradolima/coleman/issues/309) | #302–#308 |
| Comparison protocol | [#310 metric references/statistics](https://github.com/jacksonpradolima/coleman/issues/310), [#311 comparable baselines](https://github.com/jacksonpradolima/coleman/issues/311) | #310 requires #309; #311 requires #309 and #310 |
| Forward feasibility | [#312 prospective source feasibility](https://github.com/jacksonpradolima/coleman/issues/312) | #309 |
| Final execution/report | [#313 confirmatory corpus](https://github.com/jacksonpradolima/coleman/issues/313) | #309–#312 and the complete parent gates |

Work started with #301 and #302, one reviewed slice at a time. Their compatible
source deliveries were collected in a release branch and published together in
**4.8** through [#316](https://github.com/jacksonpradolima/coleman/pull/316).
No release number or subtask closure substitutes for an actual source-integrity,
diversity, temporal-admission or scientific evidence gate.

**4.8.0** delivers #301's reviewed ZIP64 selection and complete RTPTorrent
archive/member integrity through [#314](https://github.com/jacksonpradolima/coleman/pull/314),
plus #302's [full analytical/native Parquet workflow](rtptorrent-parquet-etl.md),
explicit eight-source-job simulation and checkpoint recovery through
[#315](https://github.com/jacksonpradolima/coleman/pull/315). The signed published
wheel was exercised outside checkout with the actual source cache, native runs
and checkpoint continuation. Remaining source admission, diversity and
confirmatory work continues through the public subtasks. #303's new profiles
preserve all raw/processed stages and use exact context/node bindings, while
retaining explicit missing temporal evidence. Remaining expansion begins with #304.

The existing public adapter issues now have consciously revised generic CI
specifications in **M4 — Point-in-time CI adapters and prospective evaluation**:

- [#258](https://github.com/jacksonpradolima/coleman/issues/258): point-in-time CI
  observations and versioned decision signals; implemented for 4.2 through existing
  temporal contracts and native extensions.
- [#259](https://github.com/jacksonpradolima/coleman/issues/259): deterministic CI
  resource accounting and cost-aware replay; implemented for 4.3 on top of #258
  and existing allocation/evaluation/benchmark foundations.
- [#260](https://github.com/jacksonpradolima/coleman/issues/260): prospective shadow
  evaluation of frozen CI candidates; implemented for 4.3 on top of #259 and
  protected evaluation, without fabricating prospective observations.

Continue the remaining public corpus phases sequentially in later 4.x releases,
with reusable core contracts and CI-centered fixtures. Provider dependencies stay
outside the core. Generic mechanisms, native CI integration, functional recovery
and actual prospective evidence each need explicit validation; offline replay
must never masquerade as a forward observation.

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
