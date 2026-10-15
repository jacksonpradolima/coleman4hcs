# ADR 001: point-in-time observations and decision signals

Status: accepted for the 4.2 integration line; implements the boundary requested in
[#258](https://github.com/jacksonpradolima/coleman/issues/258).

## Context

CI outcomes and durations may arrive late or be corrected. Replacing a stored row
would erase what a policy could know at an earlier decision. The protected
`Timestamp`, `TemporalObservation` and `TemporalPlan` contracts already validate
UTC instants, revisions, groups and temporal eligibility. The native runner already
owns lifecycle, processes, result sinks and checkpoints.

## Decision

Add `coleman.observations` as a transport-independent typed boundary. Immutable
revision histories preserve source availability, first local receipt and explicit
predecessors. As-of selection uses only eligible revisions. Source adapter SDKs,
credentials, collection permissions, clock reconciliation and source-specific
identity rules stay in provider-owned packages implementing `ObservationProvider`.
The base package supplies an offline provider for repository-owned CI fixtures.
No dynamic provider import, network fetch, CLI/MCP dependency or new runner exists.

Reuse protected time/split validation and #251 measurements/provenance. Fit a simple
nonnegative quantity mean only on eligible training observations, frozen before
use. Preserve its evidence and explicit rounding; label outputs as estimates or
priority scores, never calibrated probabilities or inference results. Freshness
uses source event age at the decision and remains an explicit policy parameter.

Use `RunnerExtension.build_environment_fn` for the native example's policy and
monitor integration. Emit typed signals and domain sidecars before the existing
checkpoint advances. Reuse `ArtifactWriter` immutable publication and native
manifest discovery. No second state/checkpoint store is introduced.

## Assumptions and consequences

- The initial source is owned, offline, synthetic CI data. It establishes functional
  behavior, not provider completeness, independence, prospective evidence or value.
- Providers attest identities and complete bounded lineages. First receipt remains
  fixed across duplicate deliveries. UTC source/local clocks must be comparable;
  impossible chronology fails. Existing Python datetime semantics have microsecond
  resolution; providers must declare normalization for finer source clocks.
- Complete single-parent lineages allow out-of-order delivery and timestamp ties.
  Missing parents, forks, conflicting duplicates and changed units/identity fail.
  Truncated histories cannot silently become complete histories.
- Selected input fingerprints exclude future values; whole-source artifact hashes
  remain separate. Later evaluation must still apply the complete protected fold
  grouping protocol; fitting a past view cannot inspect future group memberships.
- Generic signal readers validate metadata/chronology, without executing a producer
  by its name/fingerprint. Mean artifacts rederive training estimates; use the
  frozen model's `verify_signal` when consuming its serialized signals.
- Provider scraping/authentication, signal calibration, sophisticated forecasting,
  streaming retention and external deployment remain outside this issue. Base
  dependencies and native orchestration remain sufficient for the installed demo.

The [human guide](../point-in-time-observations.md) defines the actual v1 API,
limits, compatibility, recovery and complete CI examples.
