# Maintainability review and improvement priorities

This review connects architectural measurements to concrete maintenance choices.
It does not certify the absence of defects. The baseline is the published
4.19.0 source; documentation changes do not establish runtime performance gains.

## What the measurements say

| Dimension | Baseline observation | Interpretation |
| --- | --- | --- |
| SDK size | 276 Python modules; 47,408 source lines | A substantial platform needs a navigable public surface and explicit boundaries |
| Cyclomatic complexity | Maximum callable complexity 20; no callable above Radon C | The current Xenon limits pass; some validation/orchestration paths are near the block limit |
| Average callable complexity | Approximately 4.54, Radon A, across 1,763 distinct reported callable blocks | A low average does not remove local hotspots |
| Maintainability index | All modules A; minimum approximately 22.15 | Some modules are close to the MI ≥ 20 floor |
| Docstring presence | Existing coverage gate requires 100% | Presence alone does not verify parameter or semantic documentation |
| NumPy parameter coverage | 495 of 622 audited public-named signatures had missing parameter entries | A separate documentation backlog was hidden by the presence gate |

The parameter audit counts explicitly defined public functions/methods and
constructors in public classes. Constructor documentation can live on the class.
It excludes function-local/private definitions, does not decide API stability,
and does not inspect generated Pydantic constructor fields or validate description
accuracy. The counts are signature findings, not a count of independently supported
user-facing APIs.

The documentation revision completes **32 audited signatures across seven entry-point
modules**, including experiment execution, artifact publication, domain records,
source materialization and existing spec I/O/identity/sweep functions. The same
repository-wide audit now reports **472 signatures with missing entries**.
Class field descriptions, units, errors, side effects and limits in the edited
contracts were reviewed manually. The global backlog remains explicit; this is
not a claim of complete NumPy documentation across the SDK.

## Architecture: retain the engine, clarify the surface

The [architecture guide](setup.md) maps the current boundaries. The native runner,
RunSpec, results sinks, source cache, artifacts and checkpoints already provide
the reusable mechanisms. Existing domain/import-boundary tests protect separation
from CLI/MCP and transport dependencies.

A wholesale rewrite is not justified by the current structural evidence. The
higher-value redesign is the reader and integration surface: connected task guides,
a complete installed example, a coherent API index and honest evidence status.

The core identity/version contracts are compatibility-sensitive. Avoid moving
large numbers of classes simply to make the folder tree look uniform. Choose a
small internal boundary when an actual behavioral change requires it, preserve
legacy readers/goldens and check import boundaries and installed behavior.

## Hotspots worth understanding

| Location | Measured callable complexity | Maintenance direction |
| --- | ---: | --- |
| `application/baseline_observations.py::_check_native` | 20 | Preserve independent native-result checks; isolate an invariant only when its semantics and error ordering remain clear |
| `resources/replay.py::ResourceReplayRun.accounting` | 20 | Keep resource categories and conservation checks visible; do not collapse unknown costs into zero |
| `runner.py::_run_experiment_impl` | 19 | Preserve one orchestration path; future extraction should use an actual lifecycle boundary |
| `allocation/results.py::choose_completed` | 19 | Retain completion, identity and conflict checks when simplifying selection logic |
| `corpus/profiles.py` | Lowest baseline module MI, approximately 22.15 | Review future profile additions for growth and repeated column interpretation |

These are review priorities, not confirmed defects or a mandate to refactor now.
Splitting a conditional can lower a score while making an invariant harder to
verify. Require unchanged behavior, focused regression evidence and a clearer
contract before accepting such a refactor.

## Legacy input and continuation responsibilities

The native run ID fingerprints configuration, not all external input bytes. Direct
callers must keep the history fixed when resuming and retain an input identity;
source adapters already bind their declared source/projection identities. The
legacy loader also normalizes a missing numeric duration to zero and does not
carry a machine-readable unit beside that column. The [input guide](input-data.md)
explains why callers must validate those meanings before native projection.

Prefer verified adapters and the existing typed source/measurement contracts.
Any stronger mandatory native-input or checkpoint identity check needs an explicit
compatibility design, rather than silently changing old run IDs or reinterpreting
missing legacy values. This is a focused boundary to improve, not a reason to
replace the runner.

## A targeted architectural opportunity

The shared `spec.producer_identity` deliberately hashes the installed version and
all SDK resources. This conservative closure means unrelated SDK docstrings also
change capture producer identity. The narrower HCS trace producer instead lists
its reviewed implementation dependencies explicitly.

A future, versioned producer-identity design could reduce unrelated invalidation
by binding a reviewed dependency closure for each producer. That requires proof
that every implementation input is included, new producer generations and preserved
old validators; simply dropping files from an existing digest would weaken its
meaning. This review records the opportunity without changing old receipt identity
or introducing an unvalidated fingerprint migration.

The installed legacy LRTS check demonstrates this boundary: the retained golden
receipt still round-trips exactly, every field except the new producer pin agrees,
and actual table bytes/rows validate against the original receipt. A fresh export
correctly has a different implementation pin after source docstring edits. The
original fixture is unchanged; complete-receipt byte equality across changed
producer source would assert the wrong contract.

## Remaining documentation work

1. Extend the parameter-documentation ratchet to additional public boundaries as
   they are reviewed. Prioritize observation/evaluation, selection/allocation,
   application services and supported policy constructors.
2. Review generated-model field descriptions and return/exception contracts;
   the signature audit alone does not cover them.
3. Keep source-specific qualification limitations in the dataset/evidence paths;
   do not replace them with release announcements or generic promises.
4. Recheck older examples against an installed distribution when changing their
   configuration or producer dependencies. A successful site build does not run
   every example or verify every external citation.

These findings are retained here rather than opening replacement issues merely
to move work around. The unfinished historical corpus program is recorded
separately in [evidence status](roadmap.md).

## Reproduce the checks

From a locked development checkout:

```bash
make cost-structural
make interrogate
make docstrings
uv run --frozen --no-sync python scripts/audit_docstrings.py > /tmp/coleman-docstring-audit.json
make typecheck
make docs
```

`make docstrings` is a strict parameter-coverage gate for the seven reviewed
entry-point modules. The broader JSON report is an audit with visible findings,
not a passing global completeness gate. Add `--check` and explicit paths to
validate another reviewed boundary.

For runtime profiling and resource measurement, follow [code-cost evaluation](code-cost.md).
Coverage, instrumentation and structural scores are different forms of evidence;
none substitutes for observed performance under a declared workload.
