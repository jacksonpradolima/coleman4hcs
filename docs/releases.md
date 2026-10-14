# Release preparation

COLEMAN keeps public product documentation centered on CI test prioritization,
selection and evidence. Generic decision contracts support that product; a release
should explain the value to CI users before listing internal implementation work.

## The 4.0 integration line

Prepare the next feature release in `release/4.0`, using small reviewed PRs in
prerequisite order. This branch collects completed capabilities without publishing
a new stable release for each PR. Keep it synchronized with reviewed security fixes
from `main`. Do not merge unvalidated parallel implementations to accelerate a
version number.

Security corrections may ship on the supported 3.x line before 4.0. A large feature
release is not a reason to hold an available vulnerability fix.

The release workflow publishes from `main`. When the integration line is complete,
review its entire diff, pass the full quality/security/install/native-CI gates, and
merge its release PR with an explicit major-release Conventional Commit, such as
`feat!: release COLEMAN 4.0`. This intentionally requests the 4.0.0 bump; do not
manually edit only the package version while leaving tags or the lock inconsistent.
The automated workflow updates the version/tag and then synchronizes `uv.lock`.

## Reviewed release descriptions

Before the final release PR, commit `docs/releases/4.0.0.md` with an en-US release
description. The publication workflow uses `docs/releases/<version>.md` when
present; otherwise it preserves the generated description. These notes are part
of the reviewed diff and must describe only completed, validated public features.

Write for a new CI user:

- the concrete problem the release solves and useful workflows;
- the delivered capabilities and a runnable starting example;
- compatibility, Python requirements and any migration steps;
- correctness/security evidence and its limits;
- known limitations and links to the full changes.

Avoid unsupported algorithm superiority, universal generalization or wall-clock
savings claims. Separate modeled objectives, measured CI outcomes, mathematical
correctness and protected empirical evidence. Negative and inconclusive outcomes
remain results.

After automatic publication, check the rendered GitHub description, assets,
checksums, attestations, SBOM, package version and source/lock consistency. If the
automatic description needs an editorial correction, update it from a reviewed
notes file rather than replacing it with an unverified marketing claim.
