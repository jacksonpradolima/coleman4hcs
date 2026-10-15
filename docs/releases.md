# Release preparation

COLEMAN keeps public product documentation centered on CI test prioritization,
selection and evidence. Generic decision contracts support that product; a release
should explain the value to CI users before listing internal implementation work.

## Versioning and integration

COLEMAN 4.2 groups reviewed corpus and point-in-time observation increments.
The next compatible feature target is **4.3.0**; compatible corrections use patch
versions. Follow the [public roadmap](roadmap.md) in dependency order and keep each PR runnable and independently reviewed.

A release branch can collect related reviewed PRs when a capability needs several
vertical slices before publication. Keep it synchronized with security fixes from
`main`; available security corrections must not wait for a feature release.
The publication workflow runs on `main`. A compatible `feat:` merge requests the
next minor version; `fix:` requests a patch. Reserve `feat!:` or an explicit
breaking-change declaration for an actual incompatible public contract.

Before integrating a release branch, review its entire diff and pass the full
quality/security/install/native-CI gates. Do not manually edit only the package
version while leaving tags or the lock inconsistent.
The automated workflow updates the package version, synchronizes `uv.lock`, builds
the distributions, and includes the lock in the version commit before tagging.
A second lock check after tagging must find no changes. This keeps the tag, source
archive and wheel metadata consistent.

## Reviewed release descriptions

Before the final release PR, commit `docs/releases/<version>.md` with an en-US release
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
