# Inspect COLEMAN's quality and security

You should be able to inspect the engineering behind a CI decision platform as
readily as its experiment outputs. COLEMAN exposes test results, advisory audit
receipts, structural limits, documentation checks and signed release artifacts.
Use this page to find that evidence and understand what each check establishes.

## Read the badges

The README links each badge to its evidence. Workflow badges show the latest
matching execution: tests and documentation use pushes to `main`, while lint and
structural checks use pull-request events. A PR badge can therefore describe a
different commit from the latest published release. Check its commit and event
before relying on it for a particular revision.

The **coverage gate 95%** badge describes the configured minimum, not a measured
percentage. Open the test run's log for its actual combined line/branch coverage,
passed tests and skips. The [coverage configuration](https://github.com/jacksonpradolima/coleman/blob/main/pyproject.toml)
defines the threshold, included source and exclusions. Coverage.py evaluates
the gate using its reporting precision; the current configuration uses the
default whole-percent rounding. Inspect the measured result and configuration
when a comparison requires finer precision.

Sonar badges report detected source vulnerabilities and a security rating in the
[COLEMAN dashboard](https://sonarcloud.io/summary/overall?id=jacksonpradolima_coleman4hcs).
The service retains `coleman4hcs` in its project identifier. This is the dashboard
linked by current source-analysis checks; its analysis revision and date identify
the code measured. Dependency advisories have their own audit and are not counted
by the source-vulnerability badge.

The former Codecov dashboard also retains the old identifier, but its public
`main` report is historical. It is not used as a current-coverage badge. Current
coverage qualification uses the pytest coverage gate and the matching Actions log.
Consult [GitHub's badge semantics](https://docs.github.com/en/actions/how-tos/monitor-workflows/add-a-status-badge)
for branch and event selection.

## Tests and coverage

The [test workflow](https://github.com/jacksonpradolima/coleman/actions/workflows/codecov.yml)
runs the Python 3.14 suite with branch measurement and a **configured 95% combined
line/branch coverage gate**. Its frozen environment includes the optional MCP extra
for integration tests. The log retains the exact test count, coverage and skips;
a new test count does not require manually updating the README.

Tests cover native prioritization and source adapters as well as failure modes:

- Versioned result round-trips, golden artifacts, legacy readers and unsupported versions.
- Units, explicit missingness, valid signed values and non-finite rejection.
- Deterministic identities and digests, duplicate handling and idempotent replay.
- Interrupted publication and checkpoint continuation.
- Application/transport import boundaries and optional integration behavior.
- Installed-package decision benchmarks and small, owned CI fixtures.

Examples of the executable contracts live in
[result tests](https://github.com/jacksonpradolima/coleman/tree/main/tests/results),
[artifact tests](https://github.com/jacksonpradolima/coleman/blob/main/tests/test_artifacts.py),
[checkpoint tests](https://github.com/jacksonpradolima/coleman/tree/main/tests/checkpoint)
and [installed benchmark tests](https://github.com/jacksonpradolima/coleman/blob/main/tests/benchmarks/test_installed_benchmarks.py).
Normal tests use owned fixtures without downloading external corpora.

For a development checkout with the contributor environment installed:

```bash
uv run --frozen --no-sync python -m pytest
```

This runs the configured coverage gate too. Follow
[contributor setup](contributing.md) to install the locked test dependencies.
Coverage measures exercised code paths; the
[evaluation guide](evaluation-guide.md) explains the separate evidence needed
for conclusions about prioritization effectiveness.

## Source, dependencies and secrets

Security evidence comes from complementary checks:

| Check | When it runs | What to inspect |
| --- | --- | --- |
| [Complete lock advisory audit](https://github.com/jacksonpradolima/coleman/actions/workflows/security-audit.yml) | Dependency PRs, dependency pushes to `main`, weekly and manual dispatch | Queried package pins, `audit.json`, job conclusion and advisory lookup errors |
| [Dependency review](https://github.com/jacksonpradolima/coleman/actions/workflows/dependency-review.yml) | Pull requests | Changed dependencies, license compliance and known vulnerabilities |
| [SonarCloud](https://sonarcloud.io/summary/overall?id=jacksonpradolima_coleman4hcs) | Connected repository analysis | Revision, date, source findings and the PR's analysis check |
| GitGuardian | Connected PR checks | Secret-detection result for the proposed commit |
| [Dependabot](https://github.com/jacksonpradolima/coleman/blob/main/.github/dependabot.yml) | Configured update schedule | Dependency update PRs and repository security alerts where access permits |

The lock audit includes optional runtime extras, development/documentation tools
and packages selected only on other operating systems. It queries advisories
without installing or executing project dependencies. Known vulnerabilities,
lookup failures and unsupported package sources fail the job. There is no blanket
ignore list. Reproduce it with the complete commands in
[dependency security maintenance](dependency-security.md#reproduce-the-audit).

A zero source-vulnerability count and a clean advisory audit answer different
questions. Read the revision, scope and collection time of each result; new
advisories can require another update. Follow the
[security policy](https://github.com/jacksonpradolima/coleman/blob/main/SECURITY.md)
to report a vulnerability privately.

## Code, documentation and complexity gates

| Gate | Enforced scope | Evidence |
| --- | --- | --- |
| Ruff lint and formatting | Repository Python files under configured rules and exclusions | [Lint workflow](https://github.com/jacksonpradolima/coleman/actions/workflows/pylint.yml) |
| NumPy parameter documentation | The public entry points listed by `make docstrings`; this is a bounded gate | Same lint workflow and [docstring guidance](contributing.md#write-useful-numpy-style-docstrings) |
| Xenon complexity | Maximum individual block C, module B, average A | [Structural workflow](https://github.com/jacksonpradolima/coleman/actions/workflows/code-cost.yml) |
| Radon maintainability | Module maintainability index at least 20; cyclomatic complexity is also reported | Same structural workflow and [measurement guide](code-cost.md) |
| Strict site build | Zensical documentation generation | [Documentation workflow](https://github.com/jacksonpradolima/coleman/actions/workflows/docs.yml) |
| Deterministic documentation bundle | Exact regeneration of `llms.txt` from reviewed Markdown | [Bundle workflow](https://github.com/jacksonpradolima/coleman/actions/workflows/llms-txt.yml) |

The workflow file named `pylint.yml` currently runs **Ruff and the NumPy parameter
gate**. Pylint is a separate local qualification step. The contributor workflow
also uses Pyright and ty, Interrogate for docstring presence, package builds,
outside-checkout examples and manual inspection of rendered documentation.
The [maintainability review](maintainability.md) records the broader docstring
backlog; a passing bounded parameter gate does not certify every public signature.

Reproduce the automated code/documentation gates from the installed development
environment:

```bash
make lint
make format-check
make docstrings
make cost-structural
make docs
uv run --frozen --no-sync python scripts/generate_llms_txt.py
git diff --exit-code -- llms.txt
```

The [contributor validation commands](contributing.md#validate-a-change) cover the
additional local checks. PR title validation and labels support review routing;
the relevant technical checks qualify the proposed change before integration.

## Installed correctness and performance observations

The [decision observations workflow](https://github.com/jacksonpradolima/coleman/actions/workflows/decision-benchmarks.yml)
builds a wheel with locked tooling, installs that wheel and runs outside the
checkout. Choose the `correctness` or `performance` profile on manual dispatch.
The Tuesday schedule executes only when the repository variable
`ENABLE_DECISION_PERFORMANCE` is `true`; it is not an unconditional PR gate.

The workflow retains reports, including available failed-attempt artifacts.
Inspect source revision, environment, resource bounds and the selected profile
before comparing measurements. The [benchmark guide](decision-benchmarks.md)
explains independent oracles, certificates and timing limits. Neither coverage
instrumentation timings nor one fast hosted run establish a performance advantage.

## Inspect a published distribution

The [release workflow](https://github.com/jacksonpradolima/coleman/actions/workflows/release.yml)
uses locked release/build tools and publishes these assets in
[GitHub releases](https://github.com/jacksonpradolima/coleman/releases):

| Asset | What it lets you inspect |
| --- | --- |
| Python wheel and source distribution | The installable package and its corresponding source archive |
| `*.sigstore.json` bundles | Sigstore signature and identity evidence for each distribution |
| `sbom.json` | CycloneDX inventory bound to the complete wheel's SHA-256 |

Read the dependency versions and hashes from **`uv.lock` at the corresponding
release tag**. For example, the installation tutorial's wheel uses
[the lock committed at `v4.19.1`](https://github.com/jacksonpradolima/coleman/blob/v4.19.1/uv.lock).
Choose the tag matching your wheel or source distribution when inspecting a
different release. The lock's repository revision remains directly accessible
independently of the release asset list.

The [distribution inventory contract](dependency-security.md#release-distribution-inventory)
explains the SBOM's scope: it identifies the wheel's package metadata and digest;
unbundled dependencies are audited separately through the lock. Signing binds
distribution bytes to a signing identity. Inspect the expected repository,
workflow identity and release before trusting an artifact.

Start with the [published-wheel installation example](getting-started.md), which
pins the release and SHA-256. Then inspect the corresponding bundles and lock
when your adoption process requires them. Engineering checks, artifact identity
and [scientific evidence](evaluation-guide.md) are all inspectable parts of the
same decision.
