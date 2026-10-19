# Security policy

## Supported versions

The latest stable release is supported for security fixes. Older pinned wheels
remain useful for reproducing an experiment, but that does not imply ongoing
security support. Published artifacts and versions are listed in
[GitHub releases](https://github.com/jacksonpradolima/coleman/releases).

## Report a vulnerability privately

Please do not open a public issue for a vulnerability. Email
[jacksonpradolima@gmail.com](mailto:jacksonpradolima@gmail.com) with a description,
minimal reproduction, affected version and potential impact. Keep credentials
and private datasets out of public discussions.

We will acknowledge receipt within **48 hours** and aim to provide a fix or
mitigation plan within **7 days** of confirmation. We follow coordinated
disclosure; security fixes are published as patch releases, with a GitHub advisory
once the fix is available.

## Scope and dependency maintenance

This policy covers the `coleman` Python package and dependencies distributed
through this repository. The committed `uv.lock` pins the dependency graph;
optional capture runtimes and third-party source/data components have their own
explicit identities and terms.

[Dependency security maintenance](docs/dependency-security.md) explains lock-wide
auditing, optional extras, platform markers and update verification. Use current
[security alerts](https://github.com/jacksonpradolima/coleman/security) and audit
results to assess advisories. A static dependency table is not a lasting claim
that a version has no vulnerabilities.

YAML configuration uses `yaml.safe_load()`. Native checkpoints use Python pickle
and must come from a source you trust; they are executable state, unlike JSON
measurement records. Source integrity checks bind bytes and declared identities;
they do not themselves grant permission to execute third-party code.

For ordinary bugs and documentation improvements, use the public issue templates
and [contributor guide](CONTRIBUTING.md).
