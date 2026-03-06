# Contributing to Coleman4HCS

Thank you for your interest in contributing to **Coleman4HCS**! This document
provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- [Python 3.14+](https://www.python.org/downloads/)
- [UV](https://docs.astral.sh/uv/) — fast Python package manager
- [GNU Make](https://www.gnu.org/software/make/)

### Setup

```bash
# Clone the repository
git clone https://github.com/jacksonpradolima/coleman4hcs.git
cd coleman4hcs

# Install all dependencies (including dev extras)
make install

# Install pre-commit hooks
make pre-commit-install
```

## Development Workflow

### Running Tests

```bash
make test
```

### Linting and Formatting

```bash
# Run ruff linter and formatter
make lint
make format
```

### Type Checking

```bash
make typecheck
```

### Building Documentation

```bash
make docs
```

## Code Style

- **Formatter**: [Ruff](https://docs.astral.sh/ruff/)
- **Linter**: [Ruff](https://docs.astral.sh/ruff/)
- **Docstrings**: [Numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style — all public modules, classes, and functions must have docstrings.
- **Type annotations**: Encouraged for all public APIs.
- **Line length**: 120 characters.

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/). Commit
messages are enforced by `pre-commit`:

```
feat: add support for new policy
fix: correct NAPFD computation edge case
docs: update installation instructions
test: add tests for sliding window agent
refactor: simplify reward evaluation logic
```

## Pull Request Process

1. **Fork** the repository and create a feature branch from `master`.
2. Make your changes following the code style guidelines above.
3. Add or update tests to cover your changes.
4. Ensure all checks pass: `make check-precommit`.
5. Open a pull request against `master`.
6. A maintainer will review your PR. Please be responsive to feedback.

## Reporting Bugs

Please open an issue on GitHub with:

- A clear title and description.
- Steps to reproduce the bug.
- Expected versus actual behavior.
- Python version and OS information.

## Suggesting Features

Feature requests are welcome! Open an issue describing:

- The problem you want to solve.
- Your proposed solution (if any).
- Any alternatives you considered.

## Security

If you discover a security vulnerability, please follow the process described in
[SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
