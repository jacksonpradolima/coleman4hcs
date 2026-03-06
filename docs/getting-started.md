# Getting Started

## Installation

### Prerequisites

- Python 3.14+
- [UV](https://docs.astral.sh/uv/)

### Install

```bash
git clone https://github.com/jacksonpradolima/coleman4hcs.git
cd coleman4hcs
uv sync
uv pip install -e .
```

### Development Setup

```bash
# Install all development dependencies
make install

# Install pre-commit hooks
make pre-commit-install
```

## Configuration

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Edit `.env` and set `CONFIG_FILE=./config.toml`.

3. Customise `config.toml` to select datasets, policies, and reward functions.

## Running Experiments

```bash
uv run python main.py
```

See the [README](https://github.com/jacksonpradolima/coleman4hcs#readme) for
detailed usage instructions covering HCS strategies, contextual bandits, and
dataset preparation.
