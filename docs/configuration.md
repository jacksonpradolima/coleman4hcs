# Configuration

Coleman4HCS is configured through two files:

- `config.toml`: experiment and algorithm settings.
- `.env`: environment variables used by the runtime.

## `config.toml`

Main sections:

- `[execution]`: parallelism and number of independent executions.
- `[experiment]`: datasets, scheduled time ratios, rewards, policies, and output directory.
- `[algorithm.*]`: per-policy hyperparameters.

Use the root `config.toml` as a template and adjust values for your experiment.

## `.env`

Create `.env` from `.env.example` and set at least:

- `CONFIG_FILE`: path to your configuration file (for example, `./config.toml`).

## Example

```bash
cp .env.example .env
# edit .env and set CONFIG_FILE
uv run python main.py
```
