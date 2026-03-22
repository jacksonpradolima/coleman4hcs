#!/usr/bin/env bash
# =============================================================================
# DevContainer post-start hook for Coleman4HCS
# =============================================================================
# Runs every time the container starts (not just on creation).
# Brings up the observability stack and ensures telemetry is enabled in the
# active config so that `uv run python main.py` sends metrics to Grafana
# without any manual steps.
# =============================================================================
set -e

# Ensure uv is on PATH (installed by post-create.sh)
export PATH="$HOME/.local/bin:$PATH"

# ── 1. Start the observability stack (OTel + Prometheus + Grafana + ClickHouse) ──
#    Uses Docker-in-Docker.  If the containers are already running this is a
#    no-op thanks to `--wait`.
echo "── Starting observability stack ──"
if command -v docker >/dev/null 2>&1; then
  (cd examples/observability && docker compose --profile clickhouse up -d --wait) || \
    echo "⚠  Could not start observability stack. Check Docker status with 'docker ps' or restart the container."
else
  echo "⚠  Docker not available — skipping observability stack."
fi

# ── 2. Ensure telemetry is enabled in config.toml ────────────────────────
#    The Python regex is anchored to the [telemetry] section header so it
#    will not affect [results] or [checkpoint] sections.
if [ -f config.toml ]; then
  python3 -c "
import re, pathlib

cfg = pathlib.Path('config.toml')
text = cfg.read_text()

# Find the [telemetry] section and flip enabled = false -> true
new = re.sub(
    r'(\[telemetry\][^\[]*?)enabled\s*=\s*false',
    r'\1enabled = true',
    text,
    count=1,
    flags=re.DOTALL,
)
if new != text:
    cfg.write_text(new)
    print('Telemetry enabled in config.toml for DevContainer.')
else:
    print('Telemetry already enabled.')
" 2>/dev/null || echo "⚠  Could not auto-enable telemetry in config.toml. The file may be missing or malformed — check [telemetry] section."
fi

echo ""
echo "✅  DevContainer started — observability is live."
echo "  Grafana           → http://localhost:3000"
echo "  Prometheus        → http://localhost:9090"
echo "  Prometheus metrics → http://localhost:8889/metrics"
echo "  ClickHouse HTTP   → http://localhost:8123"
echo ""
