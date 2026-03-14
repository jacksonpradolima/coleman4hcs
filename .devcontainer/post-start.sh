#!/usr/bin/env bash
# =============================================================================
# DevContainer post-start hook for Coleman4HCS
# =============================================================================
# Runs every time the container starts (not just on creation).
# Brings up the observability stack and ensures telemetry is enabled in the
# active config so that `uv run python main.py` sends metrics to Grafana
# without any manual steps.
# =============================================================================
set -euo pipefail

# ── 1. Start the observability stack (OTel Collector + Grafana) ──────────
#    Uses Docker-in-Docker.  If the containers are already running this is a
#    no-op thanks to `--wait`.
echo "── Starting observability stack ──"
if command -v docker >/dev/null 2>&1; then
  (cd examples/observability && docker compose up -d --wait) || \
    echo "⚠  Could not start observability stack (Docker may not be ready yet)."
else
  echo "⚠  Docker not available — skipping observability stack."
fi

# ── 2. Ensure telemetry is enabled in config.toml ────────────────────────
#    If the DevContainer overlay exists and config.toml still has
#    `enabled = false` for telemetry, patch it in-place.
if [ -f .devcontainer/config.devcontainer.toml ] && [ -f config.toml ]; then
  if grep -q 'enabled = false' config.toml 2>/dev/null; then
    # Only patch the [telemetry] section, not [results] or [checkpoint]
    if python3 -c "
import re, pathlib

cfg = pathlib.Path('config.toml')
text = cfg.read_text()

# Find the [telemetry] section and flip enabled = false → true
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
" 2>/dev/null; then
      :
    else
      echo "⚠  Could not auto-enable telemetry in config.toml."
    fi
  fi
fi

echo ""
echo "✅  DevContainer started — observability is live."
echo "  Grafana          → http://localhost:3000"
echo "  Prometheus metrics → http://localhost:8889/metrics"
echo ""
