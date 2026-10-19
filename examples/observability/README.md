# Follow experiments with Grafana and ClickHouse

This optional local stack connects COLEMAN's OpenTelemetry export to Prometheus
and a provisioned **Coleman Overview** Grafana dashboard. The ClickHouse profile
adds an analytical server for native results and, with the alternate collector
configuration, telemetry persistence.

Follow the [complete observability guide](../../docs/observability.md) for the
pinned client installation, a runnable CI example, resource units and sampling
limits, SQL, dashboard behavior and sink recovery differences.

Run the following from the repository root with Docker Compose available:

```bash
# Collector + Prometheus + Grafana
docker compose -f examples/observability/docker-compose.yaml up -d

# Alternatively, add ClickHouse and enable its collector exporter
OTEL_COLLECTOR_CONFIG=otel-collector-config-clickhouse.yaml \
  docker compose -f examples/observability/docker-compose.yaml --profile clickhouse up -d
```

Enable `telemetry.enabled: true` in your experiment configuration and point
`telemetry.otlp_endpoint` to `http://localhost:4318`. The `telemetry/local` pack
supplies those settings in a checkout. Run your experiment from the directory
against which its dataset paths are resolved.

Open Grafana at `http://localhost:3000`, Prometheus at `http://localhost:9090`, or
the collector exporter at `http://localhost:8889/metrics`. With the ClickHouse
profile, wait for `http://localhost:8123/ping` to return `Ok.` before use.

The supplied Grafana setup allows anonymous admin access for trusted local use.
The dashboard includes progress, CPU, resident memory, ranking/evaluation latency
and checkpoint activity. Its active-run panels can be empty after an experiment
finishes; use persisted results to inspect completed runs.

Native result rows require `results.sink: clickhouse` independently of telemetry
export. The native runner emits metrics; the telemetry facade supports explicit
trace instrumentation. It does not automatically emit per-step spans or forward
Python logs to the collector.

Stop the services, retaining the named ClickHouse data volume:

```bash
docker compose -f examples/observability/docker-compose.yaml --profile clickhouse down
```

Prometheus has no persistent volume in this example. Keep experiment artifacts
for durable results; the live dashboard is an operational view.
