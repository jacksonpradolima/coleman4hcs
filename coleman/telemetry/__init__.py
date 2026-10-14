"""
coleman.telemetry - Optional OpenTelemetry Instrumentation.

This subpackage provides optional observability via OpenTelemetry metrics and
traces.  When the ``opentelemetry`` SDK is not installed (or telemetry is
disabled) all instrumentation resolves to lightweight no-op stubs.

Enable the telemetry extra from a COLEMAN GitHub release wheel; see
https://github.com/jacksonpradolima/coleman#installation.
"""

from coleman.telemetry.otel import NoOpTelemetry, Telemetry, get_telemetry

__all__ = ["Telemetry", "NoOpTelemetry", "get_telemetry"]
