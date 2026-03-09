"""
coleman4hcs.telemetry.otel - OpenTelemetry Initialization and No-Op Fallback.

When ``opentelemetry`` is installed **and** telemetry is enabled the module
configures a ``Telemetry`` facade that exposes pre-defined metrics and
trace helpers.

When the SDK is absent or telemetry is disabled, ``NoOpTelemetry`` provides
the same interface with near-zero overhead.

Metric names (for documentation / Grafana dashboard authors)
------------------------------------------------------------
* ``coleman.cycles_total``            – counter
* ``coleman.bandit_update_latency``   – histogram (seconds)
* ``coleman.prioritization_latency``  – histogram (seconds)
* ``coleman.evaluation_latency``      – histogram (seconds)
* ``coleman.napfd``                   – histogram
* ``coleman.apfdc``                   – histogram
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try importing OpenTelemetry; fall back gracefully
# ---------------------------------------------------------------------------
_HAS_OTEL = False
try:
    from opentelemetry import metrics as otel_metrics
    from opentelemetry import trace as otel_trace
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _HAS_OTEL = True
except ImportError:  # pragma: no cover – tested via NoOpTelemetry path
    pass


class Telemetry:
    """OpenTelemetry facade for Coleman4HCS.

    Parameters
    ----------
    service_name : str
        OTLP service/resource name.
    otlp_endpoint : str
        OTLP HTTP endpoint.

    Attributes
    ----------
    meter : opentelemetry.metrics.Meter
        OTel meter for creating instruments.
    tracer : opentelemetry.trace.Tracer
        OTel tracer for creating spans.
    """

    def __init__(
        self,
        service_name: str = "coleman4hcs",
        otlp_endpoint: str = "http://localhost:4318",
    ) -> None:
        if not _HAS_OTEL:
            raise ImportError(
                "opentelemetry SDK is required for Telemetry. Install it with: pip install coleman4hcs[telemetry]"
            )

        resource = Resource.create({"service.name": service_name})

        # Metrics
        reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=f"{otlp_endpoint}/v1/metrics"))
        meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
        otel_metrics.set_meter_provider(meter_provider)
        self.meter = meter_provider.get_meter("coleman4hcs")

        # Traces
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{otlp_endpoint}/v1/traces")))
        otel_trace.set_tracer_provider(tracer_provider)
        self.tracer = tracer_provider.get_tracer("coleman4hcs")

        # Pre-create instruments
        self.cycles_total = self.meter.create_counter("coleman.cycles_total", description="Total cycles processed")
        self.bandit_update_latency = self.meter.create_histogram(
            "coleman.bandit_update_latency", unit="s", description="Bandit update latency"
        )
        self.prioritization_latency = self.meter.create_histogram(
            "coleman.prioritization_latency", unit="s", description="Prioritization latency"
        )
        self.evaluation_latency = self.meter.create_histogram(
            "coleman.evaluation_latency", unit="s", description="Evaluation latency"
        )
        self.napfd = self.meter.create_histogram("coleman.napfd", description="NAPFD distribution")
        self.apfdc = self.meter.create_histogram("coleman.apfdc", description="APFDc distribution")

    def record_cycle(self, attributes: dict[str, Any] | None = None) -> None:
        """Increment the cycle counter.

        Parameters
        ----------
        attributes : dict or None
            Optional OTel attributes (avoid high cardinality).
        """
        self.cycles_total.add(1, attributes=attributes)

    def record_latency(self, name: str, duration: float, attributes: dict[str, Any] | None = None) -> None:
        """Record a latency observation.

        Parameters
        ----------
        name : str
            One of ``bandit_update``, ``prioritization``, ``evaluation``.
        duration : float
            Duration in seconds.
        attributes : dict or None
            Optional OTel attributes.
        """
        hist = getattr(self, f"{name}_latency", None)
        if hist is not None:
            hist.record(duration, attributes=attributes)

    def record_fitness(self, fitness: float, cost: float, attributes: dict[str, Any] | None = None) -> None:
        """Record NAPFD/APFDc observations.

        Parameters
        ----------
        fitness : float
            NAPFD value.
        cost : float
            APFDc value.
        attributes : dict or None
            Optional OTel attributes.
        """
        self.napfd.record(fitness, attributes=attributes)
        self.apfdc.record(cost, attributes=attributes)

    @contextlib.contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None):
        """Create a trace span context manager.

        Parameters
        ----------
        name : str
            Span name.
        attributes : dict or None
            Span attributes (step allowed here, unlike metrics).

        Yields
        ------
        opentelemetry.trace.Span
        """
        with self.tracer.start_as_current_span(name, attributes=attributes) as s:
            yield s


class NoOpTelemetry:
    """No-op telemetry stub with the same public interface as ``Telemetry``.

    All methods are instant no-ops.  Used when telemetry is disabled or the
    OpenTelemetry SDK is not installed.
    """

    def record_cycle(self, attributes: dict[str, Any] | None = None) -> None:
        """No-op.

        Parameters
        ----------
        attributes : dict or None
            Ignored.
        """

    def record_latency(self, name: str, duration: float, attributes: dict[str, Any] | None = None) -> None:
        """No-op.

        Parameters
        ----------
        name : str
            Ignored.
        duration : float
            Ignored.
        attributes : dict or None
            Ignored.
        """

    def record_fitness(self, fitness: float, cost: float, attributes: dict[str, Any] | None = None) -> None:
        """No-op.

        Parameters
        ----------
        fitness : float
            Ignored.
        cost : float
            Ignored.
        attributes : dict or None
            Ignored.
        """

    @contextlib.contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None):
        """No-op context manager.

        Parameters
        ----------
        name : str
            Ignored.
        attributes : dict or None
            Ignored.

        Yields
        ------
        None
        """
        yield None


def get_telemetry(
    enabled: bool = False,
    service_name: str = "coleman4hcs",
    otlp_endpoint: str = "http://localhost:4318",
) -> Telemetry | NoOpTelemetry:
    """Return the appropriate telemetry implementation.

    Parameters
    ----------
    enabled : bool
        Whether telemetry is enabled.
    service_name : str
        OTLP service name.
    otlp_endpoint : str
        OTLP endpoint URL.

    Returns
    -------
    Telemetry or NoOpTelemetry
        Active telemetry if enabled and SDK present, else no-op.
    """
    if not enabled:
        return NoOpTelemetry()

    if not _HAS_OTEL:
        logger.warning(
            "Telemetry enabled but opentelemetry SDK not installed. "
            "Install with: pip install coleman4hcs[telemetry]. "
            "Falling back to NoOpTelemetry."
        )
        return NoOpTelemetry()

    try:
        return Telemetry(service_name=service_name, otlp_endpoint=otlp_endpoint)
    except Exception:
        logger.exception("Failed to initialize OpenTelemetry; falling back to NoOpTelemetry")
        return NoOpTelemetry()
