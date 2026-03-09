"""Tests for the telemetry module (no-op adapter and factory)."""

from coleman4hcs.telemetry.otel import NoOpTelemetry, get_telemetry

# ============================================================================
# NoOpTelemetry
# ============================================================================


class TestNoOpTelemetry:
    def test_record_cycle(self):
        t = NoOpTelemetry()
        t.record_cycle()  # should not raise
        t.record_cycle(attributes={"scenario": "test"})

    def test_record_latency(self):
        t = NoOpTelemetry()
        t.record_latency("bandit_update", 0.5)
        t.record_latency("prioritization", 1.2, attributes={"policy": "UCB"})

    def test_record_fitness(self):
        t = NoOpTelemetry()
        t.record_fitness(0.8, 0.6)
        t.record_fitness(0.9, 0.7, attributes={"scenario": "test"})

    def test_span_context_manager(self):
        t = NoOpTelemetry()
        with t.span("test_span") as s:
            assert s is None  # NoOp yields None
        with t.span("test_span", attributes={"step": 1}) as s:
            pass  # should not raise


# ============================================================================
# get_telemetry factory
# ============================================================================


class TestGetTelemetry:
    def test_disabled_returns_noop(self):
        t = get_telemetry(enabled=False)
        assert isinstance(t, NoOpTelemetry)

    def test_enabled_without_sdk_returns_noop(self):
        """Even if enabled, without OTel SDK we get NoOp (graceful fallback)."""
        # Since the test environment may or may not have OTel installed,
        # the important thing is that it doesn't crash
        t = get_telemetry(enabled=True)
        # Should be either Telemetry or NoOpTelemetry (both work)
        assert hasattr(t, "record_cycle")
        assert hasattr(t, "record_latency")
        assert hasattr(t, "record_fitness")
        assert hasattr(t, "span")

    def test_interface_consistency(self):
        """Verify NoOpTelemetry has the same public methods as the interface."""
        noop = NoOpTelemetry()
        expected_methods = ["record_cycle", "record_latency", "record_fitness", "span"]
        for method in expected_methods:
            assert hasattr(noop, method)
            assert callable(getattr(noop, method))
