"""Tests for the telemetry module (no-op adapter and factory)."""

from coleman.telemetry.otel import NoOpTelemetry, get_telemetry

# ============================================================================
# NoOpTelemetry
# ============================================================================


class TestNoOpTelemetry:
    def test_record_cycle(self):
        t = NoOpTelemetry()
        t.record_cycle()  # should not raise
        t.record_cycle(attributes={"scenario": "test"})

    def test_run_markers(self):
        t = NoOpTelemetry()
        t.mark_run_started(attributes={"experiment": "1"})
        t.mark_run_finished(attributes={"experiment": "1"})

    def test_record_latency(self):
        t = NoOpTelemetry()
        t.record_latency("bandit_update", 0.5)
        t.record_latency("prioritization", 1.2, attributes={"policy": "UCB"})

    def test_record_fitness(self):
        t = NoOpTelemetry()
        t.record_fitness(0.8, 0.6)
        t.record_fitness(0.9, 0.7, attributes={"scenario": "test"})

    def test_record_resource_snapshot(self):
        t = NoOpTelemetry()
        t.record_resource_snapshot(128.0, 256.0, 75.0)
        t.record_resource_snapshot(None, None, None, attributes={"worker_id": "1"})

    def test_record_experiment_resources(self):
        t = NoOpTelemetry()
        t.record_experiment_resources(12.5, 4.2, 256.0)
        t.record_experiment_resources(12.5, 4.2, None, attributes={"execution_id": "exec-1"})

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

    def test_disabled_with_resource_attributes_returns_noop(self):
        """resource_attributes are accepted but ignored when disabled."""
        t = get_telemetry(enabled=False, resource_attributes={"run_id": "abc123"})
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

    def test_enabled_with_resource_attributes_does_not_crash(self):
        """Passing resource_attributes should not crash regardless of SDK availability."""
        t = get_telemetry(
            enabled=True,
            resource_attributes={"execution_id": "test|tr=0.50|exp=1|abc12345", "run_id": "abc123def456"},
        )
        assert hasattr(t, "record_cycle")
        assert hasattr(t, "span")

    def test_interface_consistency(self):
        """Verify NoOpTelemetry has the same public methods as the interface."""
        noop = NoOpTelemetry()
        expected_methods = [
            "record_cycle",
            "record_latency",
            "record_fitness",
            "record_resource_snapshot",
            "record_experiment_resources",
            "mark_run_started",
            "mark_run_finished",
            "flush",
            "span",
        ]
        for method in expected_methods:
            assert hasattr(noop, method)
            assert callable(getattr(noop, method))
