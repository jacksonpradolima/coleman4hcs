"""Tests for process resource tracking helpers."""

import time

from coleman4hcs.telemetry.resources import ProcessResourceTracker


def test_process_resource_tracker_returns_monotonic_samples():
    tracker = ProcessResourceTracker()

    first = tracker.sample()
    time.sleep(0.01)
    second = tracker.sample()

    assert second.wall_time_seconds >= first.wall_time_seconds
    assert second.cpu_time_seconds >= first.cpu_time_seconds
    assert second.current_rss_mib is None or second.current_rss_mib >= 0
    assert second.peak_rss_mib is None or second.peak_rss_mib >= 0
