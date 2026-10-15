"""Offline operational CI pilot, using native runs and independent event accounting."""

from coleman.pilot.contracts import (
    CREDITS,
    OVERHEAD_SECONDS,
    RATE,
    SECONDS,
    WALL_SECONDS,
    AdoptionThresholds,
    ExecutionEvent,
    PilotBuild,
    PilotManifest,
    PilotRun,
    PilotTest,
)
from coleman.pilot.replay import ReplayMetrics, replay_events, serial_events, verify_run
from coleman.pilot.report import CandidateFinding, PilotReport, publish_pilot_report, render_pilot_report

__all__ = [
    "CandidateFinding",
    "PilotReport",
    "publish_pilot_report",
    "render_pilot_report",
    "CREDITS",
    "OVERHEAD_SECONDS",
    "RATE",
    "SECONDS",
    "WALL_SECONDS",
    "AdoptionThresholds",
    "ExecutionEvent",
    "PilotBuild",
    "PilotManifest",
    "PilotRun",
    "PilotTest",
    "ReplayMetrics",
    "replay_events",
    "serial_events",
    "verify_run",
]
