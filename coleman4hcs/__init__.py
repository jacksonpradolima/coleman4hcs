"""Backward-compatible alias for the ``coleman`` package.

This package re-exports everything from ``coleman`` so that existing code
importing ``coleman4hcs`` continues to work without changes.
"""

from __future__ import annotations

import importlib
import sys

_ALIASED_MODULES = [
    "agent",
    "api",
    "bandit",
    "checkpoint",
    "cli",
    "environment",
    "environment_base",
    "evaluation",
    "exceptions",
    "policy",
    "results",
    "reward",
    "runner",
    "scenarios",
    "spec",
    "statistics",
    "telemetry",
    "utils",
]

for _module_name in _ALIASED_MODULES:
    sys.modules[f"coleman4hcs.{_module_name}"] = importlib.import_module(f"coleman.{_module_name}")
