"""Compatibility namespace for the new ``coleman`` package path."""

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

for module_name in _ALIASED_MODULES:
    sys.modules[f"{__name__}.{module_name}"] = importlib.import_module(f"coleman4hcs.{module_name}")
