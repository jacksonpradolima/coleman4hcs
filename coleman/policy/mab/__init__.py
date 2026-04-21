"""Sliding-window Multi-Armed Bandit (MAB) policies."""

from .frrmab import FRRMABPolicy
from .slmab import SlMABPolicy

__all__ = ["FRRMABPolicy", "SlMABPolicy"]
