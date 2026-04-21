"""Abstract environment contract for extensible environment implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod


class AbstractEnvironment(ABC):
    """Base class describing the minimum environment API."""

    @abstractmethod
    def reset(self):
        """Reset state for a new simulation run."""

    @abstractmethod
    def run_single(self, experiment, trials=100, bandit_type=None, restore=True):
        """Execute a single experiment."""

    @abstractmethod
    def run(self, experiments=1, trials=100, bandit_type=None, restore=True):
        """Execute one or more experiments."""
