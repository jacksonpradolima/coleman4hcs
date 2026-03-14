"""
coleman4hcs.utils.monitor_params - Data Structure for MonitorCollector Parameters.

This module defines the ``CollectParams`` dataclass, which serves as a container for the
parameters required by the ``MonitorCollector.collect`` method. By grouping these parameters
into a single object, it improves method readability, reduces complexity, and provides a
structured approach to managing data during experiment execution.
"""

from dataclasses import dataclass
from typing import Any, Protocol

from coleman4hcs.evaluation import EvaluationMetric


class ScenarioProviderLike(Protocol):
    """Minimum interface needed by MonitorCollector during collection."""

    name: str
    avail_time_ratio: float


@dataclass
class CollectParams:
    """Dataclass encapsulating parameters required for collecting experiment data.

    Parameters
    ----------
    scenario_provider : ScenarioProviderLike
        Any object satisfying the ``ScenarioProviderLike`` protocol (must
        expose ``name: str`` and ``avail_time_ratio: float``).
    available_time : float
        The time available for scheduling or execution.
    experiment : int
        The experiment identifier.
    t : int
        The specific step or part of the experiment being analyzed.
    policy : str
        The name of the policy being evaluated.
    reward_function : str
        The reward function used by the agent to observe the environment.
    metric : EvaluationMetric
        The evaluation metric containing experiment results.
    total_build_duration : int
        The total duration of the build process.
    prioritization_time : int
        The time spent on prioritizing the test cases.
    rewards : float
        The average reward from the prioritized test set.
    prioritization_order : list
        The order of prioritized test cases.
    variant : str or None
        The variant name (for HCS systems).  ``None`` for non-variant runs.
    """

    scenario_provider: ScenarioProviderLike
    available_time: float
    experiment: int
    t: int
    policy: str
    reward_function: str
    metric: EvaluationMetric
    total_build_duration: int
    prioritization_time: int
    rewards: float
    prioritization_order: list[Any]
    variant: str | None = None
