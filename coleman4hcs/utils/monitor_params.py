"""
coleman4hcs.utils.monitor_params - Data Structure for MonitorCollector Parameters.

This module defines the ``CollectParams`` dataclass, which serves as a container for the
parameters required by the ``MonitorCollector.collect`` method. By grouping these parameters
into a single object, it improves method readability, reduces complexity, and provides a
structured approach to managing data during experiment execution.
"""

from dataclasses import dataclass
from typing import List, Union
from coleman4hcs.evaluation import EvaluationMetric
from coleman4hcs.scenarios import VirtualScenario, VirtualHCSScenario, VirtualContextScenario, \
    IndustrialDatasetScenarioProvider, IndustrialDatasetHCSScenarioProvider, IndustrialDatasetContextScenarioProvider


@dataclass
class CollectParams:
    """Dataclass encapsulating parameters required for collecting experiment data.

    Parameters
    ----------
    scenario_provider : Union[VirtualScenario, VirtualHCSScenario, VirtualContextScenario, IndustrialDatasetScenarioProvider, IndustrialDatasetHCSScenarioProvider, IndustrialDatasetContextScenarioProvider]
        The scenario being analyzed, which provides context or settings for
        the experiment.
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
    """
    scenario_provider: Union[
        VirtualScenario,
        VirtualHCSScenario,
        VirtualContextScenario,
        IndustrialDatasetScenarioProvider,
        IndustrialDatasetHCSScenarioProvider,
        IndustrialDatasetContextScenarioProvider,
    ]
    available_time: float
    experiment: int
    t: int
    policy: str
    reward_function: str
    metric: EvaluationMetric
    total_build_duration: int
    prioritization_time: int
    rewards: float
    prioritization_order: List[any]
