"""
`CollectParams` - A Data Structure for Encapsulating MonitorCollector Parameters

This dataclass serves as a container for the parameters required by the
`MonitorCollector.collect` method. By grouping these parameters into a
single object, it improves method readability, reduces complexity, and
provides a structured approach to managing data during experiment execution.

Attributes:
    scenario_provider (Union[VirtualScenario, VirtualHCSScenario, VirtualContextScenario,
                             IndustrialDatasetScenarioProvider, IndustrialDatasetHCSScenarioProvider,
                             IndustrialDatasetContextScenarioProvider]):
        The scenario being analyzed, which provides context or settings for the experiment. The
        scenario_provider can be one of the following classes:
        - `VirtualScenario`: Basic virtual scenario to manipulate data for each commit.
        - `VirtualHCSScenario`: Extends `VirtualScenario` to handle HCS context.
        - `VirtualContextScenario`: Extends `VirtualScenario` to handle context information.
        - `IndustrialDatasetScenarioProvider`: Provider to process CSV files for experiments.
        - `IndustrialDatasetHCSScenarioProvider`: Extends `IndustrialDatasetScenarioProvider` to handle HCS scenarios.
        - `IndustrialDatasetContextScenarioProvider`: Extends `IndustrialDatasetScenarioProvider` to handle context scenarios.
    available_time (float): The available time ratio or duration for the analysis step.
    experiment (int): Experiment number, used for tracking and identification.
    t (int): Part number (Build) from the scenario that is being analyzed.
    policy (str): Policy name that is evaluating a specific part of the scenario.
    reward_function (str): Reward function used by the agent to observe the environment.
    metric (EvaluationMetric): The result (metric) of the analysis, encapsulating experiment results.
    total_build_duration (float): Total build duration, representing the overall time for the build process.
    prioritization_time (float): Prioritization time, indicating the time spent on prioritization.
    rewards (float): Average reward from the prioritized test set, indicating the effectiveness of the prioritization.
    prioritization_order (list): Prioritized test set, showing the order in which tests are executed.

Usage:
    Initialize an instance of `CollectParams` with the required attributes
    and pass it to the `MonitorCollector.collect` method to ensure a cleaner
    and more maintainable approach to parameter handling.
"""

from dataclasses import dataclass
from typing import List, Union
from coleman4hcs.evaluation import EvaluationMetric
from coleman4hcs.scenarios import VirtualScenario, VirtualHCSScenario, VirtualContextScenario, \
    IndustrialDatasetScenarioProvider, IndustrialDatasetHCSScenarioProvider, IndustrialDatasetContextScenarioProvider


@dataclass
class CollectParams:
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
