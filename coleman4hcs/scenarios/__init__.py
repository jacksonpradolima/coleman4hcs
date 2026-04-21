"""
coleman4hcs.scenarios - Scenario Management for the Coleman4HCS Framework.

This module provides utilities for managing and processing different scenarios in the context
of the Coleman4HCS framework. This includes virtual scenarios for commits, scenarios specific
to HCS context, and scenarios that consider context information for each commit. The module
also provides utilities to process CSV files for experimental evaluations.

Classes
-------
VirtualScenario
    Basic virtual scenario to manipulate data for each commit.
VirtualHCSScenario
    Extends VirtualScenario to handle HCS context.
VirtualContextScenario
    Extends VirtualScenario to handle context information.
IndustrialDatasetScenarioProvider
    Provider to process CSV files for experiments.
IndustrialDatasetHCSScenarioProvider
    Extends IndustrialDatasetScenarioProvider to handle HCS scenarios.
IndustrialDatasetContextScenarioProvider
    Extends IndustrialDatasetScenarioProvider to handle context scenarios.

Short Aliases
-------------
Scenario
    Alias for VirtualScenario.
HCSScenario
    Alias for VirtualHCSScenario.
ContextScenario
    Alias for VirtualContextScenario.
ScenarioProvider
    Alias for IndustrialDatasetScenarioProvider.
HCSScenarioProvider
    Alias for IndustrialDatasetHCSScenarioProvider.
ContextScenarioProvider
    Alias for IndustrialDatasetContextScenarioProvider.
"""

from .providers import (
    ContextScenarioProvider,
    HCSScenarioProvider,
    IndustrialDatasetContextScenarioProvider,
    IndustrialDatasetHCSScenarioProvider,
    IndustrialDatasetScenarioProvider,
    ScenarioProvider,
)
from .virtual import (
    ContextScenario,
    HCSScenario,
    Scenario,
    VirtualContextScenario,
    VirtualHCSScenario,
    VirtualScenario,
)

__all__ = [
    # Original names (backward-compatible)
    "VirtualScenario",
    "VirtualHCSScenario",
    "VirtualContextScenario",
    "IndustrialDatasetScenarioProvider",
    "IndustrialDatasetHCSScenarioProvider",
    "IndustrialDatasetContextScenarioProvider",
    # Short aliases
    "Scenario",
    "HCSScenario",
    "ContextScenario",
    "ScenarioProvider",
    "HCSScenarioProvider",
    "ContextScenarioProvider",
]

