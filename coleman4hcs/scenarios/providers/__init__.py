"""Industrial dataset scenario providers."""

from .dataset_providers import (
    IndustrialDatasetContextScenarioProvider,
    IndustrialDatasetHCSScenarioProvider,
    IndustrialDatasetScenarioProvider,
)

# Short-name aliases
ScenarioProvider = IndustrialDatasetScenarioProvider
HCSScenarioProvider = IndustrialDatasetHCSScenarioProvider
ContextScenarioProvider = IndustrialDatasetContextScenarioProvider

__all__ = [
    "IndustrialDatasetScenarioProvider",
    "IndustrialDatasetHCSScenarioProvider",
    "IndustrialDatasetContextScenarioProvider",
    "ScenarioProvider",
    "HCSScenarioProvider",
    "ContextScenarioProvider",
]
