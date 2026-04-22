"""Tests for the ``coleman`` backward-compatibility namespace."""

import importlib


def test_coleman_namespace_aliases_new_modules():
    """Coleman compat namespace should expose the same objects as coleman."""
    new_env_module = importlib.import_module("coleman.environment")
    legacy_env_module = importlib.import_module("coleman.environment")
    new_api_module = importlib.import_module("coleman.api")
    legacy_api_module = importlib.import_module("coleman.api")
    new_runner_module = importlib.import_module("coleman.runner")
    legacy_runner_module = importlib.import_module("coleman.runner")
    new_scenarios_module = importlib.import_module("coleman.scenarios")
    legacy_scenarios_module = importlib.import_module("coleman.scenarios")

    assert new_env_module.Environment is legacy_env_module.Environment
    assert new_api_module.run is legacy_api_module.run
    assert new_runner_module.get_scenario_provider is legacy_runner_module.get_scenario_provider
    assert (
        new_scenarios_module.IndustrialDatasetScenarioProvider
        is legacy_scenarios_module.IndustrialDatasetScenarioProvider
    )
