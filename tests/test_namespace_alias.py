"""Tests for the ``coleman`` compatibility namespace."""

import importlib


def test_coleman_namespace_aliases_legacy_modules():
    """The new namespace should expose the same core modules."""
    new_env_module = importlib.import_module("coleman.environment")
    legacy_env_module = importlib.import_module("coleman4hcs.environment")

    assert new_env_module.Environment is legacy_env_module.Environment
