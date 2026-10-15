"""Load the owned native example under a stable importable name for checkpoint pickle."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def example():
    name = "coleman_owned_shadow_ci_fixture"
    path = Path(__file__).parents[2] / "examples/prospective-shadow/ci.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)
