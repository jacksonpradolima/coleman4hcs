"""Read reviewed versioned benchmark profiles shipped with the installed package."""

from importlib.resources import files
from typing import Literal

from coleman.benchmarks.contracts import BenchmarkSuite

Profile = Literal["correctness", "performance"]


def load_suite(profile: Profile = "correctness") -> BenchmarkSuite:
    """Read one bounded package resource, rejecting arbitrary paths and unknown profiles.

    No network, checkout or current-directory dependency exists. Unknown/future
    schema versions and duplicate JSON keys fail through the shared version reader.
    Manifests describe synthetic mathematical cases, not independent real CI data.
    """
    if profile not in {"correctness", "performance"}:
        raise ValueError("Unknown bundled benchmark profile")
    source = files("coleman.benchmarks").joinpath("cases", f"{profile}-v1.json")
    return BenchmarkSuite.from_json(source.read_text(encoding="utf-8"))
