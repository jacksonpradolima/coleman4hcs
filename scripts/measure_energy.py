"""Estimate energy and carbon impact for a coleman4hcs experiment run.

This script uses CodeCarbon to track estimated CO2 emissions during
an actual experiment execution.  It serves as a starting point for
comparing implementations through their estimated energy cost.

Usage
-----
.. code-block:: bash

    uv run python scripts/measure_energy.py
"""

from codecarbon import EmissionsTracker

from coleman4hcs.api import load_spec, run


def workload() -> None:
    """Run a coleman4hcs experiment as the measured workload."""
    spec = load_spec("run.yaml")
    result = run(spec)
    print(f"Completed run_id={result.run_id}")


if __name__ == "__main__":
    tracker = EmissionsTracker()
    tracker.start()
    workload()
    emissions: float = tracker.stop()
    print(f"Estimated emissions (kg CO2eq): {emissions}")
