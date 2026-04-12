"""Estimate energy and carbon impact for a representative workload.

This script uses CodeCarbon to track estimated CO2 emissions during
a simple computational workload.  It serves as a starting point for
comparing implementations through their estimated energy cost.

Usage
-----
.. code-block:: bash

    uv run python scripts/measure_energy.py
"""

from codecarbon import EmissionsTracker


def workload() -> None:
    """Run a CPU-bound workload used for energy estimation."""
    total = 0
    for i in range(5_000_000):
        total += i * i
    print(f"Workload result: {total}")


if __name__ == "__main__":
    tracker = EmissionsTracker()
    tracker.start()
    workload()
    emissions: float = tracker.stop()
    print(f"Estimated emissions (kg CO2eq): {emissions}")
