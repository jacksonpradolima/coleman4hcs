# Code Cost Evaluation

Coleman4HCS treats code cost as a **multi-dimensional scorecard**, not a
single number.  This guide explains the tooling and commands available for
evaluating four complementary cost dimensions.

## Dimensions

| Dimension | What it measures | Primary tools |
|-----------|-----------------|---------------|
| **Structural cost** | Maintainability, complexity, and change risk | Radon, Xenon, Wily |
| **Runtime cost** | CPU time, execution hotspots, and memory pressure | Scalene, py-spy |
| **Energy cost** | Estimated energy or carbon impact | CodeCarbon, pyRAPL |
| **Operational cost proxy** | Basis for translating hotspots into infrastructure effort | All of the above |

---

## Quick start

Install all cost evaluation tools together with the rest of the dev
dependencies:

```bash
uv sync --extra dev
```

---

## 1 — Structural cost

### Cyclomatic complexity (Radon CC)

```bash
uv run radon cc -s -a coleman4hcs/
# or
make cost-complexity
```

### Maintainability index (Radon MI)

```bash
uv run radon mi -s coleman4hcs/
# or
make cost-maintainability
```

### Raw metrics (Radon raw)

```bash
uv run radon raw coleman4hcs/
```

### Xenon complexity gate

Xenon wraps Radon and returns a non-zero exit code when complexity
thresholds are exceeded.  This is the first CI guardrail:

```bash
uv run xenon --max-absolute C --max-modules B --max-average A coleman4hcs/
# or
make cost-xenon
```

**Current CI thresholds:**

| Parameter | Threshold | Meaning |
|-----------|-----------|---------|
| `--max-absolute` | **C** | No single block worse than C |
| `--max-modules` | **B** | No module average worse than B |
| `--max-average` | **A** | Project-wide average must be A |

!!! note "Target thresholds"
    The long-term target is `--max-absolute B --max-modules A --max-average A`.
    Current thresholds are relaxed to match the existing codebase and should
    be tightened as complexity hotspots are refactored.

### Run all structural checks at once

```bash
make cost-structural
```

### Trend analysis with Wily

[Wily](https://wily.readthedocs.io/) tracks complexity metrics across
Git history.  It is included as a dev dependency:

```bash
uv run wily build coleman4hcs/
uv run wily report coleman4hcs/
uv run wily diff coleman4hcs/ -r HEAD^1
```

---

## 2 — Runtime cost

### Scalene (CPU + memory profiling)

[Scalene](https://github.com/plasma-umass/scalene) profiles CPU time,
memory allocation, and copy volume line-by-line:

```bash
uv run scalene -m coleman4hcs.cli --help
# or
make cost-profile-scalene
```

Replace `--help` with a real workload for meaningful results.  For
example, profile an actual experiment run:

```bash
uv run scalene -m coleman4hcs.cli run --config run.yaml
```

### py-spy (sampling profiler)

[py-spy](https://github.com/benfred/py-spy) is a sampling profiler that
attaches to a running process with minimal overhead:

```bash
uv run py-spy top -- python -m coleman4hcs.cli --help
```

!!! note
    `py-spy` may require elevated privileges (`sudo`) on some systems.

---

## 3 — Energy cost

### CodeCarbon

[CodeCarbon](https://mlco2.github.io/codecarbon/) estimates the carbon
emissions of compute workloads.  A minimal example lives at
`scripts/measure_energy.py`:

```bash
uv run python scripts/measure_energy.py
# or
make cost-energy
```

You can adapt the script to wrap any representative workload:

```python
from codecarbon import EmissionsTracker

tracker = EmissionsTracker()
tracker.start()
# ... your workload here ...
emissions = tracker.stop()
print(f"Estimated emissions (kg CO2eq): {emissions}")
```

### pyRAPL (Intel hardware only)

[pyRAPL](https://github.com/powerapi-ng/pyRAPL) reads Intel RAPL
counters for direct energy measurement.  It is included as a dev
dependency but only works on supported Intel hardware:

```bash
uv run python -c "import pyRAPL; pyRAPL.setup(); print('pyRAPL available')"
```

!!! warning "Platform limitation"
    pyRAPL requires an Intel CPU with RAPL support **and** read access to
    `/sys/class/powercap/intel-rapl/`.  It does not work on AMD, ARM, or
    virtualised environments without RAPL passthrough.

---

## 4 — CI integration

The GitHub Actions workflow **Code Cost — Structural checks**
(`.github/workflows/code-cost.yml`) runs automatically on every pull
request and enforces two gates:

1. **Xenon complexity gate** — fails when cyclomatic complexity exceeds
   the configured thresholds (see above).
2. **Radon maintainability index gate** — fails when any module scores
   below A (MI < 20).

If either gate fails, the PR cannot be merged until the structural
cost is addressed.

Radon cyclomatic complexity is also reported (blocks with rank C or
worse are highlighted) but does not fail the build — Xenon already
covers this dimension.

The heavier tools (Scalene, py-spy, CodeCarbon, pyRAPL) are intended
for **local or manual use** and are not part of the CI pipeline.

---

## Representative workloads

| Purpose | Command |
|---------|---------|
| Structural analysis | `make cost-structural` |
| Runtime profiling | `make cost-profile-scalene` |
| Energy estimation | `make cost-energy` |

For more realistic profiling, replace `--help` with an actual experiment
configuration:

```bash
uv run scalene -m coleman4hcs.cli run --config run.yaml
```

---

## Risks and constraints

- Complexity does **not** directly equal energy use.
- Energy must be measured from **runtime execution**, not inferred
  purely from structure.
- Profiling quality depends on how realistic the workload is.
- `pyRAPL` is hardware-dependent and should stay optional.
- Over-enforcement too early may create friction; thresholds should be
  tuned after first real runs.
