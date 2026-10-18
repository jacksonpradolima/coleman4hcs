"""Pure counter-keyed synthetic truth and predecision covariates, generated one build at a time."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from coleman.corpus.stress_contracts import StressCase
from coleman.results.domain import canonical_json


@dataclass(frozen=True)
class StressObservation:
    """One constructed variant execution, not an independent project or mapped fault.

    ``duration_seconds`` is exact integer runner occupancy (zero is real).
    ``context_score`` is a signed predecision covariate, not a probability or
    realized verdict; null means missing. Display names may collide; identity
    comes from logical test and variant. These internal generated rows are
    verified by regeneration at artifact boundaries, not parsed as trusted DTOs.
    """

    build: int
    test_id: str
    variant: str
    display_name: str
    duration_seconds: int
    verdict: int
    context_score: int | None
    latent_risk: int

    @property
    def execution_name(self) -> str:
        """Return a compact injective native key; display names never identify observations."""
        return self.variant + ":" + self.test_id

    def to_bytes(self) -> bytes:
        """Encode canonical JSONL with explicit units in field names, without an ambient RNG."""
        return (canonical_json(asdict(self)) + "\n").encode("utf-8")


def _counter(seed: int, channel: str, *indices: int) -> int:
    """Key an independent deterministic channel; no global/random-library state is read."""
    text = ":".join(map(str, (seed, channel, *indices)))
    return int.from_bytes(hashlib.sha256(text.encode("ascii")).digest()[:8], "big")


def _risk(case: StressCase, build: int, index: int) -> int:
    """Declare predecision risk changes; outcome noise is absent from this computation."""
    factors = case.factors
    group = index % 4
    if factors.drift == "sudden":
        return int(group == (0 if build < case.builds // 2 else 1))
    if factors.drift == "gradual":
        return int((index + build) % case.builds < build + 1)
    if factors.drift == "recurring":
        return int(group == (build // 2) % 2)
    return int(group == _counter(case.seed, "environment-context", build) % 4)


def _verdict(case: StressCase, build: int, index: int, risk: int) -> int:
    """Combine declared latent truth/flakiness; zero regime overrides even shared flips.

    Rare forces one logical failing test per build. Moderate selects the current
    risk group; bursts occur only every fourth build. Shared variant copies use
    identical truth. History-dependent flips use the preceding latent verdict,
    not previous executed feedback, so policy selection cannot change the world.
    """
    f = case.factors
    if f.failures == "zero":
        return 0
    if f.failures == "rare":
        return int(index == _counter(case.seed, "rare-location", build) % f.test_count)
    baseline = risk if f.failures == "moderate" else int(build % 4 == 2 and index % 2 == 0)
    if f.flakiness == "independent":
        flip = _counter(case.seed, "outcome-noise", build, index) % 32 == 0
    elif f.flakiness == "history":
        previous = _verdict(case, build - 1, index, _risk(case, build - 1, index)) if build else 0
        flip = bool(previous) and build % 3 == 1
    else:
        flip = _counter(case.seed, "shared-environment-outcome", build) % 4 == 0
    # Bursty sequences preserve long all-pass intervals explicitly.
    return int(bool(baseline) != bool(flip)) if f.failures != "bursty" or build % 4 == 2 else 0


def _duration(case: StressCase, build: int, index: int, verdict: int) -> int:
    """Use <=128 integer seconds with an explicit zero-cost logical test.

    A bounded power-of-two tail is declared rather than described as a sampled
    log-normal distribution. Correlated duration intentionally uses hidden truth;
    it is not automatically an available predecision scheduling feature.
    """
    if index == 0:
        return 0
    kind = case.factors.duration
    if kind == "heavy_tail":
        draw = _counter(case.seed, "duration-tail", index) & 127
        return 1 << (7 - draw.bit_length())
    if kind == "failure_correlated":
        return 16 if verdict else 1
    if kind == "changing":
        return 1 << ((build + index % 4) % 5)
    return 1


def _identity(case: StressCase, build: int, index: int) -> tuple[str, str] | None:
    """Apply churn without collapsing aliases; collision is a display-only ambiguity."""
    churn, count = case.factors.churn, case.factors.test_count
    if churn == "additions" and index >= max(1, count * (build + 1) // case.builds):
        return None
    if churn == "removals" and index >= max(1, count * (case.builds - build) // case.builds):
        return None
    logical = f"t{index:05}"
    if churn == "renames" and build >= case.builds // 2:
        return logical, "renamed-" + logical
    if churn == "collisions":
        # Equal labels, distinct logical tests; also rename the pair at mid-run.
        return logical, f"label-{index // 2:05}-{int(build >= case.builds // 2)}"
    return logical, logical


def generate_build(case: StressCase, build: int) -> tuple[StressObservation, ...]:
    """Generate at most 150,000 rows with exact truth, independent of execution order.

    Build indices are zero-based within the case. Unknown versions, bool/out-of-
    bounds indices reject before allocation. Only one build is materialized; no
    global RNG, I/O, native metric, CLI or MCP is used. Variants deliberately share
    logical truth and must never inflate the count of independent faults.
    """
    case = StressCase.model_validate(case)
    if type(build) is not int or not 0 <= build < case.builds:
        raise ValueError("Stress build index outside declared case")
    rows = []
    for index in range(case.factors.test_count):
        identity = _identity(case, build, index)
        if identity is None:
            continue
        risk = _risk(case, build, index)
        verdict = _verdict(case, build, index, risk)
        context = _context(case, build, index, risk)
        for variant in range(3):
            if case.factors.variants == "sparse" and (index + variant) % 3 != 0:
                continue
            if case.factors.variants == "changing" and variant != build % 3:
                continue
            rows.append(
                StressObservation(
                    build,
                    identity[0],
                    f"v{variant}",
                    identity[1],
                    _duration(case, build, index, verdict),
                    verdict,
                    context,
                    risk,
                )
            )
    return tuple(rows)


def _context(case: StressCase, build: int, index: int, risk: int) -> int | None:
    """Compute signed predecision covariates without reading outcome noise or future builds."""
    kind = case.factors.context
    if kind == "missing":
        return None
    if kind == "noisy":
        return _counter(case.seed, "context-noise", build, index) % 3 - 1
    return (2 * risk - 1) * (-1 if kind == "adversarial" else 1)
