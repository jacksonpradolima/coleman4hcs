"""Cost-aware historical CI replay over the existing frozen pilot/native contracts."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.allocation.quantities import ExactPayoff
from coleman.pilot.contracts import CREDITS, OVERHEAD_SECONDS, SECONDS, PilotManifest, PilotRun
from coleman.pilot.replay import verify_run
from coleman.resources.contracts import (
    ELAPSED_MS,
    QUEUE_MS,
    RUNNER_MS,
    SETUP_MS,
    IntentEvent,
    OccupancyEvent,
    OverheadEvent,
    ReserveEvent,
    ResourceAmount,
    ResourceEvent,
)
from coleman.resources.ledger import ResourceLedger, _Replay
from coleman.resources.selection import ResourceRequest, ResourceSelection, TestEstimate, select_resources
from coleman.results.domain import Identifier, Measurement, fingerprint
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable, Quantity, decimal_value, ticks


def _whole_ms(value: Quantity) -> int:
    """Explicitly convert declared pilot occupancy seconds to exact integer milliseconds.

    Fractional milliseconds are unsupported source mechanics, not rounded values.
    This adapter does not convert arbitrary units or elapsed/queue quantities.
    """
    if value.semantics != SECONDS:
        raise ValueError("Source occupancy must use pilot runner seconds")
    whole, remainder = divmod(ticks(value.value), 10**9)
    if remainder or whole > 10**12:
        raise ValueError("Resource replay requires whole milliseconds <=10^12")
    return whole


def _overhead_ms(value: Quantity) -> int:
    """Ceil measured prioritization seconds to whole charged milliseconds explicitly.

    The native pilot measurement is elapsed prioritization time. This replay
    models one occupied runner during that interval; it does not measure a CI
    provider's bill. Ceiling error is less than 1 ms per decision.
    """
    if value.semantics != OVERHEAD_SECONDS:
        raise ValueError("Expected measured native prioritization overhead seconds")
    whole, remainder = divmod(ticks(value.value), 10**9)
    return whole + bool(remainder)


class BuildEstimates(Immutable):
    """One build's frozen predecision cost information, separate from actual test rows."""

    build_id: Identifier
    tests: Annotated[tuple[TestEstimate, ...], Field(max_length=1000)]
    priority_status: Literal["available", "stale", "unavailable"] = "available"

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        """Canonicalize membership; the request validator rejects duplicate test IDs."""
        object.__setattr__(self, "tests", tuple(sorted(self.tests, key=lambda test: test.test_id)))
        return self


class ResourceReplayProtocol(VersionedContract):
    """Reuse the pilot's source, temporal splits, native runs, baseline and trial registry.

    Estimation evidence must be available before the protocol is frozen, and the
    existing pilot requires that freeze before protected testing. Expired/missing
    estimates remain auditable no-action cases. Actual full-suite outcomes stay
    outside selector inputs. V1 uses serial whole runner milliseconds, at most
    1,000 tests/build; no refunds, capacity adjustment or parallel replay is claimed.
    This is a historical/synthetic counterfactual, never prospective evidence.
    """

    native_manifest: PilotManifest
    capacity: ResourceAmount
    estimates: Annotated[tuple[BuildEstimates, ...], Field(max_length=1000)]
    fallback: Literal["no_action", "full_suite"] = "no_action"

    @model_validator(mode="after")
    def _frozen(self) -> Self:
        """Bind full universes and supported native/estimation mechanics before replay."""
        if self.capacity.semantics != RUNNER_MS:
            raise ValueError("CI replay capacity requires occupied runner milliseconds")
        builds = {build.build_id: build for build in self.native_manifest.builds}
        if len({item.build_id for item in self.estimates}) != len(self.estimates):
            raise ValueError("Duplicate build estimates")
        if set(builds) != {item.build_id for item in self.estimates}:
            raise ValueError("Estimates must declare every native build")
        for item in self.estimates:
            build = builds[item.build_id]
            work = sum(_whole_ms(test.duration) for test in build.tests)
            if _whole_ms(self.native_manifest.budget) < work:
                raise ValueError("Native evidence must execute the full required suite")
            if build.decided_at + timedelta(milliseconds=work) > build.outcomes_available_at:
                raise ValueError("Source outcomes precede completion of serial occupied work")
            ResourceRequest(
                schema_version=1,
                decision_id=build.build_id,
                configuration_fingerprint=self.native_manifest.fingerprint,
                decided_at=build.decided_at,
                capacity=self.capacity,
                tests=item.tests,
                ranking=build.baseline_order,
            )
            if any(
                test.estimate is not None and test.estimate.available_at > self.native_manifest.frozen_at
                for test in item.tests
            ):
                raise ValueError("Estimation source unavailable when protocol was frozen")
        object.__setattr__(self, "estimates", tuple(sorted(self.estimates, key=lambda item: item.build_id)))
        return self


class ResourceReplayRun(VersionedContract):
    """Verified native full-suite execution plus a modeled estimate-based selection.

    Actual native runs retain their timing evidence. Selected execution is a
    counterfactual resource replay: required native CI tests remain fully run.
    Queue/setup elapsed and occupied setup are explicit null when unmeasured;
    null setup occupancy makes cost accounting incomplete, never silently zero.
    All derived ledgers/summaries are recomputed from typed sources on read.
    Legacy pilot/native data is embedded unchanged, not implicitly migrated.
    """

    protocol: ResourceReplayProtocol
    native_run: PilotRun
    queue_elapsed: ResourceAmount | None
    setup_elapsed: ResourceAmount | None
    shared_setup_occupancy: ResourceAmount | None

    @model_validator(mode="after")
    def _verified(self) -> Self:
        """Verify real native evidence and distinct overhead units before projection."""
        verify_run(self.protocol.native_manifest, self.native_run)
        for value, semantics in (
            (self.queue_elapsed, QUEUE_MS),
            (self.setup_elapsed, SETUP_MS),
            (self.shared_setup_occupancy, RUNNER_MS),
        ):
            if value is not None and value.semantics != semantics:
                raise ValueError("Incompatible queue/setup/occupancy semantics")
        # Replaying here makes malformed or unsupported projections fail on read,
        # including overlarge event logs, rather than only during report rendering.
        self._ledger()
        return self

    def request(self) -> ResourceRequest:
        """Expose only IDs, frozen estimates and cutoff; actual rows never cross this boundary."""
        manifest = self.protocol.native_manifest
        build = next(build for build in manifest.builds if build.build_id == self.native_run.build_id)
        inputs = next(item for item in self.protocol.estimates if item.build_id == build.build_id)
        return ResourceRequest(
            schema_version=1,
            decision_id=build.build_id,
            configuration_fingerprint=self.protocol.fingerprint,
            decided_at=build.decided_at,
            capacity=self.protocol.capacity,
            tests=inputs.tests,
            ranking=self.native_run.ranking,
            priority_status=inputs.priority_status,
            fallback=self.protocol.fallback,
        )

    @property
    def selection(self) -> ResourceSelection:
        """Return initial estimated selection, separately from contained actual replay."""
        return select_resources(self.request())

    @property
    def logical_id(self) -> str:
        """Identify one frozen source/protocol/candidate/build, excluding native attempt timing."""
        return fingerprint(
            {
                "namespace": "resource-replay-v1",
                "protocol": self.protocol.fingerprint,
                "candidate": self.native_run.candidate_id,
                "build": self.native_run.build_id,
            }
        )

    @property
    def ledger(self) -> ResourceLedger:
        """Recompute actual charged occupancy for the counterfactual selected prefix."""
        return ResourceReplayRun.model_validate(self)._ledger()

    def _ledger(self) -> ResourceLedger:
        """Project a bounded serial replay; containment stops new hypothetical dispatches."""
        manifest = self.protocol.native_manifest
        build = next(build for build in manifest.builds if build.build_id == self.native_run.build_id)
        replay = _Replay(self.protocol.capacity)
        events: list[ResourceEvent] = []

        def emit(event: ResourceEvent) -> None:
            """Use the shared accounting kernel once per event; validate the complete log at the end."""
            replay.apply(event)
            events.append(event)

        # Source receipt is the observed build availability. Synthetic serial
        # offsets describe the replay, excluding unmeasured queue/setup delays.
        elapsed = 0

        def metadata(kind, test_id="common"):
            """Describe synthetic event time and original source receipt without a local clock."""
            return {
                "event_id": f"{kind}:{test_id}",
                "sequence": len(events) + 1,
                "event_at": build.decided_at + timedelta(milliseconds=elapsed),
                "recorded_at": build.outcomes_available_at,
            }

        overhead = _overhead_ms(self.native_run.prioritization_overhead)
        emit(
            OverheadEvent(
                **metadata("dispatch"),
                component="dispatch",
                amount=ResourceAmount(value=overhead, semantics=RUNNER_MS),
            )
        )
        if self.shared_setup_occupancy is not None:
            emit(OverheadEvent(**metadata("setup"), component="shared_setup", amount=self.shared_setup_occupancy))
        estimates = {test.test_id: test.estimate for test in self.request().tests}
        tests = {test.test_id: test for test in build.tests}
        for test_id in self.selection.selected_ids:
            estimate = estimates[test_id]
            if estimate is None:
                raise ValueError("Selected test has no eligible estimate")
            if (
                replay.contained
                or estimate.amount.value > replay.available
                or build.decided_at + timedelta(milliseconds=elapsed) > estimate.valid_until
            ):
                break
            emit(IntentEvent(**metadata("intent", test_id), execution_id=test_id, estimate=estimate))
            emit(ReserveEvent(**metadata("reserve", test_id), execution_id=test_id, amount=estimate.amount))
            elapsed += _whole_ms(tests[test_id].duration)
            emit(
                OccupancyEvent(
                    **metadata("complete", test_id),
                    execution_id=test_id,
                    amount=ResourceAmount(value=_whole_ms(tests[test_id].duration), semantics=RUNNER_MS),
                    terminal=tests[test_id].outcome,
                )
            )
        return ResourceLedger(
            schema_version=1,
            ledger_id=self.logical_id,
            source_fingerprint=manifest.dataset_fingerprint,
            configuration_fingerprint=self.protocol.fingerprint,
            capacity=self.protocol.capacity,
            events=tuple(events),
        )

    def accounting(self) -> ResourceAccounting:
        """Recompute descriptive work/quality, preserving unknown and censored feedback.

        All source failures remain in the denominator; unresolved source labels
        make recall unavailable even when excluded. Serial elapsed excludes
        separately charged overhead and unmeasured queue/setup. No confidence
        interval or probability calibration is inferred from these quantities.
        """
        run = ResourceReplayRun.model_validate(self)
        ledger = run.ledger
        build = next(b for b in run.protocol.native_manifest.builds if b.build_id == run.native_run.build_id)
        selected = tuple(event.execution_id for event in ledger.events if isinstance(event, OccupancyEvent))
        tests = {test.test_id: test for test in build.tests}
        unresolved = any(test.outcome in {"unknown", "timeout"} for test in tests.values())
        failed = sum(test.outcome == "failed" for test in tests.values())
        detected = sum(tests[test_id].outcome == "failed" for test_id in selected)
        recall = None if unresolved or not failed else float(Fraction(detected, failed))
        recall_status: Literal["observed", "unavailable", "not_applicable"] = "observed"
        if unresolved:
            recall_status = "unavailable"
        elif not failed:
            recall_status = "not_applicable"
        elapsed = 0
        first_failure = None
        for test_id in selected:
            elapsed += _whole_ms(tests[test_id].duration)
            if first_failure is None and tests[test_id].outcome == "failed":
                first_failure = elapsed
        return ResourceAccounting(
            selected_ids=selected,
            initial_selection=run.selection,
            charged_ledger=ledger,
            runner_work=Quantity(value=Decimal(ledger.balance.runner_occupancy), semantics=RUNNER_MS),
            execution_elapsed=Quantity(value=Decimal(elapsed), semantics=ELAPSED_MS),
            overhead=Quantity(value=Decimal(ledger.balance.overhead), semantics=RUNNER_MS),
            failing_test_recall=Measurement(
                name="resource_replay_recall",
                unit="1",
                meaning="complete-universe-failing-test-recall-v1",
                direction="maximize",
                value=recall,
                status=recall_status,
            ),
            first_failure_elapsed=Measurement(
                name="resource_replay_first_failure",
                unit="ms",
                meaning=ELAPSED_MS.meaning,
                direction="minimize",
                value=float(first_failure) if first_failure is not None else None,
                status="observed" if first_failure is not None else "unavailable",
            ),
            first_failure_status="detected" if first_failure is not None else "right_censored",
            accounted_modeled_cost=run.modeled_cost(),
            accounting_status="partial_unmeasured_setup"
            if run.shared_setup_occupancy is None
            else "declared_runner_scope",
        )

    def modeled_cost(self) -> ExactPayoff:
        """Price charged replay work/overhead with the frozen modeled credit/second rate.

        Integer ms conversion and up to 12-place rate multiply exactly at 24 places,
        without Decimal-context rounding. Fixed/unmeasured setup/queue/provider
        costs remain excluded. This is modeled credits, not an observed bill.
        """
        rate = ticks(self.protocol.native_manifest.variable_price.value)
        cost = decimal_value(self.ledger.balance.consumed * rate * 10**9, places=24)
        return ExactPayoff(value=cost, semantics=CREDITS)


class ResourceAccounting(Immutable):
    """Recomputed summary; full typed source runs accompany persisted artifacts.

    Work/elapsed/overhead use explicit distinct semantics. Modeled credits price
    only charged work/overhead; unknown setup occupancy limits completeness.
    This is descriptive counterfactual evidence, never measured CI cost savings.
    """

    selected_ids: tuple[Identifier, ...]
    initial_selection: ResourceSelection
    charged_ledger: ResourceLedger
    runner_work: Quantity
    execution_elapsed: Quantity
    overhead: Quantity
    failing_test_recall: Measurement
    first_failure_elapsed: Measurement
    first_failure_status: Literal["detected", "right_censored"]
    accounted_modeled_cost: ExactPayoff
    accounting_status: Literal["partial_unmeasured_setup", "declared_runner_scope"]
