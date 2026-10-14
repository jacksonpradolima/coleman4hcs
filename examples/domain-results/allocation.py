"""Synthetic CI runner-slot allocation accounting, without a new solver.

Run: python allocation.py OUTPUT_DIRECTORY. Credits are synthetic modeled units,
not currency or a claim of measured savings. Negative outcomes remain visible.
"""

import sys
from datetime import UTC, datetime

from coleman.results.domain import DomainRecord, LearningReward, Measurement, ResultProvenance, fingerprint
from coleman.results.duckdb_sink import DuckDBSink


def main(output):
    inputs = {"runner_slots": 2, "jobs": ["unit-tests", "integration-tests"]}
    record = DomainRecord(
        schema_version=1,
        namespace="ci.runner_allocation",
        run_id="run-example",
        execution_id="execution-1",
        decision_id="decision-7",
        data_fingerprint=fingerprint(inputs),
        available_at=datetime(2026, 1, 2, 12, tzinfo=UTC),
        provenance=ResultProvenance(
            producer="synthetic-ci-allocation",
            producer_version="1",
            spec_fingerprint=fingerprint({"allocation": inputs}),
            evaluation_fingerprint=fingerprint({"protocol": "synthetic-runner-credit-v1"}),
        ),
        measurements=(
            Measurement(
                name="duration",
                value=3.0,
                unit="s",
                direction="minimize",
                meaning="serial-job-duration-v1",
                status="observed",
                minimum=0.0,
            ),
            Measurement(
                name="net_value",
                value=-2.0,
                unit="credit",
                direction="maximize",
                meaning="synthetic-net-runner-credit-v1",
                status="observed",
            ),
            Measurement(
                name="unknown_cost",
                value=None,
                unit="credit",
                direction="minimize",
                meaning="synthetic-runner-cost-v1",
                status="unknown",
            ),
        ),
        learning_reward=LearningReward(value=0.25, unit="1", transform_id="bounded-v1"),
    )
    sink = DuckDBSink(out_dir=output)
    try:
        sink.write_domain_record(record)
        sink.write_domain_record(record)  # identical replay, including across program restarts
        sink.flush()
        assert (
            sink.read_domain_record("run-example", "execution-1", "decision-7", namespace="ci.runner_allocation")
            == record
        )
        assert record.measurement("net_value").value == -2.0
        assert len(list(sink.domain_records())) == 1
        print(record.to_json())
    finally:
        sink.close()


if __name__ == "__main__":
    main(sys.argv[1])
