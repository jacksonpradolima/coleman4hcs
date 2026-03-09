"""
coleman4hcs.results.clickhouse_sink - Optional ClickHouse Results Sink.

Requires ``pip install coleman4hcs[clickhouse]`` which pulls in the
``clickhouse-connect`` driver.  This sink inserts result rows into a
ClickHouse table in batches.

When the extra is **not** installed, importing this module will raise
``ImportError`` at construction time only — the rest of the framework
remains unaffected.
"""

from __future__ import annotations

import threading
from typing import Any

from coleman4hcs.results.sink_base import ResultsSink

_CLICKHOUSE_TABLE = "coleman_results"

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {_CLICKHOUSE_TABLE} (
    scenario           String,
    experiment         Int64,
    step               Int64,
    policy             String,
    reward_function    String,
    sched_time         Float64,
    sched_time_duration Float64,
    total_build_duration Float64,
    prioritization_time Float64,
    detected           Int64,
    missed             Int64,
    tests_ran          Int64,
    tests_not_ran      Int64,
    ttf                Float64,
    ttf_duration       Float64,
    time_reduction     Float64,
    fitness            Float64,
    cost               Float64,
    rewards            Float64,
    avg_precision      Float64,
    prioritization_order String
) ENGINE = MergeTree()
ORDER BY (scenario, policy, reward_function, experiment, step)
"""

_INSERT_COLS = [
    "scenario",
    "experiment",
    "step",
    "policy",
    "reward_function",
    "sched_time",
    "sched_time_duration",
    "total_build_duration",
    "prioritization_time",
    "detected",
    "missed",
    "tests_ran",
    "tests_not_ran",
    "ttf",
    "ttf_duration",
    "time_reduction",
    "fitness",
    "cost",
    "rewards",
    "avg_precision",
    "prioritization_order",
]


class ClickHouseSink(ResultsSink):
    """ClickHouse results sink (optional).

    Parameters
    ----------
    host : str
        ClickHouse server hostname.
    port : int
        ClickHouse HTTP interface port.
    database : str
        Target database name.
    batch_size : int
        Rows buffered before an automatic insert.

    Raises
    ------
    ImportError
        If ``clickhouse-connect`` is not installed.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8123,
        database: str = "default",
        batch_size: int = 1000,
    ) -> None:
        try:
            import clickhouse_connect  # noqa: F811
        except ImportError as exc:
            raise ImportError(
                "clickhouse-connect is required for ClickHouseSink. "
                "Install it with: pip install coleman4hcs[clickhouse]"
            ) from exc

        self._client = clickhouse_connect.get_client(host=host, port=port, database=database)
        self._client.command(_CREATE_TABLE_SQL)
        self.batch_size = batch_size
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def write_row(self, row: dict[str, Any]) -> None:
        """Buffer one result row.

        Parameters
        ----------
        row : dict[str, Any]
            Column-name → value mapping.
        """
        with self._lock:
            # Ensure prioritization_order is a string for ClickHouse
            r = dict(row)
            if isinstance(r.get("prioritization_order"), list):
                r["prioritization_order"] = str(r["prioritization_order"])
            self._buffer.append(r)
            if len(self._buffer) >= self.batch_size:
                self._flush_locked()

    def flush(self) -> None:
        """Force-write the current buffer to ClickHouse."""
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        """Flush remaining data and close the connection."""
        self.flush()

    def _flush_locked(self) -> None:
        """Insert buffered rows into ClickHouse (caller holds lock)."""
        if not self._buffer:
            return
        data = [[row.get(col) for col in _INSERT_COLS] for row in self._buffer]
        self._client.insert(_CLICKHOUSE_TABLE, data, column_names=_INSERT_COLS)
        self._buffer.clear()
