"""
coleman.results.sink_base - Results Sink Interface and Null Adapter.

Defines the ``ResultsSink`` abstract interface that every results backend must
implement, plus a ``NullSink`` no-op adapter used when results collection is
disabled.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from typing import Any

from coleman.results.domain import DomainRecord


class ResultsSink(abc.ABC):
    """Abstract interface for experiment-results backends.

    Implementations receive lightweight row dicts via :meth:`write_row` and
    are responsible for batching, flushing and persistence.

    Methods
    -------
    write_row(row)
        Persist a single result row.
    flush()
        Force any buffered data to storage.
    close()
        Release resources.
    """

    @abc.abstractmethod
    def write_row(self, row: dict[str, Any]) -> None:
        """Persist a single experiment-result row.

        Parameters
        ----------
        row : dict[str, Any]
            Column-name → value mapping for one result record.
        """

    @abc.abstractmethod
    def flush(self) -> None:
        """Force any buffered data to the underlying storage."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release resources held by this sink."""

    def write_domain_record(self, record: DomainRecord) -> None:
        """Commit a typed record or fail explicitly when unsupported.

        Additive capability: existing third-party sinks need not implement it.
        Local sinks commit synchronously before checkpoints. Remote/Null sinks
        raise ``NotImplementedError`` rather than silently discard evidence.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support domain records")

    def read_domain_record(
        self, run_id: str, execution_id: str, decision_id: str, *, namespace: str
    ) -> DomainRecord | None:
        """Query a logical key or raise ``NotImplementedError`` when unsupported."""
        raise NotImplementedError(f"{type(self).__name__} does not support domain records")

    def domain_records(self) -> Iterator[DomainRecord]:
        """Iterate committed typed records or fail explicitly when unsupported."""
        raise NotImplementedError(f"{type(self).__name__} does not support domain records")


class NullSink(ResultsSink):  # pylint: disable=abstract-method
    """No-op results sink used when results collection is disabled.

    All operations are instant and discard data, with near-zero overhead.
    """

    def write_row(self, row: dict[str, Any]) -> None:
        """Discard the row (no-op).

        Parameters
        ----------
        row : dict[str, Any]
            Ignored.
        """

    def flush(self) -> None:
        """No-op flush."""

    def close(self) -> None:
        """No-op close."""
