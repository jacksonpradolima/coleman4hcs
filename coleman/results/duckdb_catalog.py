"""
coleman.results.duckdb_catalog - Optional DuckDB Views over Parquet.

Provides a thin helper that creates analytical DuckDB views on top of the
Hive-partitioned Parquet dataset produced by ``ParquetSink``.  This allows
users to run ad-hoc SQL queries over experiment results without loading data
into RAM.

Usage
-----
>>> from coleman.results.duckdb_catalog import DuckDBCatalog
>>> cat = DuckDBCatalog("./runs")
>>> df = cat.query("SELECT scenario, AVG(fitness) FROM results GROUP BY 1")
"""

from __future__ import annotations

import duckdb
import polars as pl


class DuckDBCatalog:
    """Read-only DuckDB view layer over a Parquet results dataset.

    Parameters
    ----------
    parquet_root : str
        Root directory of the Hive-partitioned Parquet dataset.
    db_path : str
        DuckDB database path.  Default ``:memory:`` (in-process, ephemeral).

    Attributes
    ----------
    parquet_root : str
        Parquet root directory.
    conn : duckdb.DuckDBPyConnection
        DuckDB connection.
    """

    def __init__(self, parquet_root: str, db_path: str = ":memory:") -> None:
        self.parquet_root = parquet_root
        self.conn = duckdb.connect(db_path)
        self._create_view()

    def _create_view(self) -> None:
        """Create the ``results`` view pointing at the Parquet dataset."""
        glob_path = f"{self.parquet_root}/**/*.parquet"
        escaped = glob_path.replace("'", "''")
        self.conn.execute(
            f"CREATE OR REPLACE VIEW results AS SELECT * FROM read_parquet('{escaped}', hive_partitioning=1)",
        )

    def query(self, sql: str) -> pl.DataFrame:
        """Execute an SQL query against the results view.

        Parameters
        ----------
        sql : str
            SQL statement.

        Returns
        -------
        polars.DataFrame
            Query result as a DataFrame.
        """
        return self.conn.execute(sql).pl()

    def close(self) -> None:
        """Close the DuckDB connection."""
        self.conn.close()
