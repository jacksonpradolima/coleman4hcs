"""Authored tiny CI sessions in the public TSV schema; no foreign data rows are bundled."""

import csv
import io

from coleman.corpus.materialization import SourceCache, materialize_source
from coleman.corpus.westermo_contracts import WestermoRecipe, WestermoSessionKey
from coleman.corpus.westermo_raw import FIELDS
from tests.corpus.test_source_views import source_spec


def owned_rows():
    """Exercise all outcomes, null/zero duration, siblings, quarantine and one unselected session."""
    rows = []
    for oid, jid, system, verdict, duration in (
        (1, 11, "owned-linux", 0, "0"),
        (2, 11, "owned-linux", 1, "3"),
        (3, 11, "owned-linux", 2, "2"),
        (4, 11, "owned-linux", 3, "0"),
        (5, 11, "owned-arm", 1, "4"),
        (6, 22, "owned-linux", 0, "1"),
    ):
        rows.append(
            dict(
                zip(
                    FIELDS,
                    (
                        str(jid),
                        system,
                        "Q" if system == "owned-arm" else "N",
                        "M",
                        "1970-03-04 01:02:03",
                        "NULL" if system == "owned-arm" else "0",
                        "owned-testware",
                        str(oid),
                        "7",
                        "owned/check",
                        "NULL",
                        str(verdict),
                        "1970-03-04 01:02:04",
                        duration,
                    ),
                    strict=True,
                )
            )
        )
    return rows


def owned_recipe(rows=None):
    """Construct exact source pins and deterministic explicit selections, without I/O."""
    rows = owned_rows() if rows is None else rows
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    payload = stream.getvalue().encode()
    recipe = WestermoRecipe(
        schema_version=1,
        source=source_spec("owned-test-results.tsv", payload),
        sessions=(WestermoSessionKey(jid=11, system="owned-linux"), WestermoSessionKey(jid=11, system="owned-arm")),
    )
    return recipe, payload


def prepare_westermo(tmp_path, rows=None):
    """Use actual licensed cache/materialization, never a mocked principal reader or artifact writer."""
    recipe, payload = owned_recipe(rows)
    cache = SourceCache(tmp_path / "cache")
    materialize_source(recipe.source, cache, lambda: io.BytesIO(payload))
    return recipe, cache
