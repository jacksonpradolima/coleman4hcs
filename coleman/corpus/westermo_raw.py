"""Bounded offline original Westermo TSV reader using the existing verified source cache."""

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Iterator
from typing import NamedTuple

from coleman.corpus.materialization import SourceCache, validate_materialization
from coleman.corpus.materialization_contracts import MaterializationReceipt
from coleman.corpus.westermo_contracts import (
    WestermoCounts,
    WestermoObservation,
    WestermoRecipe,
    WestermoSession,
    WestermoSessionKey,
)

FIELDS = (
    "jid",
    "system",
    "sessiontype",
    "rel",
    "sstarted",
    "sduration",
    "fgstate",
    "oid",
    "tid",
    "path",
    "parameters",
    "result",
    "ostarted",
    "oduration",
)


class WestermoSlice(NamedTuple):
    """Typed selected records plus conserved counters; never an unvalidated persistence contract."""

    observations: tuple[WestermoObservation, ...]
    counts: WestermoCounts


def _integer(value: str, *, maximum: int = 2**63 - 1, minimum: int = 1) -> int:
    """Reject signs, fractional/nonfinite values, leading zeros and out-of-contract original integers."""
    if re.fullmatch(r"0|[1-9][0-9]{0,18}", value) is None or not minimum <= int(value) <= maximum:
        raise ValueError("Westermo source integer is invalid or outside bounds")
    return int(value)


def _observation(row: dict[str, str], position: int) -> WestermoObservation:
    """Decode exact source columns at the parsing boundary; no label, unit or clock inference."""
    session = WestermoSession.model_validate(
        {
            "key": WestermoSessionKey(jid=_integer(row["jid"]), system=row["system"]),
            "sessiontype": row["sessiontype"],
            "rel": row["rel"],
            "sstarted": row["sstarted"],
            "sduration": None if row["sduration"] == "NULL" else _integer(row["sduration"], maximum=10**12, minimum=0),
            "fgstate": row["fgstate"],
        }
    )
    return WestermoObservation(
        source_position=position,
        session=session,
        oid=_integer(row["oid"]),
        tid=_integer(row["tid"]),
        path=row["path"],
        parameters=row["parameters"],
        result=_integer(row["result"], maximum=3, minimum=0),
        ostarted=row["ostarted"],
        oduration=_integer(row["oduration"], maximum=10**12, minimum=0),
    )


def _source_observations(recipe: WestermoRecipe, cache: SourceCache) -> Iterator[WestermoObservation]:
    """Verify the full package and yield schema-checked, bounded original records, including unselected rows."""
    receipt = MaterializationReceipt(schema_version=1, spec=recipe.source)
    validate_materialization(receipt, cache)
    source = cache.member_path(receipt, recipe.source.members[0].name)
    try:
        with source.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t", strict=True)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError("Unsupported Westermo original TSV schema")
            for position, row in enumerate(reader, 1):
                if position > 2_000_000 or None in row or any(v is None or len(v) > 1024 for v in row.values()):
                    raise ValueError("Westermo TSV row/field bounds exceeded")
                yield _observation(row, position)
    except csv.Error as error:
        raise ValueError("Malformed Westermo original TSV") from error


def read_westermo_slice(recipe: WestermoRecipe, cache: SourceCache) -> WestermoSlice:
    """Reverify archive/member correspondence and stream at most two million source rows.

    Every row is decoded, including outside-selection rows. Duplicate original
    outcome IDs and inconsistent session headers reject globally; at most 20,000
    selected rows and 100,000 total sessions are retained in bounded collections.
    Missing selections, malformed/oversized fields, future statuses and invalid
    quantities raise ValueError. Original TSV bytes remain unchanged in the cache.
    No network, native policy, CLI, wall clock or external process is accessed.
    """
    recipe = WestermoRecipe.model_validate(recipe)
    selected = {(item.jid, item.system) for item in recipe.sessions}
    observations, seen, sessions = [], set(), {}
    found = set()
    total = 0
    for observation in _source_observations(recipe, cache):
        total = observation.source_position
        if observation.oid in seen:
            raise ValueError("Duplicate Westermo original outcome identity")
        seen.add(observation.oid)
        session = observation.session
        key = session.key.jid, session.key.system
        if sessions.setdefault(key, session) != session:
            raise ValueError("Conflicting Westermo source session metadata")
        if len(sessions) > 100_000:
            raise ValueError("Westermo source session limit exceeded")
        if key in selected:
            found.add(key)
            if len(observations) >= 20_000:
                raise ValueError("Westermo selected slice exceeds 20000 observations")
            observations.append(observation)
    if found != selected:
        raise ValueError("Selected Westermo session is absent from source")
    return WestermoSlice(
        observations=tuple(sorted(observations, key=lambda item: item.oid)),
        counts=_selected_counts(total, observations, tuple(sessions[key] for key in selected)),
    )


def _selected_counts(
    total: int, observations: list[WestermoObservation], sessions: tuple[WestermoSession, ...]
) -> WestermoCounts:
    """Conserve source rows, all four outcomes and complete/incomplete selected sessions exactly once."""
    outcomes = Counter(item.result for item in observations)
    incomplete = sum(session.sduration is None for session in sessions)
    return WestermoCounts(
        input_rows=total,
        selected_rows=len(observations),
        outside_selection=total - len(observations),
        passed=outcomes[0],
        failed=outcomes[1],
        invalid=outcomes[2],
        unmappable=outcomes[3],
        complete_sessions=len(sessions) - incomplete,
        incomplete_sessions=incomplete,
    )
