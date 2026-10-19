"""Deterministic bounded Westermo Parquet export, without synthetic clocks or second budgets."""

from __future__ import annotations

import hashlib
from importlib.resources import files
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from coleman.artifacts import ArtifactWriter
from coleman.corpus.lrts_etl_contracts import ETLFile
from coleman.corpus.materialization import SourceCache, validate_cache_separation, validate_source_storage
from coleman.corpus.materialization_contracts import safe_member_name
from coleman.corpus.westermo_contracts import SESSION_SECONDS, TEST_DURATION, WestermoReceipt, WestermoRecipe
from coleman.corpus.westermo_raw import WestermoSlice, read_westermo_slice
from coleman.results.domain import fingerprint

_SHA256 = "sha256:"
_COMPLETION = "complete.json"


def load_westermo_recipe() -> WestermoRecipe:
    """Read the packaged original-release system13/first-16-nightly-session recipe, without source acquisition.

    This selection is a descriptive example, not a protected holdout or sixteen
    independent projects. Modifications require an explicit new recipe identity.
    """
    return WestermoRecipe.from_json(
        files("coleman.corpus").joinpath("resources/westermo-system13-first16-v1.json").read_bytes()
    )


def westermo_producer() -> str:
    """Bind conversion/contract/cache/storage source, not a caller-supplied version label.

    Changing any bound implementation creates a new export namespace. Physical
    Parquet reproduction additionally requires the receipt's Arrow version.
    """
    modules = {
        "coleman.corpus": (
            "westermo_contracts.py",
            "westermo_raw.py",
            "westermo_etl.py",
            "contracts.py",
            "materialization.py",
            "materialization_contracts.py",
            "archives.py",
            "zip_metadata.py",
            "lrts_etl_contracts.py",
        ),
        "coleman": ("artifacts.py",),
        "coleman.results": ("domain.py", "versioned.py"),
        "coleman.selection": ("contracts.py",),
    }
    return fingerprint(
        [
            [package + "/" + name, hashlib.sha256(files(package).joinpath(name).read_bytes()).hexdigest()]
            for package, names in sorted(modules.items())
            for name in names
        ]
    )


def _target(root: Path, relative: str) -> Path:
    """Reject portable-path violations, symlinks/escapes and Git-backed content before access."""
    safe_member_name(relative)
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Westermo output escapes its trusted artifact tree")
    validate_source_storage(path)
    return path


def preflight_westermo_etl(recipe: WestermoRecipe, cache: Path, output: Path) -> Path:
    """Derive all output paths and reject overlapping/Git roots, without creating or acquiring files.

    Local parent directories must be cooperative/trusted; this is not a hostile
    filesystem sandbox. Invalid paths/metadata raise ValueError, I/O errors propagate.
    """
    recipe = WestermoRecipe.model_validate(recipe)
    validate_source_storage(cache)
    validate_source_storage(output)
    validate_cache_separation(cache, output)
    identity = fingerprint([recipe.fingerprint, westermo_producer(), pa.__version__]).removeprefix(_SHA256)
    root = _target(output, "westermo-etl/" + identity)
    for relative in (_COMPLETION, "observations.parquet", "sessions.parquet"):
        _target(root, relative)
    return root


def _tables(source: WestermoSlice) -> tuple[pa.Table, pa.Table]:
    """Build explicitly typed single-chunk tables; original IDs/statuses/dates/units remain distinct.

    Invalid/unmappable rows retain null binary verdicts with an explicit status.
    Unknown physical work/availability remains null. No inference or native
    projection is performed. Session envelopes are represented once per session.
    """
    rows, sessions = [], {}
    for observation in source.observations:
        session = observation.session
        key = session.key.jid, session.key.system
        sessions[key] = session
        rows.append(
            {
                "execution_id": observation.execution_id,
                "source_position": observation.source_position,
                "jid": key[0],
                "system": key[1],
                "oid": observation.oid,
                "tid": observation.tid,
                "path": observation.path,
                "parameters_raw": observation.parameters,
                "result_raw": observation.result,
                "verdict": observation.result if observation.result in {0, 1} else None,
                "verdict_status": "observed" if observation.result in {0, 1} else "unavailable",
                "ostarted_shifted_naive": observation.ostarted,
                "oduration_raw": observation.oduration,
                "oduration_unit": TEST_DURATION.unit,
                "outcome_available_at": None,
                "runner_seconds": None,
            }
        )
    string, integer = pa.string(), pa.int64()
    observations = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("execution_id", string),
                ("source_position", integer),
                ("jid", integer),
                ("system", string),
                ("oid", integer),
                ("tid", integer),
                ("path", string),
                ("parameters_raw", string),
                ("result_raw", integer),
                ("verdict", integer),
                ("verdict_status", string),
                ("ostarted_shifted_naive", string),
                ("oduration_raw", integer),
                ("oduration_unit", string),
                ("outcome_available_at", string),
                ("runner_seconds", integer),
            ]
        ),
    )
    session_rows = [
        {
            "jid": key[0],
            "system": key[1],
            "sessiontype": session.sessiontype,
            "rel": session.rel,
            "sstarted_shifted_naive": session.sstarted,
            "sduration": session.sduration,
            "sduration_unit": SESSION_SECONDS.unit,
            "sduration_status": "unavailable" if session.sduration is None else "observed",
            "fgstate": session.fgstate,
        }
        for key, session in sorted(sessions.items())
    ]
    session_table = pa.Table.from_pylist(
        session_rows,
        schema=pa.schema(
            [
                ("jid", integer),
                ("system", string),
                ("sessiontype", string),
                ("rel", string),
                ("sstarted_shifted_naive", string),
                ("sduration", integer),
                ("sduration_unit", string),
                ("sduration_status", string),
                ("fgstate", string),
            ]
        ),
    )
    return observations, session_table


def _encode(table: pa.Table) -> bytes:
    """Write fixed-size row groups without dictionaries/compression/statistics; cap each file at 64 MiB.

    Single-chunk Arrow construction avoids Polars execution-thread chunk layout
    entering physical bytes. Arrow version remains part of the receipt identity.
    """
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        row_group_size=1024,
        version="2.6",
        compression="NONE",
        use_dictionary=False,
        write_statistics=False,
        data_page_version="1.0",
    )
    payload = sink.getvalue().to_pybytes()
    if len(payload) > 64 * 1024**2:
        raise ValueError("Westermo Parquet exceeds 64 MiB")
    return payload


def _projection(recipe: WestermoRecipe, cache: SourceCache) -> tuple[WestermoReceipt, tuple[bytes, ...]]:
    """Recompute source-derived counters, logical rows and physical files before any publication."""
    source = read_westermo_slice(recipe, cache)
    tables = _tables(source)
    payloads = tuple(_encode(table) for table in tables)
    logical_rows = [
        observation.model_dump(mode="json", exclude={"source_position"}) for observation in source.observations
    ]
    return WestermoReceipt(
        schema_version=1,
        recipe=recipe,
        producer_sha256=westermo_producer(),
        arrow_writer=pa.__version__,
        counts=source.counts,
        logical_rows_sha256=fingerprint(logical_rows),
        files=tuple(
            ETLFile(
                path=name,
                sha256=_SHA256 + hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
                rows=table.num_rows,
            )
            for name, payload, table in zip(("observations.parquet", "sessions.parquet"), payloads, tables, strict=True)
        ),
    ), payloads


def export_westermo_parquet(recipe: WestermoRecipe, cache: SourceCache, writer: ArtifactWriter) -> Path:
    """Publish deterministic analytical observations/sessions, completing only after both immutable files.

    Cache/source bytes are reverified offline before conversion. Identical
    concurrent retries converge; interruption leaves reusable individual files
    without a false completion marker. This is not an execution checkpoint or
    scientific admission. Legacy receipts/readers are unchanged. Invalid source,
    paths, units or conflicts raise ValueError; I/O errors propagate.
    """
    recipe = WestermoRecipe.model_validate(recipe)
    root = preflight_westermo_etl(recipe, cache.root, writer.run_root)
    receipt, payloads = _projection(recipe, cache)
    for item, payload in zip(receipt.files, payloads, strict=True):
        writer.write_immutable_stream(_target(root, item.path), (payload,), maximum_bytes=item.size_bytes)
    completion = _target(root, _COMPLETION)
    writer.write_immutable(completion, receipt.to_json() + "\n")
    return completion


def validate_westermo_etl(receipt: WestermoReceipt, cache: SourceCache, root: Path) -> None:
    """Verify the full source projection and output bytes offline; never repair or trust edited counters.

    Unknown producer/writer requires the recorded implementation; no automatic
    migration or relabeling occurs. Metadata parsing alone proves no source truth.
    Missing/oversized/tampered data raises ValueError/OSError. No Parquet from the
    caller is decoded, and neither network nor runner is invoked.
    """
    receipt = WestermoReceipt.model_validate(receipt)
    validate_cache_separation(cache.root, root)
    validate_source_storage(root)
    if (receipt.producer_sha256, receipt.arrow_writer) != (westermo_producer(), pa.__version__):
        raise ValueError("Unsupported Westermo producer/writer; use its recorded implementation")
    expected, _ = _projection(receipt.recipe, cache)
    if receipt != expected:
        raise ValueError("Westermo receipt disagrees with source-derived projection")
    for item in receipt.files:
        with _target(root, item.path).open("rb") as stream:
            payload = stream.read(item.size_bytes + 1)
        if len(payload) != item.size_bytes or _SHA256 + hashlib.sha256(payload).hexdigest() != item.sha256:
            raise ValueError("Westermo artifact bytes disagree with verified source")
