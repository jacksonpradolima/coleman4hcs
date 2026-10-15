"""Installed-package local transport for corpus descriptions and fixture checks."""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import cast

from coleman.artifacts import ArtifactWriter
from coleman.corpus import (
    CorpusManifest,
    CorpusRegistry,
    MaterializationReceipt,
    MaterializationSpec,
    SourceCache,
    fixture_readiness,
    load_registry,
    materialize_source,
    profile_native_source,
    publish_fixture_report,
    publish_materialization,
    publish_source_profile,
    validate_bundled_source,
    validate_materialization,
)


def main(argv: list[str] | None = None) -> int:
    """Inspect metadata, profile local native bytes or validate curated tiny correctness sources.

    Output paths must remain within the caller's workspace. Exit 0 means a
    successful metadata/fixture operation, never completed corpus diversity or
    policy efficacy. Exit 2 means invalid/blocked input or publication failure.
    Source integrity is separate from conversion and confirmatory evaluation.
    Public HTTPS is explicit and restricted; no source URI/import is executed implicitly.
    """
    parser = argparse.ArgumentParser(description="Inspect CI corpus registrations and exact tiny correctness fixtures")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    commands.add_parser("schema")
    show = commands.add_parser("show")
    show.add_argument("benchmark_id")
    validate = commands.add_parser("validate")
    validate.add_argument("benchmark_id")
    validate.add_argument("--output", default="corpus-artifacts")
    profile = commands.add_parser("profile")
    profile.add_argument("benchmark_id")
    profile.add_argument("source", type=Path)
    profile.add_argument("--output", default="corpus-artifacts")
    commands.add_parser("materialization-schema")
    fetch = commands.add_parser("fetch")
    fetch.add_argument("spec", type=Path)
    fetch.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "coleman" / "corpus")
    fetch.add_argument("--output", default="corpus-artifacts")
    source = fetch.add_mutually_exclusive_group()
    source.add_argument("--source", type=Path)
    source.add_argument("--public-https")
    cached = commands.add_parser("validate-cache")
    cached.add_argument("receipt", type=Path)
    cached.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "coleman" / "corpus")
    args = parser.parse_args(argv)
    try:
        registry = load_registry()
        if args.command in {"fetch", "validate-cache"}:
            payload = _materialization(args)
        elif args.command == "materialization-schema":
            payload = MaterializationSpec.model_json_schema()
        elif args.command == "schema":
            payload = CorpusManifest.model_json_schema()
        elif args.command == "list":
            payload = json.loads(fixture_readiness(registry).to_json())
        elif args.command == "show":
            payload = json.loads(registry.card(args.benchmark_id).to_json())
        else:
            payload = _publish(args, registry)
    except (KeyError, ValueError, OSError) as exc:
        print(f"Corpus input/validation/publication error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


def _publish(args: argparse.Namespace, registry: CorpusRegistry) -> dict:
    """Confine complete output to the workspace; input paths are explicit read-only local files."""
    family = "corpus-profiles" if args.command == "profile" else "corpus-validations"
    writer = _workspace_writer(args.output, family)
    if args.command == "profile":
        profile = profile_native_source(registry.card(args.benchmark_id), args.source)
        path = publish_source_profile(profile, writer)
        return {
            "profile_status": "descriptive_only",
            "artifact": str(path),
            "rows": profile.rows,
            "generalization": profile.generalization,
        }
    report = validate_bundled_source(args.benchmark_id, registry)
    path = publish_fixture_report(report, writer)
    return {
        "fixture_status": "verified",
        "artifact": str(path),
        "source_sha256": report.manifest.source.sha256,
        "evidence_scope": report.evidence_scope,
        "generalization": report.generalization,
    }


def _contract_bytes(path: Path) -> bytes:
    """Read bounded metadata only; source/archive data uses the streaming materialization core."""
    with path.open("rb") as stream:
        payload = stream.read(256 * 1024 + 1)
    if len(payload) > 256 * 1024:
        raise ValueError("Source package metadata exceeds 256 KiB")
    return payload


def _materialization(args: argparse.Namespace) -> dict:
    """Use explicit licensed metadata, external cache and optional local/public source without eval admission."""
    if args.command == "validate-cache":
        receipt = MaterializationReceipt.from_json(_contract_bytes(args.receipt))
        cache = SourceCache(args.cache)
        validate_materialization(receipt, cache)
        return {"integrity": "verified", "admission": receipt.admission, "fingerprint": receipt.fingerprint}
    spec = MaterializationSpec.from_json(_contract_bytes(args.spec))
    writer = _workspace_writer(args.output, "corpus-materializations")
    cache = SourceCache(args.cache)
    factory = None
    if args.source is not None:
        factory = partial(args.source.open, "rb")
    elif args.public_https is not None:
        from coleman.transports.corpus_http import public_https_source

        factory = partial(public_https_source, args.public_https, cast(int, spec.source.size_bytes))
    receipt = materialize_source(spec, cache, factory, refresh_source=args.source is not None)
    path = publish_materialization(receipt, cache, writer)
    return {
        "integrity": "verified",
        "admission": receipt.admission,
        "artifact": str(path),
        "fingerprint": receipt.fingerprint,
    }


def _workspace_writer(output_name: str, family: str) -> ArtifactWriter:
    """Check trusted workspace output/family parents before acquisition/publication side effects."""
    workspace = Path.cwd().resolve()
    output = (workspace / output_name).resolve()
    if not output.is_relative_to(workspace) or not (output / family).resolve().is_relative_to(workspace):
        raise ValueError("Corpus output must remain within the current workspace")
    return ArtifactWriter(output)


if __name__ == "__main__":
    raise SystemExit(main())
