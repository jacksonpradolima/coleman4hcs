"""Installed-package local transport for corpus descriptions and fixture checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from coleman.artifacts import ArtifactWriter
from coleman.corpus import (
    CorpusManifest,
    CorpusRegistry,
    fixture_readiness,
    load_registry,
    profile_native_source,
    publish_fixture_report,
    publish_source_profile,
    validate_bundled_source,
)


def main(argv: list[str] | None = None) -> int:
    """Inspect metadata, profile local native bytes or validate curated tiny correctness sources.

    Output paths must remain within the caller's workspace. Exit 0 means a
    successful metadata/fixture operation, never completed corpus diversity or
    policy efficacy. Exit 2 means invalid/blocked input or publication failure.
    Fetching, conversion and confirmatory experiment commands are not implemented
    by this phase-one transport and are intentionally absent from its interface.
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
    args = parser.parse_args(argv)
    try:
        registry = load_registry()
        if args.command == "schema":
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
    workspace = Path.cwd().resolve()
    output = (workspace / args.output).resolve()
    family = "corpus-profiles" if args.command == "profile" else "corpus-validations"
    if not output.is_relative_to(workspace) or not (output / family).resolve().is_relative_to(workspace):
        raise ValueError("Corpus output must remain within the current workspace")
    writer = ArtifactWriter(output)
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


if __name__ == "__main__":
    raise SystemExit(main())
