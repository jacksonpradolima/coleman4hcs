"""Installed-package local transport for phase-one corpus registration and fixture checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from coleman.artifacts import ArtifactWriter
from coleman.corpus import (
    CorpusManifest,
    fixture_readiness,
    load_registry,
    publish_fixture_report,
    validate_bundled_source,
)


def main(argv: list[str] | None = None) -> int:
    """List/show metadata or validate a curated tiny source, without network or provider imports.

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
            workspace = Path.cwd().resolve()
            output = (workspace / args.output).resolve()
            publication_parent = (output / "corpus-validations").resolve()
            if not output.is_relative_to(workspace) or not publication_parent.is_relative_to(workspace):
                raise ValueError("Corpus output must remain within the current workspace")
            report = validate_bundled_source(args.benchmark_id, registry)
            path = publish_fixture_report(report, ArtifactWriter(output))
            payload = {
                "fixture_status": "verified",
                "artifact": str(path),
                "source_sha256": report.manifest.source.sha256,
                "evidence_scope": report.evidence_scope,
                "generalization": report.generalization,
            }
    except (KeyError, ValueError, OSError) as exc:
        print(f"Corpus input/validation/publication error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
