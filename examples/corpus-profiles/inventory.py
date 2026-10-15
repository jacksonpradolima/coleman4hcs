"""Describe existing local native datasets without fetching, copying or admitting sources.

Usage: python inventory.py /path/to/coleman OUTPUT_DIRECTORY
Output must remain in the current workspace; the checkout is a read-only source.
"""

import json
import sys
from pathlib import Path
from urllib.parse import unquote

from coleman.artifacts import ArtifactWriter
from coleman.corpus import (
    CorpusProfileInventory,
    load_registry,
    profile_native_source,
    publish_profile_inventory,
    publish_source_profile,
)
from coleman.manifest import generate_manifest


def main(checkout, output):
    """Hash registered sources and atomically publish a complete bounded local inventory."""
    root = Path(checkout).resolve()
    workspace = Path.cwd().resolve()
    destination = (workspace / output).resolve()
    if not destination.is_relative_to(workspace):
        raise ValueError("Inventory output must remain within the current workspace")
    for family in ("corpus-profiles", "corpus-profile-inventories"):
        if not (destination / family).resolve().is_relative_to(workspace):
            raise ValueError("Inventory artifact parents must remain within the current workspace")
    registry = load_registry()
    writer = ArtifactWriter(destination)
    reports = []
    prefix = "https://github.com/jacksonpradolima/coleman/blob/b743f51d5ba444410b32f2c0e318b2587b882c43/"
    for card in registry.cards:
        if card.manifest.adapter.name != "native-parquet-corpus":
            continue
        uri = card.manifest.source.uri
        if not uri.startswith(prefix):
            raise ValueError("Example accepts only the registered existing-source snapshot")
        path = (root / unquote(uri.removeprefix(prefix))).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Source must remain within the explicit read-only checkout")
        reports.append(profile_native_source(card, path))
    inventory = CorpusProfileInventory(
        schema_version=1,
        registry=registry,
        profiles=tuple(reports),
        uncovered_required_regimes=registry.required_regimes,
    )
    # No complete inventory is published until every requested source has passed integrity checks.
    for report in inventory.profiles:
        publish_source_profile(report, writer)
    artifact = publish_profile_inventory(inventory, writer)
    generate_manifest(destination, run_id=inventory.fingerprint)
    print(
        json.dumps(
            {
                "inventory": str(artifact),
                "profiled_sources": len(inventory.profiles),
                "raw_rows": sum(report.rows for report in inventory.profiles),
                "uncovered_regimes": len(inventory.uncovered_required_regimes),
                "evidence_status": inventory.evidence_status,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
