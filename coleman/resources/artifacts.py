"""Replay checkpoints reuse atomic immutable artifact publication, not a second store."""

from pathlib import Path

from coleman.artifacts import ArtifactWriter
from coleman.resources.ledger import ResourceLedger
from coleman.results.domain import canonical_json


def publish_ledger(ledger: ResourceLedger, writer: ArtifactWriter) -> Path:
    """Publish event identities before a full replay checkpoint with create-if-absent.

    Stable event/sequence paths reject concurrent conflicting branches. The full
    checkpoint appears only after every event is durable. Interrupted orphan
    events are not acknowledged state; retry the same original event to recover.
    Identical processes/retries are idempotent. Requires the existing writer's
    trusted local filesystem and hard-link/fsync support. Power-loss durability
    of newly created ancestor directories is outside that contract. No deletion,
    learning checkpoint advancement or application event acknowledgement occurs.
    """
    ledger = ResourceLedger.model_validate(ledger)
    root = writer.run_root / "resource-ledgers" / ledger.logical_id.removeprefix("sha256:")
    header = ledger.model_dump(mode="json", exclude={"events"})
    writer.write_immutable(root / "header.json", canonical_json(header) + "\n")
    for event in ledger.events:
        writer.write_immutable(
            root / "events" / f"{event.sequence:04d}.json", canonical_json(event.model_dump(mode="json")) + "\n"
        )
    path = root / "checkpoints" / f"{len(ledger.events):04d}.json"
    writer.write_immutable(path, ledger.to_json() + "\n")
    return path


def restore_ledger(path: Path) -> ResourceLedger:
    """Read an explicit committed snapshot, revalidating every event and balance.

    No directory scan selects a guessed latest version; a caller retains its
    acknowledged checkpoint path or replays its durable source. Missing/partial
    files and future schema versions fail. A complete checkpoint embeds all
    events, so copying it alone preserves replayability. No code/pickle executes.
    """
    return ResourceLedger.from_json(path.read_bytes())
