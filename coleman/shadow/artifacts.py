"""Shadow event commits and checkpoint references through the existing immutable writer."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Self

from pydantic import Field, model_validator

from coleman.artifacts import ArtifactWriter
from coleman.results.domain import Fingerprint
from coleman.results.versioned import VersionedContract
from coleman.shadow.contracts import AdmittedEvent, ShadowProtocol
from coleman.shadow.state import ShadowState


class ShadowJournalFrame(VersionedContract):
    """One atomically committed admission, with explicit ordinal and predecessor digest.

    Frame publication is the source-event commit point. An interrupted optional
    checkpoint-reference write does not lose a committed event. The caller may
    acknowledge only after successful publication, and retains its durable source
    for duplicate replay after an ambiguous I/O failure.
    """

    ordinal: Annotated[int, Field(strict=True, ge=1, le=4096)]
    protocol_fingerprint: Fingerprint
    previous_frame: Fingerprint | None
    admission: AdmittedEvent

    @model_validator(mode="after")
    def _binding(self) -> Self:
        """Reject a mismatched source/protocol or invalid first-frame predecessor."""
        if self.protocol_fingerprint != self.admission.protocol_fingerprint:
            raise ValueError("Journal frame and admission protocols disagree")
        if (self.ordinal == 1) != (self.previous_frame is None):
            raise ValueError("Only the first journal frame has no predecessor")
        return self


class ShadowCheckpoint(VersionedContract):
    """Small verified reference to an immutable committed journal prefix.

    References do not duplicate the full growing source log at every checkpoint.
    Disk use is bounded by the journal byte budget plus O(event count) fixed-size
    references. Complete portable state remains available as ShadowState JSON.
    """

    protocol_fingerprint: Fingerprint
    event_count: Annotated[int, Field(strict=True, ge=0, le=4096)]
    last_frame: Fingerprint | None
    state_fingerprint: Fingerprint

    @model_validator(mode="after")
    def _empty(self) -> Self:
        """Require a final frame digest exactly when the checkpoint is nonempty."""
        if (self.event_count == 0) != (self.last_frame is None):
            raise ValueError("Empty checkpoint and last frame digest disagree")
        return self


def _root(writer: ArtifactWriter, protocol: ShadowProtocol) -> Path:
    """Use only a validated digest as the experiment path component."""
    return writer.run_root / "shadow-journals" / protocol.logical_id.removeprefix("sha256:")


def publish_shadow_state(state: ShadowState, writer: ArtifactWriter) -> Path:
    """Commit verified source admissions before a compact native-resume checkpoint reference.

    Reuse ArtifactWriter's atomic create-if-absent/hard-link/fsync semantics.
    Existing identical prefixes are idempotent; conflicting concurrent branches
    fail at their first differing ordinal. No mutable latest pointer can regress.
    Each complete frame is an accepted/quarantined event commit, even if process
    interruption prevents its derived checkpoint reference. Temporary files are
    never commits. Requires trusted local roots and the existing writer's storage
    support; ancestor-directory power-loss durability remains outside that contract.
    """
    state = ShadowState.model_validate(state)
    root = _root(writer, state.protocol)
    header = state.protocol.to_json() + "\n"
    if len(header.encode("utf-8")) > 1024 * 1024:
        raise ValueError("Shadow protocol header exceeds supported size")
    writer.write_immutable(root / "protocol.json", header)
    previous = None
    for ordinal, entry in enumerate(state.journal, start=1):
        frame = ShadowJournalFrame(
            schema_version=1,
            ordinal=ordinal,
            protocol_fingerprint=state.protocol.fingerprint,
            previous_frame=previous,
            admission=entry,
        )
        writer.write_immutable(root / "frames" / f"{ordinal:04d}.json", frame.to_json() + "\n")
        previous = frame.fingerprint
    checkpoint = ShadowCheckpoint(
        schema_version=1,
        protocol_fingerprint=state.protocol.fingerprint,
        event_count=len(state.journal),
        last_frame=previous,
        state_fingerprint=state.fingerprint,
    )
    path = root / "checkpoints" / f"{len(state.journal):04d}.json"
    writer.write_immutable(path, checkpoint.to_json() + "\n")
    return path


def load_shadow_state(protocol: ShadowProtocol, writer: ArtifactWriter) -> ShadowState:
    """Replay the complete validated committed prefix, including post-checkpoint event commits.

    Read at most the frozen frame/byte limits. Only atomically published `.json`
    frames count; `.tmp` files never do. Explicit ordinals, predecessor hashes and
    protocol bindings reject gaps, tampering, unsupported versions and forked
    histories. This is domain-event replay using existing artifacts, not a new
    scheduler or a replacement native checkpoint backend. No code/pickle executes.
    """
    protocol = ShadowProtocol.model_validate(protocol)
    root = _root(writer, protocol)
    header = root / "protocol.json"
    if not header.exists():
        if (root / "frames").exists():
            raise ValueError("Shadow frames exist without a frozen protocol header")
        return ShadowState(schema_version=1, protocol=protocol)
    if header.stat().st_size > 1024 * 1024:
        raise ValueError("Shadow protocol header exceeds supported size")
    if ShadowProtocol.from_json(header.read_bytes()).to_json() != protocol.to_json():
        raise ValueError("Stored shadow protocol differs from requested experiment")
    paths = sorted((root / "frames").glob("*.json"))
    if len(paths) > protocol.maximum_events:
        raise ValueError("Shadow journal exceeds frozen event retention limit")
    previous = None
    admissions = []
    total = 0
    for ordinal, path in enumerate(paths, start=1):
        if path.name != f"{ordinal:04d}.json":
            raise ValueError("Committed shadow journal has a missing or invalid ordinal")
        size = path.stat().st_size
        total += size
        if size > protocol.maximum_event_bytes + 4096 or total > protocol.maximum_journal_bytes + 4096 * len(paths):
            raise ValueError("Committed shadow journal exceeds frozen byte limits")
        frame = ShadowJournalFrame.from_json(path.read_bytes())
        if (
            frame.ordinal != ordinal
            or frame.previous_frame != previous
            or frame.protocol_fingerprint != protocol.fingerprint
        ):
            raise ValueError("Committed frame ordinal, predecessor or protocol is invalid")
        previous = frame.fingerprint
        admissions.append(frame.admission)
    return ShadowState(schema_version=1, protocol=protocol, journal=tuple(admissions))


def verify_shadow_checkpoint(path: Path, protocol: ShadowProtocol, writer: ArtifactWriter) -> ShadowState:
    """Restore one acknowledged prefix and verify its state/final-frame digests.

    Later complete source frames remain available for recovery, but are excluded
    from the returned explicit checkpoint. Missing or altered prefix data fails;
    a reference alone is not a portable complete source archive.
    """
    checkpoint = ShadowCheckpoint.from_json(path.read_bytes())
    if checkpoint.protocol_fingerprint != protocol.fingerprint:
        raise ValueError("Checkpoint belongs to another frozen experiment")
    latest = load_shadow_state(protocol, writer)
    if len(latest.journal) < checkpoint.event_count:
        raise ValueError("Checkpoint references an incomplete committed prefix")
    restored = ShadowState(schema_version=1, protocol=protocol, journal=latest.journal[: checkpoint.event_count])
    if restored.fingerprint != checkpoint.state_fingerprint:
        raise ValueError("Checkpoint state digest mismatch")
    if checkpoint.event_count:
        frame_path = _root(writer, protocol) / "frames" / f"{checkpoint.event_count:04d}.json"
        if ShadowJournalFrame.from_json(frame_path.read_bytes()).fingerprint != checkpoint.last_frame:
            raise ValueError("Checkpoint final-frame digest mismatch")
    return restored
