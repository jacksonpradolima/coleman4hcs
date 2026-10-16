"""Shared strict v1 contracts for deterministic, domain-independent JSON artifacts."""

from __future__ import annotations

import json
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field

from coleman.results.domain import canonical_json, decode_json, fingerprint


class VersionedContract(BaseModel):
    """Strict immutable contract base with fail-closed v1 artifact reading.

    Required integer schema version 1 rejects missing, legacy and future versions.
    Serialization revalidates unchecked Pydantic instances before JSON conversion.
    No files are read or written by these models.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    schema_version: Annotated[int, Field(strict=True, ge=1, le=1)]

    def to_json(self) -> str:
        """Return deterministic finite JSON, revalidating nested contracts first."""
        validated = type(self).model_validate(self)
        return canonical_json(validated.model_dump(mode="json"))

    @classmethod
    def from_json(cls, payload: str | bytes) -> Self:
        """Preserve numeric decimal literals and reject duplicate keys/unknown versions without migration.

        Decimal-valued fields retain exact JSON quantities before precision checks;
        binary64 fields still explicitly project to floats through their typed
        models. Canonical decimal strings and valid existing artifacts are unchanged.
        """
        return cls.model_validate(decode_json(payload, exact_decimals=True))

    @property
    def fingerprint(self) -> str:
        """Hash the complete versioned contract, independent of Python hash/ordering."""
        return fingerprint(json.loads(self.to_json()))
