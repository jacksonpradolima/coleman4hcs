"""Strict bounded JSON command boundary shared by local CLI and optional tool adapters."""

from __future__ import annotations

from typing import Literal

from coleman.application.contracts import ApplicationError, Caller, Handle, SafeId, Submit
from coleman.application.service import ApplicationService
from coleman.results.domain import canonical_json
from coleman.results.versioned import VersionedContract


class JobCommand(VersionedContract):
    """Versioned status/cancel identity with no caller or filesystem parameters."""

    job_id: SafeId


Operation = Literal["submit", "status", "cancel", "result", "export", "import"]


class ApplicationCommands:
    """Host-bound caller context; clients cannot change identity through command JSON."""

    def __init__(self, service: ApplicationService, *, caller: Caller) -> None:
        """Bind trusted local identity once for this adapter/session."""
        self.service = service
        self.caller = Caller.model_validate(caller)

    def invoke(self, operation: Operation, payload: str) -> str:
        """Return a finite canonical JSON success/error envelope with stable application codes.

        Non-import commands are capped at 16 KiB before parsing, imports at the
        server's output cap. Duplicate keys and unknown schema versions fail closed.
        Response framing adds bounded JSON overhead to the native bundle cap.
        """
        try:
            cap = self.service.policy.caps.max_output_bytes if operation == "import" else 16384
            if not isinstance(payload, str) or len(payload.encode("utf-8")) > cap:
                raise ApplicationError("resource_limit", "Command exceeds input byte cap")
            if operation == "submit":
                data = self.service.submit(Submit.from_json(payload), caller=self.caller).model_dump(mode="json")
            elif operation in {"status", "cancel"}:
                command = JobCommand.from_json(payload)
                method = self.service.status if operation == "status" else self.service.cancel
                data = method(command.job_id, caller=self.caller).model_dump(mode="json")
            elif operation in {"result", "export"}:
                handle = Handle.from_json(payload)
                # Export returns the same portable bundle, not a double-encoded JSON string.
                data = self.service.result(handle, caller=self.caller).model_dump(mode="json")
            elif operation == "import":
                data = self.service.import_result(payload, caller=self.caller).model_dump(mode="json")
            else:
                raise ApplicationError("invalid_input", "Unknown application operation")
            return canonical_json({"schema_version": 1, "ok": True, "data": data})
        except ApplicationError as exc:
            return canonical_json({"schema_version": 1, "ok": False, "error": exc.code, "message": str(exc)})
        except (ValueError, TypeError, RecursionError):
            return canonical_json(
                {
                    "schema_version": 1,
                    "ok": False,
                    "error": "invalid_input",
                    "message": "Invalid versioned command contract",
                }
            )
