"""No-model optional MCP lifecycle and shared command equivalence using the actual SDK."""

import asyncio
import json
import sys
import time

import pytest

from coleman.application import ApplicationService, Caller, ResultBundle, Submit, default_registry
from coleman.application.commands import ApplicationCommands


def text_reply(reply):
    assert not reply.is_error
    return json.loads(reply.content[0].text)


async def lifecycle(client, expected):
    listed = await client.list_tools()
    assert {tool.name for tool in listed.tools} == {
        "discover",
        "submit",
        "status",
        "cancel",
        "result",
        "export_result",
        "import_result",
    }
    discovery = text_reply(await client.call_tool("discover", {}))
    assert discovery["transport_contract"] == "coleman-application-v1"
    request = Submit(schema_version=1, request_id="tool-1", spec_id="weighted-overlap-3")
    initial = text_reply(await client.call_tool("submit", {"request_json": request.to_json()}))["data"]
    job_command = json.dumps({"schema_version": 1, "job_id": initial["job_id"]})
    deadline = time.monotonic() + 10
    while True:
        view = text_reply(await client.call_tool("status", {"command_json": job_command}))["data"]
        if view["state"] in {"completed", "failed", "cancelled", "expired"}:
            break
        assert time.monotonic() < deadline
        await asyncio.sleep(0.02)
    assert view["state"] == "completed"
    handle = json.dumps(view["handle"])
    native = text_reply(await client.call_tool("result", {"handle_json": handle}))["data"]
    bundle = ResultBundle.model_validate(native)
    assert bundle == expected
    exported = text_reply(await client.call_tool("export_result", {"handle_json": handle}))["data"]
    imported = text_reply(await client.call_tool("import_result", {"bundle_json": json.dumps(exported)}))["data"]
    assert imported["handle"]["digest"] == view["handle"]["digest"]
    assert imported["handle"]["job_id"] != view["handle"]["job_id"]
    assert text_reply(await client.call_tool("cancel", {"command_json": job_command}))["data"]["state"] == "completed"
    queued = text_reply(
        await client.call_tool(
            "submit", {"request_json": request.model_copy(update={"request_id": "tool-cancel"}).to_json()}
        )
    )["data"]
    cancelled = text_reply(
        await client.call_tool(
            "cancel", {"command_json": json.dumps({"schema_version": 1, "job_id": queued["job_id"]})}
        )
    )["data"]
    assert cancelled["state"] == "cancelled"
    malicious = json.loads(request.to_json())
    malicious["caller"] = {"subject": "admin"}
    assert (
        text_reply(await client.call_tool("submit", {"request_json": json.dumps(malicious)}))["error"]
        == "invalid_input"
    )
    return bundle.fingerprint


def test_actual_optional_sdk_tool_lifecycle(tmp_path):
    pytest.importorskip("mcp")
    from mcp.client import Client

    from coleman.transports.mcp import create_server

    async def exercise():
        with ApplicationService(default_registry(), root=tmp_path) as service:
            caller = Caller(subject="local")
            commands = ApplicationCommands(service, caller=caller)
            raw = commands.invoke(
                "submit", Submit(schema_version=1, request_id="library", spec_id="weighted-overlap-3").to_json()
            )
            view = service.wait(json.loads(raw)["data"]["job_id"], caller=caller)
            assert view.handle is not None
            expected = service.result(view.handle, caller=caller)
            async with Client(create_server(service, caller=caller)) as client:
                await lifecycle(client, expected)

    asyncio.run(exercise())


def test_actual_stdio_transport_lifecycle_without_model(tmp_path):
    pytest.importorskip("mcp")
    from mcp.client import Client
    from mcp.client.stdio import StdioServerParameters

    from coleman.application.worker import WorkItem, execute

    expected = execute(
        WorkItem(
            configuration=default_registry()[0],
            limits=Submit(schema_version=1, request_id="reference", spec_id="weighted-overlap-3").limits,
        )
    )

    async def exercise():
        parameters = StdioServerParameters(
            command=sys.executable, args=["-m", "coleman.transports.mcp"], cwd=str(tmp_path)
        )
        async with Client(parameters) as client:
            await lifecycle(client, expected)

    asyncio.run(asyncio.wait_for(exercise(), timeout=20))
