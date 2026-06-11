"""
MCP subprocess client for HiFi agents (P3-E1).

Agents call Phase 2 financial tools via the MCP stdio server. This module provides
a synchronous call_tool() helper that manages the subprocess lifetime and translates
MCP JSON-RPC into plain Python dicts.

The server process is started fresh per call_tool() invocation. This is intentionally
simple for Phase 3 (one agent, one analysis). Phase 8+ (full population, concurrent
agents) will use a persistent server process with a connection pool.

Usage
-----
    from hifi.agents.mcp_client import call_tool

    result = call_tool(
        server_module="hifi.mcp.financial_server",
        tool_name="get_technical_indicators",
        params={"ticker": "AAPL", "date": "2023-03-31", "window": 20},
        data_dir="/path/to/data",
    )
    # result is a plain dict with the tool output fields + call_id
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import uuid
from typing import Any

from hifi.observability.tracing import _current_trace_id, get_tracer

logger = logging.getLogger(__name__)

# JSON-RPC version used by the MCP protocol
_JSONRPC = "2.0"


def call_tool(
    tool_name: str,
    params: dict[str, Any],
    server_module: str = "hifi.mcp.financial_server",
    data_dir: str | None = None,
) -> dict[str, Any]:
    """
    Call one tool on the Phase 2 MCP stdio server.

    Starts the server as a subprocess, sends the JSON-RPC initialize + tools/call
    sequence, reads the response, and terminates the subprocess.

    When an active trace context exists (set by trace_context() in run_analysis()
    or run_ensemble()), each call is wrapped in a LangFuse child span that captures
    the tool name, arguments, result, and call_id for audit trail linkage (DJ-023).

    Parameters
    ----------
    tool_name : str
        MCP tool name (e.g. "get_technical_indicators").
    params : dict
        Tool input parameters.
    server_module : str
        Python module to run as the MCP server (default: hifi.mcp.financial_server).
    data_dir : str | None
        Path to the data directory. Passed as HIFI_DATA_DIR env var to the server
        subprocess. If None, the server uses its own default ("data").

    Returns
    -------
    dict
        Tool result as a plain Python dict (includes call_id if the tool produced one).
        On error: {"error": "<ERROR_CODE>", "detail": "<message>"}.

    Raises
    ------
    RuntimeError
        If the subprocess fails to start or returns a malformed JSON-RPC response.
    """
    trace_id = _current_trace_id.get()
    if trace_id is None:
        return _call_subprocess(tool_name, params, server_module, data_dir)

    tracer = get_tracer()
    with tracer.span(trace_id, f"mcp_{tool_name}", input=params) as span_ctx:
        result = _call_subprocess(tool_name, params, server_module, data_dir)
        span_ctx.output = result
        call_id = result.get("call_id") if isinstance(result, dict) else None
        if call_id:
            span_ctx.metadata = {"call_id": call_id}
    return result


def _call_subprocess(
    tool_name: str,
    params: dict[str, Any],
    server_module: str,
    data_dir: str | None,
) -> dict[str, Any]:
    """Execute one MCP tool call as a subprocess (implementation detail)."""
    env = os.environ.copy()
    if data_dir is not None:
        env["HIFI_DATA_DIR"] = data_dir

    proc = subprocess.Popen(
        [sys.executable, "-m", server_module],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )

    try:
        return _execute(proc, tool_name, params)
    finally:
        proc.stdin.close()  # type: ignore[union-attr]
        proc.wait(timeout=10)


def _execute(
    proc: subprocess.Popen,
    tool_name: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Send initialize + tools/call and return the parsed tool result."""
    init_id = str(uuid.uuid4())
    call_id = str(uuid.uuid4())

    # Step 1: initialize handshake (required by MCP protocol before any tool call)
    _send(proc, {
        "jsonrpc": _JSONRPC,
        "id": init_id,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "hifi-agent", "version": "0.1"},
        },
    })
    init_response = _read_response(proc)
    if "error" in init_response:
        raise RuntimeError(f"MCP initialize failed: {init_response['error']}")

    # Send initialized notification (required after initialize response)
    _send(proc, {
        "jsonrpc": _JSONRPC,
        "method": "notifications/initialized",
        "params": {},
    })

    # Step 2: call the tool
    _send(proc, {
        "jsonrpc": _JSONRPC,
        "id": call_id,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": params},
    })
    tool_response = _read_response(proc, expected_id=call_id)

    if "error" in tool_response:
        logger.warning("MCP tool %s returned error: %s", tool_name, tool_response["error"])
        return {"error": tool_response["error"].get("message", "UNKNOWN_ERROR")}

    # MCP wraps the result in result.content[0].text as a JSON string
    result_obj = tool_response.get("result", {})
    content = result_obj.get("content", [])
    if content and isinstance(content, list):
        text = content[0].get("text", "{}")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"MCP tool result is not valid JSON: {exc}") from exc

    return result_obj


def _send(proc: subprocess.Popen, message: dict[str, Any]) -> None:
    """Write one JSON-RPC message to the subprocess stdin."""
    line = json.dumps(message) + "\n"
    proc.stdin.write(line)  # type: ignore[union-attr]
    proc.stdin.flush()  # type: ignore[union-attr]


def _read_response(
    proc: subprocess.Popen,
    expected_id: str | None = None,
) -> dict[str, Any]:
    """
    Read JSON-RPC lines from stdout until we find a response with the expected id.

    Skips notification messages (those without an "id" field). If expected_id is
    None, returns the first message that has an "id" field.
    """
    while True:
        line = proc.stdout.readline()  # type: ignore[union-attr]
        if not line:
            stderr_output = proc.stderr.read()  # type: ignore[union-attr]
            raise RuntimeError(
                f"MCP server closed stdout unexpectedly. stderr: {stderr_output[:500]}"
            )
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("Non-JSON line from MCP server: %s", line[:200])
            continue
        # Skip notifications (no "id" field)
        if "id" not in msg:
            continue
        if expected_id is None or str(msg.get("id")) == expected_id:
            return msg
        # Different id -- keep reading (could be an out-of-order message)
        logger.debug("Skipping MCP message with id %s (expected %s)", msg.get("id"), expected_id)
