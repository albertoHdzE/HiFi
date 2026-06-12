"""
Unit tests for the HiFi Extended Indicators MCP server (P8-E0).

Design principle: these tests validate the server's module structure and data
loading helpers without requiring pandas-ta or venvs/ta/ to be set up. All
pandas-ta imports are lazy (inside tool functions), so importing the module
in the main environment is safe.

For integration-level subprocess tests that actually call the tools, see
tests/integration/test_indicators_server.py — those tests are skipped when
venvs/ta/ has not been set up.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

_VENVS_TA_PYTHON = Path(__file__).parent.parent.parent / "venvs" / "ta" / "bin" / "python"


# ---------------------------------------------------------------------------
# Module structure tests (no pandas-ta required)
# ---------------------------------------------------------------------------


def test_module_is_importable():
    """indicators_server can be imported in the main environment (no pandas-ta)."""
    import hifi.mcp.indicators_server as srv  # noqa: F401


def test_mcp_server_instance_exists():
    """The module-level `mcp` FastMCP instance exists."""
    import hifi.mcp.indicators_server as srv

    assert srv.mcp is not None


def test_get_extended_indicators_is_registered():
    """get_extended_indicators is registered as an MCP tool."""
    import hifi.mcp.indicators_server as srv

    # FastMCP exposes registered tools via its internal registry
    tool_names = [t.name for t in srv.mcp._tool_manager.list_tools()]
    assert "get_extended_indicators" in tool_names


def test_call_id_is_12_chars():
    """_call_id returns a 12-character hex string."""
    from hifi.mcp.indicators_server import _call_id

    cid = _call_id(tool="get_extended_indicators", ticker="AAPL", date="2023-03-31", window=14)
    assert len(cid) == 12
    assert all(c in "0123456789abcdef" for c in cid)


def test_call_id_is_deterministic():
    """Same inputs always produce the same call_id."""
    from hifi.mcp.indicators_server import _call_id

    cid1 = _call_id(tool="get_extended_indicators", ticker="AAPL", date="2023-03-31", window=14)
    cid2 = _call_id(tool="get_extended_indicators", ticker="AAPL", date="2023-03-31", window=14)
    assert cid1 == cid2


def test_call_id_differs_for_different_inputs():
    """Different inputs produce different call_ids."""
    from hifi.mcp.indicators_server import _call_id

    cid_aapl = _call_id(tool="get_extended_indicators", ticker="AAPL", date="2023-03-31")
    cid_jpm = _call_id(tool="get_extended_indicators", ticker="JPM", date="2023-03-31")
    assert cid_aapl != cid_jpm


def test_unknown_ticker_returns_error_dict(tmp_path):
    """get_extended_indicators returns TICKER_NOT_FOUND for unknown ticker."""
    import hifi.mcp.indicators_server as srv

    os.environ["HIFI_DATA_DIR"] = str(tmp_path)
    result = srv.get_extended_indicators(ticker="ZZZZZ", date="2023-03-31")
    assert "error" in result
    assert result["error"] == "TICKER_NOT_FOUND"
    assert "call_id" in result


# ---------------------------------------------------------------------------
# call_tool extension: python_executable parameter
# ---------------------------------------------------------------------------


def test_call_tool_signature_has_python_executable():
    """call_tool() accepts python_executable keyword argument."""
    import inspect

    from hifi.agents.mcp_client import call_tool

    sig = inspect.signature(call_tool)
    assert "python_executable" in sig.parameters


def test_call_tool_default_python_executable_is_none():
    """python_executable defaults to None (use sys.executable)."""
    import inspect

    from hifi.agents.mcp_client import call_tool

    sig = inspect.signature(call_tool)
    assert sig.parameters["python_executable"].default is None


# ---------------------------------------------------------------------------
# venvs/ta subprocess integration (skipped when venv not present)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _VENVS_TA_PYTHON.exists(),
    reason="venvs/ta/ not set up; run: bash scripts/setup_ta_venv.sh",
)
def test_venvs_ta_python_can_import_pandas_ta():
    """venvs/ta/bin/python can import pandas_ta (basic smoke test)."""
    result = subprocess.run(
        [str(_VENVS_TA_PYTHON), "-c", "import pandas_ta; print(pandas_ta.version)"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"


@pytest.mark.skipif(
    not _VENVS_TA_PYTHON.exists(),
    reason="venvs/ta/ not set up; run: bash scripts/setup_ta_venv.sh",
)
def test_indicators_server_responds_to_tools_list_via_subprocess(tmp_path):
    """
    indicators_server started via venvs/ta/bin/python responds to MCP protocol.

    This test sends a minimal JSON-RPC sequence (initialize) to confirm the
    server is functional. It does not call get_extended_indicators (which would
    require a parquet fixture) — tool execution is tested in integration tests.
    """
    import subprocess as sp

    src_dir = str(Path(__file__).parent.parent.parent / "src")
    env = os.environ.copy()
    env["PYTHONPATH"] = src_dir
    env["HIFI_DATA_DIR"] = str(tmp_path)

    proc = sp.Popen(
        [str(_VENVS_TA_PYTHON), "-m", "hifi.mcp.indicators_server"],
        stdin=sp.PIPE,
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        env=env,
        text=True,
    )

    import uuid

    init_id = str(uuid.uuid4())
    init_msg = json.dumps({
        "jsonrpc": "2.0",
        "id": init_id,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0.1"},
        },
    }) + "\n"

    try:
        proc.stdin.write(init_msg)
        proc.stdin.flush()

        # Read until we get the initialize response
        for _ in range(20):
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                if "id" in msg and str(msg["id"]) == init_id:
                    assert "result" in msg
                    assert "serverInfo" in msg["result"] or "capabilities" in msg["result"]
                    return
            except json.JSONDecodeError:
                continue
        pytest.fail("Did not receive initialize response from indicators_server")
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)
