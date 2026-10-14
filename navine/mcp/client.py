from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from navine.mcp.config import get_server_spec, list_configured_servers, load_mcp_config, mcp_sdk_available


def _run(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def _require_sdk() -> None:
    if not mcp_sdk_available():
        raise RuntimeError("MCP SDK missing. Install with: pip install 'mcp>=1.28,<3'")


async def _connect_and(spec: Dict[str, Any], op):
    from mcp import Client, StdioServerParameters

    url = str(spec.get("url") or "").strip()
    if url:
        async with Client(url) as client:
            return await op(client)

    command = str(spec.get("command") or "").strip()
    if not command:
        raise ValueError(f"MCP server '{spec.get('name')}' needs command or url")
    args = [str(a) for a in (spec.get("args") or [])]
    env = None
    if isinstance(spec.get("env"), dict) and spec["env"]:
        env = {**os.environ, **{str(k): str(v) for k, v in spec["env"].items()}}
    cwd = str(spec.get("cwd") or "").strip() or None
    params = StdioServerParameters(command=command, args=args, env=env, cwd=cwd)
    async with Client(params) as client:
        return await op(client)


async def _list_tools_async(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    async def op(client):
        listed = await client.list_tools()
        tools = getattr(listed, "tools", listed) or []
        rows = []
        for tool in tools:
            rows.append(
                {
                    "name": str(getattr(tool, "name", "") or ""),
                    "description": str(getattr(tool, "description", "") or ""),
                    "input_schema": getattr(tool, "inputSchema", None)
                    or getattr(tool, "input_schema", None),
                }
            )
        return rows

    return await _connect_and(spec, op)


async def _call_tool_async(spec: Dict[str, Any], tool: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    async def op(client):
        result = await client.call_tool(str(tool), arguments or {})
        structured = getattr(result, "structured_content", None)
        content = getattr(result, "content", None)
        text_parts: List[str] = []
        if content:
            for item in content:
                text = getattr(item, "text", None)
                if text is not None:
                    text_parts.append(str(text))
                else:
                    text_parts.append(str(item))
        return {
            "ok": True,
            "tool": tool,
            "structured": structured,
            "text": "\n".join(text_parts).strip(),
            "raw": str(result),
        }

    return await _connect_and(spec, op)


def status() -> Dict[str, Any]:
    cfg = load_mcp_config()
    return {
        "ok": True,
        "enabled": bool(cfg.get("enabled", True)),
        "sdk_installed": mcp_sdk_available(),
        "client_enabled": bool((cfg.get("client") or {}).get("enabled", True)),
        "serve_enabled": bool((cfg.get("serve") or {}).get("enabled", True)),
        "servers": list_configured_servers(cfg),
        "serve": dict(cfg.get("serve") or {}),
    }


def list_servers() -> List[Dict[str, Any]]:
    return list_configured_servers()


def list_tools(server: str) -> Dict[str, Any]:
    _require_sdk()
    spec = get_server_spec(server)
    if not spec:
        return {"ok": False, "error": f"Unknown MCP server: {server}"}
    if not bool(spec.get("enabled", True)):
        return {"ok": False, "error": f"MCP server disabled: {server}"}
    try:
        tools = _run(_list_tools_async(spec))
        return {"ok": True, "server": server, "tools": tools}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "server": server}


def call_tool(server: str, tool: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    _require_sdk()
    spec = get_server_spec(server)
    if not spec:
        return {"ok": False, "error": f"Unknown MCP server: {server}"}
    if not bool(spec.get("enabled", True)):
        return {"ok": False, "error": f"MCP server disabled: {server}"}
    try:
        result = _run(_call_tool_async(spec, tool, arguments or {}))
        result["server"] = server
        return result
    except Exception as exc:
        return {"ok": False, "error": str(exc), "server": server, "tool": tool}


def parse_call_args(pairs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for raw in pairs or []:
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        try:
            out[key] = json.loads(value)
        except Exception:
            out[key] = value
    return out
