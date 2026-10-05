from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from navine.utils.config import load_config


DEFAULT_MCP: Dict[str, Any] = {
    "enabled": True,
    "client": {"enabled": True, "timeout_seconds": 60},
    "servers": {},
    "serve": {
        "enabled": True,
        "name": "Navine AI - Python",
        "version": "0.2.2",
        "transport": "stdio",
        "http_host": "127.0.0.1",
        "http_port": 8767,
        "path": "/mcp",
    },
}


def load_mcp_config() -> Dict[str, Any]:
    try:
        raw = load_config("mcp")
    except Exception:
        raw = {}
    out = dict(DEFAULT_MCP)
    if isinstance(raw, dict):
        out.update({k: v for k, v in raw.items() if k not in ("client", "servers", "serve")})
        if isinstance(raw.get("client"), dict):
            client = dict(out["client"])
            client.update(raw["client"])
            out["client"] = client
        if isinstance(raw.get("serve"), dict):
            serve = dict(out["serve"])
            serve.update(raw["serve"])
            out["serve"] = serve
        if isinstance(raw.get("servers"), dict):
            out["servers"] = dict(raw["servers"])
    return out


def list_configured_servers(cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    cfg = cfg or load_mcp_config()
    rows: List[Dict[str, Any]] = []
    for name, spec in (cfg.get("servers") or {}).items():
        if not isinstance(spec, dict):
            continue
        rows.append(
            {
                "name": str(name),
                "enabled": bool(spec.get("enabled", True)),
                "command": str(spec.get("command") or ""),
                "args": list(spec.get("args") or []),
                "url": str(spec.get("url") or ""),
                "transport": str(spec.get("transport") or ("url" if spec.get("url") else "stdio")),
            }
        )
    return rows


def get_server_spec(name: str, cfg: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    cfg = cfg or load_mcp_config()
    spec = (cfg.get("servers") or {}).get(name)
    if not isinstance(spec, dict):
        return None
    out = dict(spec)
    out["name"] = name
    return out


def mcp_sdk_available() -> bool:
    try:
        import mcp  # noqa: F401

        return True
    except Exception:
        return False


def _mcp_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "mcp.yaml"


def save_mcp_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    path = _mcp_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, default_flow_style=False, sort_keys=False)
    return load_mcp_config()


def _valid_server_name(name: str) -> str:
    key = re.sub(r"[^a-zA-Z0-9_-]+", "_", (name or "").strip()).strip("_").lower()
    if not key:
        raise ValueError("Server name is required")
    if len(key) > 48:
        raise ValueError("Server name too long")
    return key


def upsert_server(
    name: str,
    *,
    command: str = "",
    args: Optional[List[str]] = None,
    url: str = "",
    enabled: bool = True,
    env: Optional[Dict[str, str]] = None,
    cwd: str = "",
) -> Dict[str, Any]:
    key = _valid_server_name(name)
    cfg = load_mcp_config()
    servers = dict(cfg.get("servers") or {})
    spec: Dict[str, Any] = {"enabled": bool(enabled)}
    url = str(url or "").strip()
    command = str(command or "").strip()
    if url:
        spec["url"] = url
        spec["transport"] = "url"
    elif command:
        spec["command"] = command
        spec["args"] = [str(a) for a in (args or []) if str(a).strip()]
        spec["transport"] = "stdio"
    else:
        raise ValueError("Provide either a URL or a command")
    if env:
        spec["env"] = {str(k): str(v) for k, v in env.items()}
    if cwd:
        spec["cwd"] = str(cwd)
    servers[key] = spec
    cfg["servers"] = servers
    save_mcp_config(cfg)
    return {"ok": True, "name": key, "server": get_server_spec(key)}


def remove_server(name: str) -> Dict[str, Any]:
    key = _valid_server_name(name)
    cfg = load_mcp_config()
    servers = dict(cfg.get("servers") or {})
    if key not in servers:
        raise ValueError(f"Unknown MCP server: {key}")
    servers.pop(key, None)
    cfg["servers"] = servers
    save_mcp_config(cfg)
    return {"ok": True, "removed": key}
