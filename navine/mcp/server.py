from __future__ import annotations

from typing import Any, Dict, Optional

from navine.mcp.config import load_mcp_config, mcp_sdk_available


def _package_root() -> str:
    return str((__package__ or "navine.mcp").split(".", 1)[0] or "navine")


def build_server():
    _require_sdk()
    from mcp.server import MCPServer
    from navine.utils.brand import load_brand

    cfg = load_mcp_config()
    serve = dict(cfg.get("serve") or {})
    brand = load_brand()
    name = str(serve.get("name") or brand.get("name") or "Navine AI - Python")
    version = str(serve.get("version") or "0.2.2")
    mcp = MCPServer(name, version=version)

    @mcp.tool()
    def runtime_info() -> dict:
        """Return where UI vs models run for this Navine host."""
        try:
            from navine.utils.runtime_place import runtime_summary

            return runtime_summary()
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @mcp.tool()
    def chat(message: str, profile: str = "auto") -> str:
        """Chat with the local Navine text model."""
        from navine.text.chat import chat as navine_chat

        return navine_chat(
            str(message or ""),
            model_profile=None if str(profile or "auto").lower() in ("", "auto") else str(profile),
        )

    @mcp.tool()
    def generate_python_code(task: str) -> str:
        """Generate Python CLI/stdlib code for a task (Discord webhook/bot supported)."""
        from navine.code.fallbacks import format_code_response
        from navine.code.generate import generate_code

        code, lang = generate_code(str(task or ""))
        return format_code_response(code, lang or "python")

    @mcp.tool()
    def discord_webhook_status() -> dict:
        """Show whether a Discord webhook URL is configured."""
        from navine.integrations.discord_webhook import webhook_status

        return webhook_status()

    @mcp.tool()
    def discord_webhook_send(content: str, url: str = "") -> dict:
        """Send a message through the configured Discord webhook (optional URL override)."""
        from navine.integrations.discord_webhook import send_webhook
        from navine.utils.brand import load_brand

        brand = load_brand()
        return send_webhook(
            str(content or ""),
            webhook_url=str(url or "").strip() or None,
            username=str(brand.get("name") or "Navine AI - Python"),
        )

    @mcp.tool()
    def mcp_client_status() -> dict:
        """Status of Navine's MCP client and configured remote servers."""
        from navine.mcp.client import status

        return status()

    @mcp.tool()
    def game_bridge_info() -> dict:
        """Game mod bridge protocol info for Minecraft, Fortnite, and other games."""
        from navine.api.game_routes import games_bridge_info

        return games_bridge_info()

    @mcp.tool()
    def game_coach(title: str, mode: str = "learn", note: str = "") -> str:
        """Coach for Minecraft, Fortnite, PVP, or speedrun play."""
        from navine.learn.gameplay import coach_advice, pvp_advice, speedrun_advice

        mode = (mode or "learn").lower()
        if mode == "pvp":
            return pvp_advice(title, {"note": note})
        if mode == "speedrun":
            return speedrun_advice(title, {"note": note})
        return coach_advice(title, note)

    @mcp.tool()
    def game_ingest_event(title: str, event_type: str, payload_json: str = "{}", mode: str = "learn") -> dict:
        """Ingest structured gameplay event from a mod for learning (JSON payload)."""
        from navine.learn.gameplay import ingest_game_event
        import json as _json

        try:
            payload = _json.loads(payload_json or "{}")
        except Exception:
            payload = {"raw": payload_json}
        payload.setdefault("title", title)
        return ingest_game_event(title, event_type, payload, mode=mode)

    @mcp.tool()
    def game_speedrun_split(title: str, segment: str, time_ms: int, delta_ms: int = 0) -> dict:
        """Log a speedrun split for Minecraft or other games."""
        from navine.learn.gameplay import ingest_speedrun_split

        return ingest_speedrun_split(title, segment, int(time_ms), int(delta_ms))

    @mcp.resource("navine://runtime")
    def runtime_resource() -> str:
        """Plain-text runtime placement summary."""
        info = runtime_info()
        return str(info.get("summary") or info)

    @mcp.prompt()
    def coding_prompt(task: str) -> str:
        """Prompt template for Python stdlib CLI coding tasks."""
        return (
            "Write complete runnable Python 3 CLI code using the standard library. "
            "Discord webhooks may use urllib.request. Discord bots may use discord.py. "
            f"Task: {task}"
        )

    return mcp


def _require_sdk() -> None:
    if not mcp_sdk_available():
        raise RuntimeError("MCP SDK missing. Install with: pip install 'mcp>=1.28,<3'")


def serve(transport: Optional[str] = None, **kwargs: Any) -> None:
    cfg = load_mcp_config()
    serve_cfg = dict(cfg.get("serve") or {})
    mode = str(transport or serve_cfg.get("transport") or "stdio").strip().lower()
    mcp = build_server()
    if mode in ("http", "streamable-http", "streamable_http"):
        host = str(kwargs.get("host") or serve_cfg.get("http_host") or "127.0.0.1")
        port = int(kwargs.get("port") or serve_cfg.get("http_port") or 8767)
        mcp.run(transport="streamable-http", host=host, port=port)
        return
    if mode == "sse":
        host = str(kwargs.get("host") or serve_cfg.get("http_host") or "127.0.0.1")
        port = int(kwargs.get("port") or serve_cfg.get("http_port") or 8767)
        mcp.run(transport="sse", host=host, port=port)
        return
    mcp.run(transport="stdio")


def cursor_mcp_snippet() -> Dict[str, Any]:
    cfg = load_mcp_config()
    serve_cfg = dict(cfg.get("serve") or {})
    from navine.utils.paths import get_project_root
    import sys

    root = get_project_root()
    pkg = _package_root()
    py = root / "venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = root / "venv" / "bin" / "python"
    command = str(py if py.exists() else sys.executable)
    return {
        "mcpServers": {
            pkg: {
                "command": command,
                "args": ["-m", f"{pkg}.mcp.server"],
                "cwd": str(root),
            },
            f"{pkg}-http": {
                "url": f"http://{serve_cfg.get('http_host') or '127.0.0.1'}:{int(serve_cfg.get('http_port') or 8767)}{serve_cfg.get('path') or '/mcp'}"
            },
        }
    }


if __name__ == "__main__":
    serve()
