import json
from pathlib import Path
from typing import Any, Dict, Optional

from navine.utils.paths import get_project_root


def _default_config() -> Dict[str, Any]:
    return {
        "allow_screen": True,
        "allow_input": False,
        "require_explicit_request": True,
        "max_type_chars": 2000,
        "click_delay_ms": 50,
        "audit": True,
        "output_dir": "outputs/desktop",
    }


def load_desktop_config() -> Dict[str, Any]:
    cfg = _default_config()
    try:
        from navine.utils.config import load_config

        loaded = load_config("desktop") or {}
        if isinstance(loaded, dict):
            cfg.update(loaded)
    except Exception:
        pass
    return cfg


def _session_path() -> Path:
    path = get_project_root() / "logs" / "desktop" / "session.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_session_flags() -> Dict[str, Any]:
    path = _session_path()
    if not path.exists():
        return {"screen_enabled": None, "input_enabled": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {
                "screen_enabled": data.get("screen_enabled"),
                "input_enabled": data.get("input_enabled"),
            }
    except Exception:
        pass
    return {"screen_enabled": None, "input_enabled": None}


def save_session_flags(
    screen_enabled: Optional[bool] = None,
    input_enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    current = load_session_flags()
    if screen_enabled is not None:
        current["screen_enabled"] = bool(screen_enabled)
    if input_enabled is not None:
        current["input_enabled"] = bool(input_enabled)
    path = _session_path()
    path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return current


def screen_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    cfg = config or load_desktop_config()
    session = load_session_flags()
    if session.get("screen_enabled") is not None:
        return bool(session.get("screen_enabled"))
    return bool(cfg.get("allow_screen", True))


def input_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    try:
        from navine.sandbox import guard_capability, load_sandbox_config

        if guard_capability("desktop_input", load_sandbox_config()):
            return False
    except Exception:
        pass
    cfg = config or load_desktop_config()
    session = load_session_flags()
    if session.get("input_enabled") is not None:
        return bool(session.get("input_enabled"))
    return bool(cfg.get("allow_input", False))


def set_desktop_flags(
    allow_screen: Optional[bool] = None,
    allow_input: Optional[bool] = None,
) -> Dict[str, Any]:
    return save_session_flags(screen_enabled=allow_screen, input_enabled=allow_input)


def output_dir(config: Optional[Dict[str, Any]] = None) -> Path:
    cfg = config or load_desktop_config()
    rel = str(cfg.get("output_dir") or "outputs/desktop")
    path = get_project_root() / rel
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_desktop_status() -> Dict[str, Any]:
    cfg = load_desktop_config()
    session = load_session_flags()
    return {
        "ok": True,
        "allow_screen_config": bool(cfg.get("allow_screen", True)),
        "allow_input_config": bool(cfg.get("allow_input", False)),
        "screen_enabled": screen_enabled(cfg),
        "input_enabled": input_enabled(cfg),
        "session": session,
        "require_explicit_request": bool(cfg.get("require_explicit_request", True)),
        "max_type_chars": int(cfg.get("max_type_chars") or 2000),
        "output_dir": str(output_dir(cfg)),
        "warning": (
            "Desktop input can type and click anywhere on this machine. "
            "Keep Allow input off unless you asked for automation."
        ),
    }


def audit_desktop(action: str, detail: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> None:
    cfg = config or load_desktop_config()
    if not cfg.get("audit", True):
        return
    path = get_project_root() / "logs" / "desktop" / "audit.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "detail": detail,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
