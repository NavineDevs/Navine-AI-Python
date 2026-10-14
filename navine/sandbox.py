from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from navine.utils.paths import get_project_root

_DEFAULT: Dict[str, Any] = {
    "enabled": True,
    "mode": "contained",
    "allow_shell": False,
    "allow_open_app": False,
    "allow_open_url": True,
    "allow_file_write": False,
    "allow_file_delete": False,
    "allow_desktop_input": False,
    "allow_media_control": False,
    "allow_clipboard_write": False,
    "workspace_only": True,
    "workspace_roots": [
        ".",
        "outputs",
        "data",
        "logs",
        "checkpoints",
        "web",
        "configs",
        "navine",
    ],
    "blocked_command_patterns": [
        "format ",
        "del /s",
        "rd /s",
        "rmdir /s",
        "rm -rf",
        "rm -r",
        "mkfs",
        ":(){",
        "diskpart",
        "cipher /w",
        "reg delete",
        "reg add",
        "shutdown",
        "restart-computer",
        "stop-computer",
        "taskkill",
        "remove-item -recurse",
        "ri -r",
        "erase ",
        "format.com",
        "bcdedit",
        "net user",
        "net localgroup",
        "powershell -enc",
        "curl |",
        "wget |",
        "invoke-expression",
        "iex ",
        "start-process",
    ],
}

_cache: Optional[Dict[str, Any]] = None


def load_sandbox_config() -> Dict[str, Any]:
    global _cache
    if _cache is not None:
        return _cache
    cfg = dict(_DEFAULT)
    path = get_project_root() / "configs" / "sandbox.yaml"
    if path.exists():
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                cfg.update(loaded)
        except Exception:
            pass
    if str(cfg.get("mode") or "").lower() in ("off", "disabled", "none"):
        cfg["enabled"] = False
    cfg["enabled"] = bool(cfg.get("enabled", True))
    _cache = cfg
    return cfg


def clear_sandbox_cache() -> None:
    global _cache
    _cache = None


def sandbox_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    cfg = config or load_sandbox_config()
    return bool(cfg.get("enabled", True))


def _workspace_roots(cfg: Dict[str, Any]) -> List[Path]:
    root = get_project_root().resolve()
    roots: List[Path] = [root]
    for rel in cfg.get("workspace_roots") or []:
        path = Path(str(rel))
        if not path.is_absolute():
            path = root / path
        try:
            roots.append(path.resolve())
        except Exception:
            continue
    return roots


def path_in_workspace(path: str | Path, config: Optional[Dict[str, Any]] = None) -> bool:
    cfg = config or load_sandbox_config()
    try:
        target = Path(path).expanduser().resolve()
    except Exception:
        return False
    for root in _workspace_roots(cfg):
        try:
            target.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def guard_path(path: str | Path, *, write: bool = False, config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    cfg = config or load_sandbox_config()
    if not sandbox_enabled(cfg):
        return None
    if write and not bool(cfg.get("allow_file_write", False)):
        return "Sandbox blocked file write/change. Contained mode only allows safe chat, code replies, and image/video generation."
    if bool(cfg.get("workspace_only", True)) and not path_in_workspace(path, cfg):
        return f"Sandbox blocked path outside the AI workspace: {path}"
    return None


def guard_delete(path: str | Path, config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    cfg = config or load_sandbox_config()
    if not sandbox_enabled(cfg):
        return None
    if not bool(cfg.get("allow_file_delete", False)):
        return "Sandbox blocked file delete. Contained mode cannot remove files on your laptop."
    return guard_path(path, write=True, config=cfg)


def guard_shell(command: str, config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    cfg = config or load_sandbox_config()
    if not sandbox_enabled(cfg):
        return None
    if not bool(cfg.get("allow_shell", False)):
        return "Sandbox blocked shell/command execution. Contained mode cannot run system commands on your laptop."
    lowered = str(command or "").lower()
    for pattern in cfg.get("blocked_command_patterns") or []:
        if str(pattern).lower() in lowered:
            return f"Sandbox blocked dangerous command pattern: {pattern}"
    return None


def guard_capability(name: str, config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    cfg = config or load_sandbox_config()
    if not sandbox_enabled(cfg):
        return None
    key = str(name or "").strip().lower()
    mapping = {
        "run_command": ("allow_shell", "Sandbox blocked shell/command execution."),
        "open_app": ("allow_open_app", "Sandbox blocked opening apps."),
        "open_url": ("allow_open_url", "Sandbox blocked opening URLs."),
        "file_write": ("allow_file_write", "Sandbox blocked writing files on your laptop."),
        "file_delete": ("allow_file_delete", "Sandbox blocked deleting files on your laptop."),
        "desktop_input": ("allow_desktop_input", "Sandbox blocked keyboard/mouse control."),
        "media_control": ("allow_media_control", "Sandbox blocked media key control."),
        "clipboard": ("allow_clipboard_write", "Sandbox blocked clipboard write."),
    }
    if key not in mapping:
        return None
    flag, message = mapping[key]
    if not bool(cfg.get(flag, False)):
        return message
    return None


def sandbox_status(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_sandbox_config()
    return {
        "enabled": sandbox_enabled(cfg),
        "mode": cfg.get("mode", "contained"),
        "allow_shell": bool(cfg.get("allow_shell", False)),
        "allow_file_write": bool(cfg.get("allow_file_write", False)),
        "allow_file_delete": bool(cfg.get("allow_file_delete", False)),
        "allow_desktop_input": bool(cfg.get("allow_desktop_input", False)),
        "allow_open_app": bool(cfg.get("allow_open_app", False)),
        "allow_open_url": bool(cfg.get("allow_open_url", True)),
        "workspace_only": bool(cfg.get("workspace_only", True)),
        "workspace": str(get_project_root()),
    }


def deny(message: str) -> Dict[str, Any]:
    return {"ok": False, "error": message, "sandboxed": True}
