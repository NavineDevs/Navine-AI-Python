import json
from pathlib import Path
from typing import Any, Dict, Optional

from navine.utils.paths import get_project_root

DEFAULT_SETTINGS: Dict[str, Any] = {
    "allow_emoji": True,
    "discord_webhook_url": "",
}


def _settings_dir() -> Path:
    path = get_project_root() / "data" / "user_settings"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_key(key: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in (key or "guest"))
    return cleaned[:80] or "guest"


def settings_path(user_key: str) -> Path:
    return _settings_dir() / f"{_safe_key(user_key)}.json"


def normalize_settings(raw: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = dict(DEFAULT_SETTINGS)
    if isinstance(raw, dict):
        if "allow_emoji" in raw:
            out["allow_emoji"] = bool(raw["allow_emoji"])
        if "discord_webhook_url" in raw:
            out["discord_webhook_url"] = str(raw.get("discord_webhook_url") or "").strip()
    return out


def load_user_settings(user_key: str) -> Dict[str, Any]:
    path = settings_path(user_key)
    if not path.is_file():
        return dict(DEFAULT_SETTINGS)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_SETTINGS)
    return normalize_settings(data if isinstance(data, dict) else None)


def save_user_settings(user_key: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_settings(settings)
    path = settings_path(user_key)
    path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    url = str(normalized.get("discord_webhook_url") or "").strip()
    try:
        from navine.integrations.discord_webhook import set_webhook_url

        set_webhook_url(url)
    except Exception:
        pass
    return normalized
