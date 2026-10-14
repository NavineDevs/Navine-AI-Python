from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from navine.utils.paths import get_project_root

DEFAULT_API_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "bind_host": "127.0.0.1",
    "port": 8765,
    "require_auth": True,
    "allow_localhost_without_auth": True,
    "allow_same_origin_without_auth": True,
    "keys_file": ".navine/api_keys.json",
    "cors_origins": ["*"],
    "rate_limit": {
        "enabled": False,
        "requests_per_minute": 60,
    },
}


def load_api_config() -> Dict[str, Any]:
    config_path = get_project_root() / "configs" / "api.yaml"
    merged = dict(DEFAULT_API_CONFIG)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        merged.update(loaded)
        if "rate_limit" in loaded and isinstance(loaded["rate_limit"], dict):
            rate = dict(DEFAULT_API_CONFIG["rate_limit"])
            rate.update(loaded["rate_limit"])
            merged["rate_limit"] = rate
    try:
        from navine.utils.brand import load_brand

        brand = load_brand()
        if brand.get("port"):
            merged["port"] = int(brand["port"])
    except Exception:
        pass
    return merged


def get_keys_path(api_config: Optional[Dict[str, Any]] = None) -> Path:
    cfg = api_config or load_api_config()
    rel = cfg.get("keys_file", DEFAULT_API_CONFIG["keys_file"])
    path = Path(rel)
    if not path.is_absolute():
        path = get_project_root() / path
    return path


def get_key_store(api_config: Optional[Dict[str, Any]] = None):
    from navine.auth.keys import ApiKeyStore

    return ApiKeyStore(get_keys_path(api_config))
