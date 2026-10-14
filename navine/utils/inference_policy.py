from typing import Any, Dict, Optional


def _navine_config() -> Dict[str, Any]:
    from navine.utils.config import load_config

    try:
        return load_config("navine")
    except Exception:
        return {}


def is_custom_only() -> bool:
    navine = _navine_config()
    inference = navine.get("inference") or {}
    if inference.get("custom_only") is True:
        return True
    enterprise = navine.get("enterprise") or {}
    return bool(enterprise.get("custom_trained_only", False))
