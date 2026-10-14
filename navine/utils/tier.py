from typing import Any, Dict, Optional

from navine.utils.config import load_config
from navine.utils.paths import get_checkpoint_dir


def enterprise_settings() -> Dict[str, Any]:
    try:
        navine = load_config("navine")
    except FileNotFoundError:
        return {}
    return dict(navine.get("enterprise") or {})


def active_tier() -> str:
    tier = str(enterprise_settings().get("active_tier", "standard")).strip().lower()
    if tier in ("enterprise", "ent"):
        return "enterprise"
    return "standard"


def modality_config_name(modality: str) -> str:
    if active_tier() == "enterprise":
        candidate = f"{modality}_enterprise"
        root_name = __import__("pathlib").Path(__file__).resolve().parents[2]
        if (root_name / "configs" / f"{candidate}.yaml").exists():
            return candidate
    return modality


def load_modality_config(modality: str) -> Dict[str, Any]:
    return load_config(modality_config_name(modality))


def checkpoint_namespace(modality: str, config: Optional[Dict[str, Any]] = None) -> str:
    if config and config.get("checkpoint_namespace"):
        return str(config["checkpoint_namespace"])
    if active_tier() == "enterprise":
        return f"{modality}_enterprise"
    return modality


def resolve_checkpoint_dir(modality: str, config: Optional[Dict[str, Any]] = None):
    return get_checkpoint_dir(checkpoint_namespace(modality, config))
