from typing import Any, Dict

import yaml

from navine.utils.paths import get_project_root

DEFAULT_GENERATION: Dict[str, Any] = {
    "path": "primary",
    "allow_external_inference": False,
    "external_for_training_only": True,
    "external_teacher_enabled": False,
}


def get_generation_settings() -> Dict[str, Any]:
    path = get_project_root() / "configs" / "navine.yaml"
    settings = dict(DEFAULT_GENERATION)
    if not path.exists():
        return settings
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    generation = dict(data.get("generation") or {})
    settings.update(generation)
    settings.setdefault("path", "primary")
    settings.setdefault("allow_external_inference", False)
    settings.setdefault("external_for_training_only", True)
    settings.setdefault("external_teacher_enabled", False)
    return settings


def get_generation_path() -> str:
    value = str(get_generation_settings().get("path") or "primary").lower()
    if value in ("secondary", "external", "diffusers", "teacher"):
        return "secondary"
    return "primary"


def allow_external_inference() -> bool:
    return False


def use_secondary_generation() -> bool:
    return False


def external_teacher_enabled() -> bool:
    settings = get_generation_settings()
    return bool(settings.get("external_teacher_enabled", False))


def require_external_teacher() -> None:
    if external_teacher_enabled():
        return
    raise RuntimeError(
        "External teacher models are disabled. Navine uses local trained weights only. "
        "Re-enable with generation.external_teacher_enabled: true in configs/navine.yaml "
        "only if you need more teacher data for training."
    )


def disable_external_teacher() -> str:
    return save_generation_settings(
        path="primary",
        allow_external_inference=False,
        external_for_training_only=True,
        external_teacher_enabled=False,
    )


def save_generation_settings(
    path: str = "primary",
    allow_external_inference: bool = False,
    external_for_training_only: bool = True,
    external_teacher_enabled: bool = True,
) -> str:
    selected = str(path or "primary").lower()
    if selected in ("secondary", "external", "diffusers", "teacher"):
        selected = "secondary"
    else:
        selected = "primary"
    config_path = get_project_root() / "configs" / "navine.yaml"
    data: Dict[str, Any] = {}
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            data = loaded
    generation = dict(data.get("generation") or {})
    generation["path"] = selected
    generation["allow_external_inference"] = bool(allow_external_inference)
    generation["external_for_training_only"] = bool(external_for_training_only)
    generation["external_teacher_enabled"] = bool(external_teacher_enabled)
    data["generation"] = generation
    config_path.write_text(yaml.safe_dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8")
    return selected
