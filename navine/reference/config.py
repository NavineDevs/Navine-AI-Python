from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from navine.utils.paths import get_project_root

_config_cache: Optional[Dict[str, Any]] = None

DEFAULT_REFERENCE_KEYWORDS: Dict[str, Dict[str, Any]] = {
    "hentai": {
        "dir": "data/nsfw/local/hentai",
        "keywords": ["hentai", "anime", "waifu", "ecchi", "doujin", "2d", "cartoon"],
    },
    "porn": {
        "dir": "data/nsfw/local/porn",
        "keywords": ["porn", "xxx", "nsfw", "nude", "naked", "explicit"],
    },
    "real": {
        "dir": "data/nsfw/local/real",
        "keywords": ["real", "photo", "photoreal", "amateur", "webcam"],
    },
    "human": {
        "dir": "data/nsfw/local/human",
        "keywords": ["human", "girl", "woman", "man", "person"],
    },
    "animated": {
        "dir": "data/nsfw/local/animated",
        "keywords": ["animated", "gif", "2d animation", "animation"],
    },
    "mixed": {
        "dir": "data/nsfw/local/mixed",
        "keywords": ["mixed"],
    },
    "cosplay": {
        "dir": "data/nsfw/local/cosplay",
        "keywords": ["cosplay", "costume", "cosplayer"],
    },
    "fetish": {
        "dir": "data/nsfw/local/fetish",
        "keywords": ["fetish", "kink", "bdsm", "latex"],
    },
}

DEFAULT_REFERENCE_SETTINGS: Dict[str, Any] = {
    "mode": "guide",
    "guide_strength": 0.35,
    "heavy_strength": 0.75,
    "require_nsfw_sidecar": True,
    "prefer_infini_refs": True,
    "default_strength": 0.58,
    "category_strength": {
        "hentai": 0.68,
        "porn": 0.82,
        "real": 0.80,
        "human": 0.75,
        "animated": 0.70,
        "mixed": 0.60,
        "cosplay": 0.72,
        "fetish": 0.74,
    },
    "img2img_lite": {
        "enabled": True,
        "noise_strength": 0.22,
        "steps_scale": 0.55,
    },
    "auto_train_on_reference": {
        "enabled": True,
        "min_uses_per_category": 12,
    },
    "video_motion_frames": 12,
    "prompt_similarity": True,
}


def _merge_keyword_lists(category: str, spec: Dict[str, Any]) -> List[str]:
    raw = [str(k).strip().lower() for k in (spec.get("keywords") or []) if str(k).strip()]
    auto = str(category).strip().lower()
    if auto and auto not in raw:
        raw.insert(0, auto)
    seen: List[str] = []
    for item in raw:
        if item not in seen:
            seen.append(item)
    return seen


def _normalize_entry(category: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    folder = str(spec.get("dir") or spec.get("path") or f"data/nsfw/local/{category}")
    return {
        "category": category,
        "dir": folder,
        "keywords": _merge_keyword_lists(category, spec),
    }


def _discover_local_categories(entries: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    root = get_project_root() / "data" / "nsfw" / "local"
    if not root.exists():
        return entries
    merged = dict(entries)
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        name = path.name.lower()
        if name in merged:
            continue
        merged[name] = _normalize_entry(name, {"dir": str(path.relative_to(get_project_root()).as_posix())})
    return merged


def _load_nsfw_yaml() -> Dict[str, Any]:
    nsfw_path = get_project_root() / "configs" / "nsfw.yaml"
    if not nsfw_path.exists():
        return {}
    with open(nsfw_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_reference_config(reload: bool = False) -> Dict[str, Any]:
    global _config_cache
    if _config_cache is not None and not reload:
        return _config_cache
    entries: Dict[str, Dict[str, Any]] = {}
    for category, spec in DEFAULT_REFERENCE_KEYWORDS.items():
        entries[category] = _normalize_entry(category, dict(spec))
    nsfw_cfg = _load_nsfw_yaml()
    configured = nsfw_cfg.get("reference_keywords") or {}
    if isinstance(configured, dict):
        for category, spec in configured.items():
            if isinstance(spec, dict):
                entries[str(category)] = _normalize_entry(str(category), spec)
            elif isinstance(spec, str):
                entries[str(category)] = _normalize_entry(str(category), {"dir": spec})
    ref_path = get_project_root() / "configs" / "reference.yaml"
    if ref_path.exists():
        with open(ref_path, "r", encoding="utf-8") as handle:
            ref_cfg = yaml.safe_load(handle) or {}
        configured = ref_cfg.get("reference_keywords") or ref_cfg
        if isinstance(configured, dict):
            for category, spec in configured.items():
                if category == "reference_keywords":
                    continue
                if isinstance(spec, dict):
                    entries[str(category)] = _normalize_entry(str(category), spec)
                elif isinstance(spec, str):
                    entries[str(category)] = _normalize_entry(str(category), {"dir": spec})
    entries = _discover_local_categories(entries)
    settings = dict(DEFAULT_REFERENCE_SETTINGS)
    nsfw_settings = nsfw_cfg.get("reference_settings") or {}
    if isinstance(nsfw_settings, dict):
        for key, value in nsfw_settings.items():
            if isinstance(value, dict) and isinstance(settings.get(key), dict):
                merged = dict(settings[key])
                merged.update(value)
                settings[key] = merged
            else:
                settings[key] = value
    _config_cache = {"reference_keywords": entries, "reference_settings": settings}
    return _config_cache


def list_reference_entries() -> Dict[str, Dict[str, Any]]:
    cfg = load_reference_config()
    return dict(cfg.get("reference_keywords") or {})


def get_reference_settings() -> Dict[str, Any]:
    cfg = load_reference_config()
    settings = dict(DEFAULT_REFERENCE_SETTINGS)
    loaded = cfg.get("reference_settings") or {}
    if isinstance(loaded, dict):
        for key, value in loaded.items():
            if isinstance(value, dict) and isinstance(settings.get(key), dict):
                merged = dict(settings[key])
                merged.update(value)
                settings[key] = merged
            else:
                settings[key] = value
    return settings


def reference_strength_for_category(category: str, has_checkpoint: bool = False) -> float:
    settings = get_reference_settings()
    mode = str(settings.get("mode") or "guide").lower()
    if mode == "off":
        return 0.0
    if mode == "guide":
        strength = float(settings.get("guide_strength", 0.35))
        cat = str(category).strip().lower()
        if cat in ("real", "porn", "human"):
            strength = max(strength, float((settings.get("category_strength") or {}).get(cat, strength)))
        if has_checkpoint:
            return max(0.18, min(0.58, strength))
        return max(0.15, min(0.62, strength))
    if mode == "heavy":
        heavy = float(settings.get("heavy_strength", 0.75))
        default = float(settings.get("default_strength", 0.58))
        per_cat = settings.get("category_strength") or {}
        cat = str(category).strip().lower()
        if isinstance(per_cat, dict) and cat in per_cat:
            return max(0.15, min(0.90, float(per_cat[cat])))
        return max(0.15, min(0.90, heavy if not has_checkpoint else default))
    default = float(settings.get("default_strength", 0.58))
    per_cat = settings.get("category_strength") or {}
    if isinstance(per_cat, dict):
        cat = str(category).strip().lower()
        if cat in per_cat:
            return max(0.15, min(0.90, float(per_cat[cat])))
        if cat in ("anime", "cartoon", "2d"):
            return max(0.15, min(0.90, float(per_cat.get("hentai", default))))
    return max(0.15, min(0.90, default))


def ensure_reference_dirs() -> None:
    for entry in list_reference_entries().values():
        path = get_project_root() / str(entry["dir"])
        path.mkdir(parents=True, exist_ok=True)


def reload_reference_config() -> Dict[str, Any]:
    return load_reference_config(reload=True)
