import re
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional

from navine.utils.paths import get_project_root

THEME_PRESETS: Dict[str, Dict[str, str]] = {
    "navine": {
        "primary": "#1246FF",
        "secondary": "#00B4FF",
        "accent": "#1E90FF",
        "royal": "#1A237E",
        "background": "#05070F",
    },
    "navuryx": {
        "primary": "#1A237E",
        "secondary": "#5E35B1",
        "accent": "#7C3AED",
        "royal": "#1246FF",
        "background": "#07050F",
    },
    "hitboy": {
        "primary": "#1246FF",
        "secondary": "#7C3AED",
        "accent": "#5E35B1",
        "royal": "#1A237E",
        "background": "#06050C",
    },
}

DEFAULT_BRAND: Dict[str, Any] = {
    "display_name": "Navine AI - Python",
    "engine_name": "Navine Engine",
    "package": "navine",
    "theme_id": "navine",
    "theme": dict(THEME_PRESETS["navine"]),
    "creator": "HitBoyXx23",
    "company": "Navine",
    "default_text_model": "text_enterprise",
    "port": 8765,
    "backend": "python",
    "python_only": False,
    "modes": ["chat", "code", "think", "detective", "analyze", "osint", "conscience"],
    "model_ids": [
        "text_enterprise",
        "text_code",
        "image_enterprise",
        "image_enterprise_v2",
        "video_enterprise",
        "voice",
        "deepfake",
    ],
}

FULL_UI_TABS = [
    "text",
    "voicechat",
    "image",
    "video",
    "osint",
    "deepfake",
    "voice",
    "music",
    "info",
    "train",
    "apigen",
    "settings",
]

PYTHON_UI_TABS = [
    "text",
    "voicechat",
    "image",
    "video",
    "osint",
    "deepfake",
    "voice",
    "music",
    "info",
    "settings",
]

DEFAULT_UI_FEATURES: Dict[str, Any] = {
    "train": True,
    "apigen": True,
    "train_strip": True,
    "deepfake": True,
    "native_accelerators": True,
    "footer_ops": True,
}

PYTHON_UI_FEATURES: Dict[str, Any] = {
    "train": True,
    "apigen": True,
    "train_strip": True,
    "deepfake": True,
    "native_accelerators": False,
    "footer_ops": True,
}

PYTHON_TRAIN_DENY = {
    "full",
    "all",
    "hitboyx23",
    "hitboyx23_python",
    "hitboyx23_ai",
    "image-lora",
    "video-lora",
}

MIXED_ONLY_MODELS = {"hitboyx23_ai"}


def _infer_theme_id(display_name: str, package: str, explicit: Any = None) -> str:
    key = str(explicit or "").strip().lower()
    if key in THEME_PRESETS:
        return key
    blob = f"{display_name} {package}".lower()
    if "navuryx" in blob:
        return "navuryx"
    if "hitboy" in blob:
        return "hitboy"
    return "navine"


def _normalize_theme(theme_id: str, theme: Any) -> Dict[str, str]:
    base = dict(THEME_PRESETS.get(theme_id) or THEME_PRESETS["navine"])
    if isinstance(theme, dict):
        for key, value in theme.items():
            if value is None:
                continue
            text = str(value).strip()
            if text:
                base[str(key)] = text
    return base


@lru_cache(maxsize=1)
def load_brand() -> Dict[str, Any]:
    brand = dict(DEFAULT_BRAND)
    brand["theme"] = dict(THEME_PRESETS["navine"])
    path = get_project_root() / "configs" / "brand.yaml"
    if path.exists():
        try:
            import yaml

            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                brand.update({k: v for k, v in loaded.items() if v is not None})
        except Exception:
            pass
    brand["display_name"] = str(brand.get("display_name") or DEFAULT_BRAND["display_name"]).strip()
    brand["engine_name"] = str(brand.get("engine_name") or DEFAULT_BRAND["engine_name"]).strip()
    brand["package"] = str(brand.get("package") or DEFAULT_BRAND["package"]).strip()
    brand["creator"] = str(brand.get("creator") or DEFAULT_BRAND.get("creator") or "HitBoyXx23").strip()
    theme_id = _infer_theme_id(brand["display_name"], brand["package"], brand.get("theme_id"))
    brand["theme_id"] = theme_id
    brand["theme"] = _normalize_theme(theme_id, brand.get("theme"))
    company_raw = brand.get("company")
    if company_raw is not None and str(company_raw).strip().lower() in ("", "none", "null", "false"):
        brand["company"] = None
    elif company_raw is None:
        if "hitboy" in theme_id:
            brand["company"] = None
        elif "navuryx" in theme_id:
            brand["company"] = "Navuryx"
        else:
            brand["company"] = "Navine"
    else:
        brand["company"] = str(company_raw).strip()
    brand["default_text_model"] = str(
        brand.get("default_text_model") or DEFAULT_BRAND["default_text_model"]
    ).strip()
    try:
        brand["port"] = int(brand.get("port") or DEFAULT_BRAND["port"])
    except (TypeError, ValueError):
        brand["port"] = int(DEFAULT_BRAND["port"])
    if not isinstance(brand.get("modes"), list) or not brand["modes"]:
        brand["modes"] = list(DEFAULT_BRAND["modes"])
    if not isinstance(brand.get("model_ids"), list) or not brand["model_ids"]:
        brand["model_ids"] = list(DEFAULT_BRAND["model_ids"])
    if theme_id == "navine" and int(brand.get("port") or DEFAULT_BRAND["port"]) == int(DEFAULT_BRAND["port"]):
        brand["model_ids"] = [
            mid
            for mid in brand["model_ids"]
            if "hitboy" not in str(mid).lower() and "navuryx" not in str(mid).lower()
        ] or list(DEFAULT_BRAND["model_ids"])
    brand["backend"] = str(brand.get("backend") or DEFAULT_BRAND["backend"]).strip().lower() or "python"
    brand["python_only"] = bool(brand.get("python_only")) or _infer_python_only(brand)
    if brand["python_only"]:
        brand["model_ids"] = [
            mid for mid in brand["model_ids"] if str(mid).lower() not in MIXED_ONLY_MODELS
        ] or list(DEFAULT_BRAND["model_ids"])
    tabs = brand.get("ui_tabs")
    if not isinstance(tabs, list) or not tabs:
        brand["ui_tabs"] = list(PYTHON_UI_TABS if brand["python_only"] else FULL_UI_TABS)
    else:
        brand["ui_tabs"] = [str(t).strip() for t in tabs if str(t).strip()]
    if "settings" not in brand["ui_tabs"]:
        brand["ui_tabs"].append("settings")
    base_features = dict(PYTHON_UI_FEATURES if brand["python_only"] else DEFAULT_UI_FEATURES)
    extra_features = brand.get("ui_features") if isinstance(brand.get("ui_features"), dict) else {}
    base_features.update({k: v for k, v in extra_features.items() if v is not None})
    brand["ui_features"] = base_features
    return brand


def _infer_python_only(brand: Dict[str, Any]) -> bool:
    if str(brand.get("backend") or "").lower() != "python":
        return False
    name = str(brand.get("display_name") or "").lower()
    package = str(brand.get("package") or "").lower()
    if "- python" in name or name.endswith(" python"):
        return True
    if package.endswith("_python") or package.endswith("_pai") or package == "navine_pai":
        return True
    if " pai" in f" {name}" or name.endswith(" pai"):
        return True
    return False


def is_python_only() -> bool:
    return bool(load_brand().get("python_only"))


def is_mixed_backend() -> bool:
    return str(load_brand().get("backend") or "").lower() == "mixed"


def brand_backend() -> str:
    return str(load_brand().get("backend") or "python")


def brand_ui_tabs() -> List[str]:
    tabs = load_brand().get("ui_tabs") or FULL_UI_TABS
    return [str(t) for t in tabs]


def brand_ui_features() -> Dict[str, Any]:
    features = load_brand().get("ui_features") or {}
    return dict(features) if isinstance(features, dict) else dict(DEFAULT_UI_FEATURES)


def python_train_denied(target: str) -> bool:
    if not is_python_only():
        return False
    key = str(target or "").strip().lower().replace("_", "-")
    return key in PYTHON_TRAIN_DENY or "hitboyx23" in key and not key.endswith("-python")


def brand_name() -> str:
    return str(load_brand()["display_name"])


def brand_package() -> str:
    return str(load_brand()["package"])


def brand_port() -> int:
    return int(load_brand()["port"])


def default_text_model() -> str:
    return str(load_brand()["default_text_model"])


def brand_theme_id() -> str:
    return str(load_brand().get("theme_id") or "navine")


def brand_theme() -> Dict[str, str]:
    theme = load_brand().get("theme") or {}
    if isinstance(theme, dict):
        return {str(k): str(v) for k, v in theme.items()}
    return dict(THEME_PRESETS["navine"])


def theme_foreign_tokens(theme_id: Optional[str] = None) -> tuple:
    tid = str(theme_id or brand_theme_id() or "navine").lower()
    if "hitboy" in tid:
        return ("navuryx",)
    if "navuryx" in tid:
        return ("hitboy",)
    return ("hitboy", "navuryx")


def is_foreign_brand_text(value: Any, theme_id: Optional[str] = None) -> bool:
    blob = str(value or "").lower()
    if not blob:
        return False
    return any(token in blob for token in theme_foreign_tokens(theme_id))


def filter_foreign_brand(
    items: Iterable[Any],
    getter=None,
    theme_id: Optional[str] = None,
) -> List[Any]:
    out: List[Any] = []
    for item in items:
        raw = getter(item) if getter else item
        if is_foreign_brand_text(raw, theme_id):
            continue
        out.append(item)
    return out


def brand_creator() -> str:
    return str(load_brand().get("creator") or "HitBoyXx23").strip()


def brand_company() -> Optional[str]:
    company = load_brand().get("company")
    if company is None:
        return None
    text = str(company).strip()
    return text or None


def is_who_made_query(message: str) -> bool:
    return bool(
        re.search(
            r"\bwho\s+(?:made|created|built|developed|designed|programmed|invented|trained|coded)\s+(?:you|navine|navuryx|hitboy)\b",
            str(message or ""),
            re.I,
        )
        or re.search(
            r"\bwho(?:'s| is| are)\s+your\s+(?:creator|maker|developer|author|owner|founder|company)\b",
            str(message or ""),
            re.I,
        )
    )


def brand_identity_reply(message: str = "", *, straight: bool = False) -> str:
    brand = load_brand()
    name = str(brand.get("display_name") or "Navine AI - Python").strip()
    creator = str(brand.get("creator") or "HitBoyXx23").strip() or "HitBoyXx23"
    company_raw = brand.get("company")
    company = str(company_raw).strip() if company_raw else None
    if company and company.lower() in {creator.lower(), name.lower(), "none", "null"}:
        company = None
    who_made = is_who_made_query(message)

    if straight or who_made:
        if company:
            return f"{creator} made me. {company} is the company name; {creator} is my creator."
        return f"{creator} made me."

    try:
        from navine.utils.runtime_place import runtime_identity_line

        place = runtime_identity_line()
    except Exception:
        place = (
            "The chat UI runs in your browser on your device. "
            "My models run on the Navine host that serves this site."
        )

    if company:
        return (
            f"I'm {name}. {creator} made me, and {company} is the company behind me. "
            f"{place} "
            "I chat, write code, and generate images and videos with my own custom-trained models."
        )
    return (
        f"I'm {name}. {creator} made me. "
        f"{place} "
        "I chat, write code, and generate images and videos with my own custom-trained models."
    )


def scrub_cross_brand_text(text: Any, replacement: Optional[str] = None) -> str:
    raw = str(text or "")
    if not raw:
        return raw
    name = replacement or brand_name()
    tokens = theme_foreign_tokens()
    creator = brand_creator()
    markers: List[tuple] = []
    protected = raw
    if creator:
        keep = "\0NAVINE_CREATOR\0"
        protected, count = re.subn(re.escape(creator), keep, protected, flags=re.I)
        if count:
            markers.append((keep, creator))

    parts: List[str] = []
    cursor = 0
    for match in re.finditer(r"```[\w+-]*\n.*?```", protected, flags=re.S):
        parts.append(_scrub_brand_chunk(protected[cursor : match.start()], name, tokens))
        parts.append(match.group(0))
        cursor = match.end()
    parts.append(_scrub_brand_chunk(protected[cursor:], name, tokens))
    out = "".join(parts)
    for keep, value in markers:
        out = out.replace(keep, value)
    out = _scrub_external_model_names(out, name)
    return re.sub(r" {2,}", " ", out).strip() if "```" not in raw else out


def _scrub_external_model_names(text: str, name: str) -> str:
    out = text
    patterns = (
        (r"(?i)\bchat\s*gpt\b", name),
        (r"(?i)\bgpt-?\s*3(?:\.5)?\b", "enterprise"),
        (r"(?i)\bgpt-?\s*4(?:o|\.0)?\b", name),
        (r"(?i)\bgpt-?\s*3\s*medium\+?\s*class\b", "enterprise class"),
        (r"(?i)\bgpt3_(?:800m|760m|1p3b|350m)\b", "800m"),
        (r"(?i)\bopenai\b", name),
        (r"(?i)\bclaude(?:\s*\d+(?:\.\d+)?)?\b", name),
        (r"(?i)\bgemini(?:\s*(?:pro|flash|ultra))?\b", name),
        (r"(?i)\bgrok(?:-\d+)?\b", name),
        (r"(?i)\bllama(?:\s*\d+(?:\.\d+)?)?\b", name),
        (r"(?i)\bmistral\b", name),
        (r"(?i)\bcopilot\b", name),
    )
    for pattern, repl in patterns:
        out = re.sub(pattern, repl, out)
    return out


def _scrub_brand_chunk(chunk: str, name: str, tokens: tuple) -> str:
    out = chunk
    if "hitboy" in tokens:
        out = re.sub(r"(?i)hitboyx+23_ai_python", name, out)
        out = re.sub(r"(?i)hitboyx+23_python", name, out)
        out = re.sub(r"(?i)hitboyx+23_ai", name, out)
        out = re.sub(r"(?i)\bhitboy\b(?!x)", name, out)
    if "navuryx" in tokens:
        out = re.sub(r"(?i)navuryx(?:\s*ai)?", name, out)
    return out
