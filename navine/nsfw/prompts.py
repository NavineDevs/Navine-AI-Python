import re
from typing import Dict, Optional

NSFW_TERMS = (
    "nsfw",
    "hentai",
    "ecchi",
    "lewd",
    "nude",
    "naked",
    "explicit",
    "porn",
    "xxx",
    "erotic",
    "sexual",
    "adult",
    "r18",
    "r-18",
    "uncensored",
    "rule34",
    "doujin",
    "ero",
    "nsfl",
)

ANIME_TERMS = (
    "anime",
    "manga",
    "waifu",
    "2d",
    "illustration",
    "doujinshi",
)

HENTAI_TERMS = (
    "hentai",
    "ecchi",
    "lewd",
    "rule34",
    "doujin",
    "doujinshi",
    "ero",
)

REAL_TERMS = (
    "real",
    "photo",
    "photoreal",
    "photography",
    "amateur",
    "live action",
    "realistic",
)

HUMAN_TERMS = (
    "human",
    "woman",
    "man",
    "girl",
    "boy",
    "lady",
    "guy",
    "model",
    "person",
    "people",
)

CHARACTER_TERMS = (
    "girl",
    "woman",
    "boy",
    "man",
    "character",
    "waifu",
    "portrait",
    "face",
    "body",
    "figure",
)

CATEGORY_ENHANCEMENTS = {
    "hentai": "anime illustration, hentai style, detailed linework, vibrant colors, explicit adult",
    "anime": "anime illustration, stylized character art, vibrant colors, detailed linework",
    "porn": "photorealistic explicit adult photo, natural lighting, detailed skin texture, sharp focus",
    "real": "photorealistic adult content, natural lighting, detailed skin texture",
    "human": "human subject, photorealistic portrait, natural pose, detailed portrait lighting",
    "mixed": "photorealistic adult content, natural lighting, high detail",
    "general": "photorealistic explicit adult content, detailed, high quality",
}


def is_nsfw_prompt(prompt: str) -> bool:
    lower = prompt.lower()
    if any(term in lower for term in NSFW_TERMS):
        return True
    if any(term in lower for term in HENTAI_TERMS):
        return True
    if any(term in lower for term in ANIME_TERMS) and any(term in lower for term in CHARACTER_TERMS):
        return True
    return False


def detect_nsfw_category(prompt: str) -> str:
    lower = prompt.lower()
    if any(term in lower for term in HENTAI_TERMS):
        return "hentai"
    if any(term in lower for term in ANIME_TERMS):
        return "anime"
    if any(term in lower for term in REAL_TERMS):
        return "real"
    if any(term in lower for term in ("porn", "xxx")):
        return "porn"
    if any(term in lower for term in HUMAN_TERMS):
        return "human"
    if any(term in lower for term in NSFW_TERMS):
        return "general"
    return "general"


def has_character_subject(prompt: str) -> bool:
    lower = prompt.lower()
    return any(term in lower for term in CHARACTER_TERMS) or any(term in lower for term in ANIME_TERMS)


def analyze_prompt(prompt: str) -> Dict[str, object]:
    lower = prompt.lower()
    category = detect_nsfw_category(prompt)
    anime = any(term in lower for term in ANIME_TERMS) or category in ("hentai", "anime")
    return {
        "nsfw": is_nsfw_prompt(prompt),
        "category": category,
        "anime": anime,
        "character": has_character_subject(prompt),
        "real": category in ("real", "human"),
        "motion": any(w in lower for w in ("danc", "mov", "animat", "walk", "run", "sway", "bounce", "wave")),
    }


def enhance_nsfw_prompt(prompt: str) -> str:
    text = prompt.strip()
    if not text:
        return text
    info = analyze_prompt(text)
    if not info["nsfw"] and not info["anime"]:
        return text
    category = str(info["category"])
    suffix = CATEGORY_ENHANCEMENTS.get(category, CATEGORY_ENHANCEMENTS["general"])
    lower = text.lower()
    has_quality = any(w in lower for w in ("quality", "detailed", "sharp", "high res", "4k"))
    has_article = lower.startswith(("a ", "an ", "the "))
    if info["character"] and not has_article:
        text = f"an {text}"
    if suffix.lower() not in lower:
        text = f"{text}, {suffix}"
    if not has_quality:
        text = f"{text}, high quality, detailed"
    return text
