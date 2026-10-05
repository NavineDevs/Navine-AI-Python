import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from navine.utils.paths import get_project_root

MEDIA_CATEGORIES = (
    "hentai",
    "real",
    "human",
    "animated",
    "mixed",
    "general",
)

SUBJECT_TYPES = (
    "human",
    "character",
    "mixed",
    "unknown",
)

STYLE_TYPES = (
    "illustration",
    "photography",
    "3d",
    "anime",
    "mixed",
    "unknown",
)

HENTAI_HINTS = (
    "hentai",
    "rule34",
    "danbooru",
    "gelbooru",
    "e621",
    "nhentai",
    "hanime",
    "ecchi",
    "animemidriff",
    "doujin",
    "anime",
    "manga",
    "2d",
)

REAL_HINTS = (
    "realgirls",
    "gonewild",
    "nsfwhardcore",
    "real",
    "amateur",
    "photography",
    "live",
    "porn",
)

HUMAN_HINTS = (
    "human",
    "realgirls",
    "gonewild",
    "amateur",
    "model",
    "couple",
    "solo",
)

ANIMATED_HINTS = (
    "animated",
    "gif",
    "cartoon",
    "3d",
    "sfm",
    "blender",
)


def _normalize_tags(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = re.split(r"[\s,;|]+", raw.strip())
        return [p.lower() for p in parts if p.strip()]
    if isinstance(raw, list):
        return [str(t).strip().lower() for t in raw if str(t).strip()]
    return []


def _hint_match(text: str, hints: tuple) -> int:
    lower = text.lower()
    return sum(1 for hint in hints if hint in lower)


def infer_media_category(
    subreddit: str = "",
    tags: Optional[List[str]] = None,
    source: str = "",
    url: str = "",
    configured: Optional[str] = None,
) -> str:
    if configured and configured in MEDIA_CATEGORIES:
        return configured
    combined = " ".join(
        [
            subreddit or "",
            " ".join(tags or []),
            source or "",
            url or "",
        ]
    ).lower()
    hentai_score = _hint_match(combined, HENTAI_HINTS)
    real_score = _hint_match(combined, REAL_HINTS)
    human_score = _hint_match(combined, HUMAN_HINTS)
    animated_score = _hint_match(combined, ANIMATED_HINTS)
    if hentai_score > 0 and real_score > 0:
        return "mixed"
    if hentai_score >= real_score and hentai_score > 0:
        return "hentai"
    if human_score > 0 or real_score > 0:
        if human_score >= real_score:
            return "human"
        return "real"
    if animated_score > 0:
        return "animated"
    return "general"


def infer_subject_type(media_category: str, tags: Optional[List[str]] = None) -> str:
    tag_text = " ".join(tags or []).lower()
    if media_category in ("real", "human"):
        return "human"
    if media_category == "hentai":
        return "character"
    if "human" in tag_text and "character" in tag_text:
        return "mixed"
    if "human" in tag_text:
        return "human"
    if "character" in tag_text or "anime" in tag_text:
        return "character"
    return "unknown"


def infer_style_type(media_category: str, tags: Optional[List[str]] = None) -> str:
    tag_text = " ".join(tags or []).lower()
    if media_category in ("real", "human"):
        return "photography"
    if media_category == "hentai":
        return "anime"
    if "3d" in tag_text or "sfm" in tag_text:
        return "3d"
    if "photo" in tag_text:
        return "photography"
    if "anime" in tag_text or "illustration" in tag_text:
        return "illustration"
    if media_category == "animated":
        return "mixed"
    return "unknown"


def merge_tags(*tag_lists: Any) -> List[str]:
    seen: Set[str] = set()
    merged: List[str] = []
    for raw in tag_lists:
        for tag in _normalize_tags(raw):
            if tag not in seen:
                seen.add(tag)
                merged.append(tag)
    return merged


def build_rich_caption(item: Dict[str, Any]) -> str:
    base = (item.get("caption") or item.get("title") or "").strip()
    tags = merge_tags(item.get("tags"), item.get("media_category"), item.get("subject_type"))
    media_cat = item.get("media_category")
    kind = item.get("kind") or "text"
    parts: List[str] = []
    if base:
        parts.append(base)
    label_bits = []
    if media_cat and media_cat not in base.lower():
        label_bits.append(media_cat)
    if kind in ("image", "video"):
        label_bits.append(kind)
    subject = item.get("subject_type")
    if subject and subject not in ("unknown",) and subject not in label_bits:
        label_bits.append(subject)
    extra_tags = [t for t in tags if t not in base.lower() and t not in label_bits][:8]
    if label_bits:
        parts.append("[" + ", ".join(label_bits) + "]")
    if extra_tags:
        parts.append("tags: " + ", ".join(extra_tags))
    return " | ".join(parts) if parts else "nsfw content"


def enrich_media_item(item: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    enriched = dict(item)
    subreddit = enriched.get("subreddit") or ""
    if not subreddit and enriched.get("url"):
        match = re.search(r"reddit\.com/r/([\w]+)", str(enriched.get("url")), re.I)
        if match:
            subreddit = match.group(1)
            enriched["subreddit"] = subreddit
    configured = enriched.get("media_category")
    tags = merge_tags(
        enriched.get("tags"),
        enriched.get("search_tags"),
        subreddit,
        enriched.get("source_group"),
    )
    media_category = infer_media_category(
        subreddit=subreddit,
        tags=tags,
        source=str(enriched.get("source") or ""),
        url=str(enriched.get("url") or enriched.get("image_url") or enriched.get("video_url") or ""),
        configured=configured,
    )
    enriched["media_category"] = media_category
    enriched["tags"] = tags
    enriched["subject_type"] = enriched.get("subject_type") or infer_subject_type(media_category, tags)
    enriched["style_type"] = enriched.get("style_type") or infer_style_type(media_category, tags)
    if enriched.get("kind") in ("image", "video"):
        enriched["caption"] = build_rich_caption(enriched)
    if cfg and cfg.get("unrestricted_mode"):
        enriched.setdefault("training_scope", "nsfw")
    return enriched


def enrich_items(items: List[Dict[str, Any]], cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    return [enrich_media_item(item, cfg) for item in items]


def apply_mix_weights(items: List[Dict[str, Any]], mix_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not mix_cfg.get("enabled", True):
        return items
    weights = {
        key: float(mix_cfg.get(key, 0))
        for key in MEDIA_CATEGORIES
        if key != "general" and float(mix_cfg.get(key, 0)) > 0
    }
    if not weights:
        return items
    total_weight = sum(weights.values()) or 1.0
    buckets: Dict[str, List[Dict[str, Any]]] = {cat: [] for cat in weights}
    general: List[Dict[str, Any]] = []
    for item in items:
        cat = item.get("media_category") or "general"
        if cat in buckets:
            buckets[cat].append(item)
        else:
            general.append(item)
    max_total = int(mix_cfg.get("max_items") or len(items) or 100)
    blended: List[Dict[str, Any]] = []
    for cat, weight in weights.items():
        quota = max(1, int(max_total * (weight / total_weight)))
        pool = buckets.get(cat) or []
        blended.extend(pool[:quota])
    remaining = max_total - len(blended)
    if remaining > 0:
        overflow: List[Dict[str, Any]] = []
        for cat, pool in buckets.items():
            quota = max(1, int(max_total * (weights[cat] / total_weight)))
            overflow.extend(pool[quota:])
        overflow.extend(general)
        blended.extend(overflow[:remaining])
    return blended


def manifest_path() -> Path:
    path = get_project_root() / "data" / "nsfw" / "media_manifest.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def append_media_manifest(items: List[Dict[str, Any]]) -> int:
    path = manifest_path()
    count = 0
    with open(path, "a", encoding="utf-8") as handle:
        for item in items:
            if item.get("kind") not in ("image", "video"):
                continue
            record = {
                "source_id": item.get("source_id"),
                "source": item.get("source"),
                "kind": item.get("kind"),
                "media_category": item.get("media_category"),
                "subject_type": item.get("subject_type"),
                "style_type": item.get("style_type"),
                "tags": item.get("tags") or [],
                "caption": item.get("caption"),
                "url": item.get("url"),
                "image_url": item.get("image_url"),
                "video_url": item.get("video_url"),
                "subreddit": item.get("subreddit"),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_media_manifest(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    path = manifest_path()
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if limit and len(rows) > limit:
        return rows[-limit:]
    return rows
