from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from navine.utils.paths import get_project_root

_config_cache: Optional[Dict[str, Any]] = None

UNRESTRICTED_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "unrestricted_mode": True,
    "reddit_subreddits": [],
    "local_dirs": [
        "data/nsfw/local",
        "data/nsfw/local/hentai",
        "data/nsfw/local/porn",
        "data/nsfw/local/real",
        "data/nsfw/local/human",
        "data/nsfw/local/mixed",
        "data/nsfw/local/animated",
        "data/nsfw/local/cosplay",
        "data/nsfw/local/fetish",
        "data/unrestricted/local",
        "data/unrestricted/local/security",
        "data/unrestricted/local/ethics",
        "data/unrestricted/local/controversial",
    ],
    "max_items_per_run": 120,
    "max_reddit_posts": 30,
    "include_images": True,
    "include_videos": True,
    "include_text": True,
    "include_unrestricted": True,
    "mixed_mode": True,
    "train_on_collect": True,
    "auto_ingest_on_generate": True,
    "auto_learn": True,
    "media_mix": {
        "enabled": True,
        "hentai": 0.35,
        "real": 0.30,
        "human": 0.15,
        "animated": 0.10,
        "mixed": 0.10,
    },
}


def load_nsfw_config() -> Dict[str, Any]:
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    merged = dict(UNRESTRICTED_DEFAULTS)
    path = get_project_root() / "configs" / "nsfw.yaml"
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        merged.update(loaded)
    navine_path = get_project_root() / "configs" / "navine.yaml"
    if navine_path.exists():
        with open(navine_path, "r", encoding="utf-8") as handle:
            navine_cfg = yaml.safe_load(handle) or {}
        if navine_cfg.get("unrestricted_mode") is not False:
            merged["unrestricted_mode"] = True
    merged["unrestricted_mode"] = True
    merged["enabled"] = bool(merged.get("enabled", True))
    _config_cache = merged
    return merged


def is_nsfw_enabled() -> bool:
    return bool(load_nsfw_config().get("enabled", True))


def is_unrestricted_mode() -> bool:
    return True


def nsfw_config_path() -> Path:
    return get_project_root() / "configs" / "nsfw.yaml"


def reload_nsfw_config() -> Dict[str, Any]:
    global _config_cache
    _config_cache = None
    return load_nsfw_config()


def _dedupe_subs(values: List[str]) -> List[str]:
    seen: List[str] = []
    for sub in values:
        name = str(sub).strip()
        if name and name not in seen:
            seen.append(name)
    return seen


def _subreddit_list(cfg: Dict[str, Any], key: str) -> List[str]:
    return _dedupe_subs([str(s).strip() for s in (cfg.get(key) or []) if str(s).strip()])


def configured_subreddits(cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    merged = cfg or load_nsfw_config()
    keys = (
        "reddit_hentai_subreddits",
        "reddit_real_subreddits",
        "reddit_human_subreddits",
        "reddit_mixed_subreddits",
        "reddit_hentai_video_subreddits",
        "reddit_real_video_subreddits",
        "reddit_text_subreddits",
        "reddit_image_subreddits",
        "reddit_video_subreddits",
        "reddit_unrestricted_subreddits",
        "reddit_subreddits",
    )
    seen: List[str] = []
    for key in keys:
        for sub in merged.get(key) or []:
            name = str(sub).strip()
            if name and name not in seen:
                seen.append(name)
    return seen


def configured_source_groups(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    merged = cfg or load_nsfw_config()
    groups: Dict[str, Dict[str, Any]] = {}
    defaults = {
        "hentai": ("reddit_hentai_subreddits", "reddit_image_subreddits", "hentai", ["hentai", "animated", "illustration"]),
        "real": ("reddit_real_subreddits", None, "real", ["real", "photography"]),
        "human": ("reddit_human_subreddits", None, "human", ["human", "real"]),
        "mixed": ("reddit_mixed_subreddits", "reddit_subreddits", "mixed", ["mixed", "nsfw"]),
        "hentai_video": ("reddit_hentai_video_subreddits", "reddit_video_subreddits", "hentai", ["hentai", "video", "animated"]),
        "real_video": ("reddit_real_video_subreddits", None, "real", ["real", "human", "video"]),
    }
    explicit = merged.get("reddit_source_groups") or {}
    if isinstance(explicit, dict) and explicit:
        for name, spec in explicit.items():
            if not isinstance(spec, dict):
                continue
            subs = _dedupe_subs([str(s) for s in (spec.get("subreddits") or []) if str(s).strip()])
            groups[name] = {
                "subreddits": subs,
                "media_category": str(spec.get("media_category") or name),
                "media_type": str(spec.get("media_type") or "any"),
                "tags": _dedupe_subs([str(t) for t in (spec.get("tags") or [])]),
            }
        return groups
    for name, (primary_key, fallback_key, media_category, tags) in defaults.items():
        subs = _subreddit_list(merged, primary_key)
        if not subs and fallback_key:
            subs = _subreddit_list(merged, fallback_key)
        if subs:
            media_type = "video" if "video" in name else "image"
            if name == "mixed":
                media_type = "any"
            groups[name] = {
                "subreddits": subs,
                "media_category": media_category,
                "media_type": media_type,
                "tags": tags,
            }
    return groups


def subreddit_media_map(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Tuple[str, List[str]]]:
    merged = cfg or load_nsfw_config()
    mapping: Dict[str, Tuple[str, List[str]]] = {}
    for name, group in configured_source_groups(merged).items():
        media_category = group.get("media_category") or name
        tags = list(group.get("tags") or [])
        for sub in group.get("subreddits") or []:
            mapping[str(sub).lower()] = (media_category, tags)
    for sub in _subreddit_list(merged, "reddit_unrestricted_subreddits"):
        mapping[sub.lower()] = ("general", ["unrestricted"])
    return mapping


def configured_unrestricted_topics(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    merged = cfg or load_nsfw_config()
    topics = merged.get("unrestricted_text_topics") or {}
    if topics:
        return topics if isinstance(topics, dict) else {}
    return {
        "security": {
            "local_dirs": ["data/unrestricted/local/security"],
            "tags": ["security", "malware-analysis", "defensive"],
            "category": "unrestricted",
        },
        "ethics": {
            "local_dirs": ["data/unrestricted/local/ethics"],
            "tags": ["ethics", "unfiltered", "dialogue"],
            "category": "unrestricted",
        },
        "controversial": {
            "local_dirs": ["data/unrestricted/local/controversial"],
            "tags": ["controversial", "unfiltered"],
            "category": "unrestricted",
        },
    }


def ensure_local_dirs(cfg: Optional[Dict[str, Any]] = None) -> List[Path]:
    merged = cfg or load_nsfw_config()
    created: List[Path] = []
    dirs: List[str] = list(merged.get("local_dirs") or [])
    for topic_cfg in configured_unrestricted_topics(merged).values():
        if isinstance(topic_cfg, dict):
            dirs.extend(topic_cfg.get("local_dirs") or [])
    for rel in _dedupe_subs(dirs):
        directory = get_project_root() / rel
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            created.append(directory)
    return created


def local_dir_status(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    merged = cfg or load_nsfw_config()
    dirs: List[str] = []
    seen_files: set = set()
    for rel in merged.get("local_dirs") or []:
        directory = get_project_root() / rel
        dirs.append(str(rel))
        if directory.exists():
            for path in directory.rglob("*"):
                if path.is_file():
                    seen_files.add(str(path.resolve()))
    return {"dirs": dirs, "file_count": len(seen_files)}


def configured_internet_sources(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    merged = cfg or load_nsfw_config()
    fourchan = merged.get("fourchan") or {}
    fourchan_boards = 0
    fourchan_groups = 0
    if isinstance(fourchan, dict) and fourchan.get("enabled", False):
        try:
            from navine.autolearn.sources.fourchan import all_configured_boards, board_groups

            fourchan_boards = len(all_configured_boards(merged))
            fourchan_groups = len(board_groups(merged))
        except Exception:
            fourchan_boards = len(fourchan.get("boards_all") or fourchan.get("boards") or [])
            fourchan_groups = len(fourchan.get("board_groups") or {})
    return {
        "reddit_post_urls": len(merged.get("reddit_post_urls") or []),
        "web_urls": len(merged.get("web_urls") or []),
        "crawl_start_urls": len(merged.get("crawl_start_urls") or []),
        "direct_media_urls": len(merged.get("direct_media_urls") or []),
        "feed_urls": len(merged.get("feed_urls") or []),
        "image_search_sources": len(merged.get("image_search_sources") or []),
        "video_search_sources": len(merged.get("video_search_sources") or []),
        "image_search_queries": len(merged.get("image_search_queries") or []),
        "fourchan_boards": fourchan_boards,
        "fourchan_groups": fourchan_groups,
        "waifu_im_enabled": bool((merged.get("waifu_im") or {}).get("enabled", False)),
        "infini_atomic_enabled": bool((merged.get("infini_atomic") or {}).get("enabled", False)),
        "source_groups": len(configured_source_groups(merged)),
        "unrestricted_topics": len(configured_unrestricted_topics(merged)),
    }


def media_mix_config(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    merged = cfg or load_nsfw_config()
    mix = dict(merged.get("media_mix") or {})
    mix.setdefault("enabled", bool(merged.get("mixed_mode", True)))
    mix.setdefault("max_items", int(merged.get("max_items_per_run", 120)))
    return mix
