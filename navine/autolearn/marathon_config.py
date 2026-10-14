from pathlib import Path
from typing import Any, Dict

import yaml

from navine.autolearn.config import load_config as load_autolearn_config
from navine.utils.paths import get_project_root

DEFAULT_MARATHON: Dict[str, Any] = {
    "default_hours": 8,
    "sleep_between_requests": 1.0,
    "sleep_between_cycles": 30,
    "train_every_samples": 50,
    "train_every_minutes": 30,
    "deep_train_every_hours": 2,
    "max_items_per_cycle": 200,
    "sources_all": True,
    "learn_images_per_cycle": 15,
    "learn_videos_per_cycle": 5,
    "learn_coding_languages_rotate": True,
    "github_token_env": "GITHUB_TOKEN",
    "save_state_every_minutes": 5,
    "log_file": "data/autolearn/marathon.log",
    "image_topics": [
        "nature", "technology", "science", "architecture", "programming",
    ],
    "gif_urls": [],
    "wikipedia_categories": ["Computer_science", "Mathematics", "Physics"],
    "reddit_subreddits": ["Python", "javascript", "rust", "webdev", "programming"],
    "github_languages": [
        "python", "javascript", "typescript", "rust", "go", "java",
        "cpp", "csharp", "ruby", "php", "swift", "kotlin", "lua", "shell",
    ],
    "train_types": [
        "coding", "chat", "general", "creative", "math", "multimodal-text", "games",
    ],
    "ingest_conversations_every_cycle": True,
    "max_log_bytes": 5 * 1024 * 1024,
    "max_used_queries": 10000,
}


def marathon_config_path() -> Path:
    return get_project_root() / "configs" / "marathon.yaml"


def load_marathon_config() -> Dict[str, Any]:
    path = marathon_config_path()
    merged = dict(DEFAULT_MARATHON)
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        merged.update(loaded)
    autolearn = load_autolearn_config()
    for key in (
        "github_languages", "reddit_subreddits", "wikipedia_categories",
        "train_types", "github_max_repos", "github_max_files",
        "github_include_gists", "github_include_awesome",
        "stackoverflow_tags", "feed_urls", "docs_topics",
    ):
        if key in autolearn and key not in merged:
            merged[key] = autolearn[key]
    return merged
