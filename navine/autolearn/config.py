from pathlib import Path
from typing import Any, Dict

import yaml

from navine.utils.paths import get_project_root

DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "sources": [
        "github",
        "gitlab",
        "reddit",
        "hackernews",
        "stackoverflow",
        "wikipedia",
        "arxiv",
        "feeds",
        "docs",
        "gutenberg",
    ],
    "interval_minutes": 60,
    "min_samples_before_train": 25,
    "max_items_per_run": 500,
    "aggressive_max_items": 500,
    "reddit_subreddits": [
        "learnprogramming",
        "Python",
        "javascript",
        "rust",
        "golang",
        "cpp",
        "java",
        "webdev",
        "cscareerquestions",
        "ExperiencedDevs",
        "programming",
        "coding",
        "learnjavascript",
        "MachineLearning",
        "datascience",
        "devops",
        "sysadmin",
    ],
    "github_languages": [
        "python",
        "javascript",
        "typescript",
        "rust",
        "go",
        "java",
        "cpp",
        "csharp",
        "ruby",
        "php",
        "swift",
        "kotlin",
        "lua",
        "shell",
    ],
    "github_max_repos": 8,
    "github_max_files": 40,
    "github_include_gists": True,
    "github_include_awesome": True,
    "cycle_all_languages": True,
    "stackoverflow_tags": [
        "python",
        "javascript",
        "java",
        "c++",
        "rust",
        "algorithm",
        "data-structures",
        "typescript",
        "go",
    ],
    "auto_train": True,
    "train_types": [
        "coding",
        "chat",
        "general",
        "creative",
        "math",
        "multimodal-text",
    ],
    "learn_images": True,
    "learn_videos": False,
    "image_topics": ["nature", "technology", "science", "programming"],
    "image_max_per_topic": 3,
    "docs_topics": ["python", "javascript"],
    "wikipedia_categories": ["Computer_science", "Mathematics", "Physics"],
    "feed_urls": [
        "https://blog.rust-lang.org/feed.xml",
        "https://hnrss.org/frontpage",
        "https://dev.to/feed",
        "https://medium.com/feed/tag/programming",
        "https://www.reddit.com/r/programming/.rss",
    ],
    "code_hosts": ["gitlab", "codeberg"],
}


def config_path() -> Path:
    return get_project_root() / "configs" / "autolearn.yaml"


def load_config() -> Dict[str, Any]:
    path = config_path()
    merged = dict(DEFAULT_CONFIG)
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        merged.update(loaded)
    return merged


def save_config(config: Dict[str, Any]) -> str:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, default_flow_style=False, sort_keys=False)
    return str(path)
