from typing import Any, Dict, List, Optional

from navine.autolearn.sources import (
    arxiv,
    docs,
    feeds,
    github,
    gitlab,
    gutenberg,
    hackernews,
    humans,
    news,
    reddit,
    stackoverflow,
    wikipedia,
)

try:
    from navine.autolearn.sources import nsfw as nsfw_source
except ImportError:
    nsfw_source = None

SOURCE_MODULES = {
    "github": github,
    "gitlab": gitlab,
    "codeberg": gitlab,
    "reddit": reddit,
    "humans": humans,
    "news": news,
    "hackernews": hackernews,
    "hn": hackernews,
    "stackoverflow": stackoverflow,
    "stackexchange": stackoverflow,
    "feeds": feeds,
    "wikipedia": wikipedia,
    "arxiv": arxiv,
    "gutenberg": gutenberg,
    "docs": docs,
}

if nsfw_source is not None:
    SOURCE_MODULES["nsfw"] = nsfw_source


def fetch_source(name: str, config: Dict[str, Any], max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    module = SOURCE_MODULES.get(name.lower())
    if not module:
        return []
    return module.fetch(config, max_items=max_items)


def fetch_all_sources(config: Dict[str, Any], max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    sources = config.get("sources") or list(SOURCE_MODULES.keys())
    per_source = max_items
    if max_items and sources:
        per_source = max(1, max_items // len(sources))
    for name in sources:
        try:
            batch = fetch_source(name, config, max_items=per_source)
            items.extend(batch)
        except Exception:
            continue
    if max_items and len(items) > max_items:
        return items[:max_items]
    return items
