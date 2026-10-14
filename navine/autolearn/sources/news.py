from typing import Any, Dict, List, Optional

from navine.autolearn.sources.feeds import fetch_feed

DEFAULT_NEWS_FEEDS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://www.theguardian.com/world/rss",
    "https://www.theguardian.com/technology/rss",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://www.reuters.com/world/rss",
]


def fetch(config: Dict[str, Any], max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    urls = config.get("news_feed_urls") or DEFAULT_NEWS_FEEDS
    per_feed = max(2, (max_items or 30) // max(1, len(urls)))
    items: List[Dict[str, Any]] = []
    for url in urls:
        try:
            batch = fetch_feed(url, max_items=per_feed)
            for entry in batch:
                entry["source"] = "news"
                entry["source_id"] = f"news:{entry.get('source_id', url)}"
                items.append(entry)
        except Exception:
            continue
        if max_items and len(items) >= max_items:
            break
    return items[: max_items or len(items)]
