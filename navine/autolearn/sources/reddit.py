import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from navine.autolearn.http import learning_http_get

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _parse_reddit_rss(content: str, subreddit: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return items
    for idx, entry in enumerate(root.findall("atom:entry", ATOM_NS)):
        title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
        link_el = entry.find("atom:link", ATOM_NS)
        link = link_el.get("href", "") if link_el is not None else ""
        entry_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS) or f"{subreddit}:{idx}"
        content_el = entry.find("atom:content", ATOM_NS)
        text = ""
        if content_el is not None and content_el.text:
            text = content_el.text.strip()
        summary = entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or ""
        if not text:
            text = summary.strip()
        if not title and not text:
            continue
        items.append(
            {
                "source": "reddit",
                "source_id": f"reddit:{subreddit}:{entry_id}",
                "title": title,
                "text": text or title,
                "url": link,
            }
        )
    return items


def fetch_reddit(subreddit: str, max_posts: int = 25) -> List[Dict[str, Any]]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    json_urls = [
        f"https://www.reddit.com/r/{subreddit}/hot.json?limit={max_posts}&raw_json=1",
        f"https://old.reddit.com/r/{subreddit}/hot.json?limit={max_posts}&raw_json=1",
    ]
    try:
        for json_url in json_urls:
            response = learning_http_get(json_url, headers=headers)
            if response.status_code == 200:
                payload = response.json()
                children = payload.get("data", {}).get("children", [])
                items: List[Dict[str, Any]] = []
                for child in children:
                    data = child.get("data", {})
                    post_id = data.get("id")
                    title = (data.get("title") or "").strip()
                    selftext = (data.get("selftext") or "").strip()
                    if not title and not selftext:
                        continue
                    permalink = data.get("permalink", "")
                    post_url = f"https://www.reddit.com{permalink}" if permalink else data.get("url", "")
                    text = selftext or title
                    items.append(
                        {
                            "source": "reddit",
                            "source_id": f"reddit:{subreddit}:{post_id}",
                            "title": title,
                            "text": text,
                            "url": post_url,
                        }
                    )
                if items:
                    return items[:max_posts]
    except Exception:
        pass
    rss_url = f"https://www.reddit.com/r/{subreddit}/.rss"
    try:
        response = learning_http_get(rss_url, headers={"Accept": "application/atom+xml"})
        if response.status_code != 200:
            return []
        items = _parse_reddit_rss(response.text, subreddit)
        return items[:max_posts]
    except Exception:
        return []


def fetch(config: Dict[str, Any], max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    subreddits = config.get("reddit_subreddits") or ["Python"]
    per_sub = max(5, (max_items or 25) // max(1, len(subreddits)))
    items: List[Dict[str, Any]] = []
    for subreddit in subreddits:
        try:
            batch = fetch_reddit(subreddit, max_posts=per_sub)
            items.extend(batch)
        except Exception:
            continue
        if max_items and len(items) >= max_items:
            break
    if max_items:
        return items[:max_items]
    return items
