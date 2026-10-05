from typing import Any, Dict, List, Optional

from navine.autolearn.http import learning_http_get


def _fetch_item(item_id: int) -> Optional[Dict[str, Any]]:
    url = f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
    try:
        response = learning_http_get(url, sleep=0.5)
        if response.status_code != 200:
            return None
        return response.json()
    except Exception:
        return None


def fetch_hackernews(max_items: int = 30) -> List[Dict[str, Any]]:
    response = learning_http_get("https://hacker-news.firebaseio.com/v0/topstories.json")
    if response.status_code != 200:
        return []
    story_ids = response.json()[:max_items]
    items: List[Dict[str, Any]] = []
    for story_id in story_ids:
        data = _fetch_item(story_id)
        if not data:
            continue
        if data.get("type") != "story":
            continue
        title = (data.get("title") or "").strip()
        text = (data.get("text") or "").strip()
        url = data.get("url") or f"https://news.ycombinator.com/item?id={story_id}"
        body = text or title
        if not body:
            continue
        items.append(
            {
                "source": "hackernews",
                "source_id": f"hn:{story_id}",
                "title": title,
                "text": body[:8000],
                "url": url,
            }
        )
    return items


def fetch(config: Dict[str, Any], max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    limit = max_items or 30
    try:
        return fetch_hackernews(max_items=limit)
    except Exception:
        return []
