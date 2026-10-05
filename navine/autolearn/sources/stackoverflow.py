from typing import Any, Dict, List, Optional

from navine.autolearn.http import learning_http_get

DEFAULT_TAGS = [
    "python",
    "javascript",
    "java",
    "c++",
    "rust",
    "algorithm",
    "data-structures",
    "typescript",
    "go",
    "c#",
]


def fetch_stackoverflow(max_items: int = 20, tag: Optional[str] = None) -> List[Dict[str, Any]]:
    tag_filter = f";tagged={tag}" if tag else ""
    url = (
        "https://api.stackexchange.com/2.3/questions"
        f"?order=desc&sort=votes&site=stackoverflow&pagesize={max_items}"
        f"{tag_filter}&filter=withbody"
    )
    response = learning_http_get(url)
    if response.status_code != 200:
        return []
    payload = response.json()
    items: List[Dict[str, Any]] = []
    for question in payload.get("items", []):
        qid = question.get("question_id")
        title = (question.get("title") or "").strip()
        body = (question.get("body") or "").strip()
        if not title:
            continue
        link = question.get("link") or f"https://stackoverflow.com/questions/{qid}"
        text = f"{title}\n\n{body}"[:8000]
        items.append(
            {
                "source": "stackoverflow",
                "source_id": f"so:{qid}",
                "title": title,
                "text": text,
                "url": link,
                "kind": "qa",
            }
        )
    return items


def fetch(config: Dict[str, Any], max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    limit = max_items or 20
    tags = config.get("stackoverflow_tags") or DEFAULT_TAGS
    per_tag = max(3, limit // max(1, len(tags)))
    items: List[Dict[str, Any]] = []
    for tag in tags:
        try:
            batch = fetch_stackoverflow(max_items=per_tag, tag=tag)
            items.extend(batch)
        except Exception:
            continue
        if len(items) >= limit:
            break
    if not items:
        try:
            items = fetch_stackoverflow(max_items=limit)
        except Exception:
            return []
    return items[:limit]
