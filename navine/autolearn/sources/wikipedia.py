import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from navine.autolearn.http import learning_http_get

STRIP_TAGS = __import__("re").compile(r"<[^>]+>")


def _strip_html(value: str) -> str:
    return STRIP_TAGS.sub("", value).strip()


def fetch_wikipedia(max_items: int = 10, category: Optional[str] = None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if category:
        url = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&format=json&list=categorymembers"
            f"&cmtitle=Category:{quote(category.replace(' ', '_'))}"
            f"&cmlimit={max_items}&cmtype=page&cmnamespace=0"
        )
        try:
            response = learning_http_get(url)
            if response.status_code != 200:
                return []
            members = response.json().get("query", {}).get("categorymembers", [])
            titles = [m.get("title", "") for m in members if m.get("title")]
        except Exception:
            return []
    else:
        url = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&format=json&list=random&rnnamespace=0&rnlimit={max_items}"
        )
        try:
            response = learning_http_get(url)
            if response.status_code != 200:
                return []
            random_pages = response.json().get("query", {}).get("random", [])
            titles = [p.get("title", "") for p in random_pages if p.get("title")]
        except Exception:
            return []
    if not titles:
        return []
    titles_param = "|".join(t.replace(" ", "_") for t in titles[:max_items])
    extract_url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=query&format=json&prop=extracts&explaintext=1&exintro=0"
        f"&titles={quote(titles_param, safe='|')}"
    )
    try:
        response = learning_http_get(extract_url)
        if response.status_code != 200:
            return []
        pages = response.json().get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id == "-1":
                continue
            title = page.get("title", "")
            text = (page.get("extract") or "").strip()
            if not text or len(text) < 100:
                continue
            page_url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
            items.append(
                {
                    "source": "wikipedia",
                    "source_id": f"wiki:{page_id}",
                    "title": title,
                    "text": text[:8000],
                    "url": page_url,
                    "category": "general",
                }
            )
    except Exception:
        return []
    return items[:max_items]


def fetch(config: Dict[str, Any], max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    limit = max_items or 10
    categories = config.get("wikipedia_categories") or []
    items: List[Dict[str, Any]] = []
    if categories:
        per_cat = max(2, limit // max(1, len(categories)))
        for category in categories:
            try:
                batch = fetch_wikipedia(max_items=per_cat, category=category)
                items.extend(batch)
            except Exception:
                continue
            if len(items) >= limit:
                break
    if len(items) < limit:
        try:
            batch = fetch_wikipedia(max_items=limit - len(items))
            items.extend(batch)
        except Exception:
            pass
    return items[:limit]
