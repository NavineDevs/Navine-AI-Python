import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from navine.autolearn.http import learning_http_get

STRIP_TAGS = re.compile(r"<[^>]+>")
WHITESPACE = re.compile(r"\s+")

DOC_TOPICS = {
    "python": [
        "https://docs.python.org/3/tutorial/index.html",
        "https://docs.python.org/3/library/functions.html",
        "https://docs.python.org/3/reference/datamodel.html",
    ],
    "javascript": [
        "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide",
        "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference",
    ],
    "html": [
        "https://developer.mozilla.org/en-US/docs/Web/HTML",
    ],
    "css": [
        "https://developer.mozilla.org/en-US/docs/Web/CSS",
    ],
}


def _strip_html(html: str) -> str:
    text = STRIP_TAGS.sub(" ", html)
    return WHITESPACE.sub(" ", text).strip()


def _extract_title(html: str) -> str:
    match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _extract_links(html: str, base_url: str, same_host: str) -> List[str]:
    links = []
    seen = set()
    for match in re.finditer(r'href=["\']([^"\']+)["\']', html, re.IGNORECASE):
        href = match.group(1)
        if href.startswith("#") or href.startswith("mailto:"):
            continue
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        if parsed.netloc != same_host:
            continue
        if full in seen:
            continue
        seen.add(full)
        links.append(full)
    return links


def fetch_doc_page(url: str, max_chars: int = 8000) -> Optional[Dict[str, Any]]:
    try:
        response = learning_http_get(url, timeout=45)
        if response.status_code != 200:
            return None
        html = response.text
        title = _extract_title(html)
        text = _strip_html(html)
        if len(text) < 200:
            return None
        return {
            "source": "docs",
            "source_id": f"docs:{url}",
            "title": title or url,
            "text": text[:max_chars],
            "url": url,
            "kind": "technical",
        }
    except Exception:
        return None


def crawl_docs(topic: str = "python", max_pages: int = 5, max_depth: int = 1) -> List[Dict[str, Any]]:
    start_urls = DOC_TOPICS.get(topic, DOC_TOPICS["python"])
    items: List[Dict[str, Any]] = []
    visited = set()
    queue: List[tuple] = [(url, 0) for url in start_urls[:2]]
    while queue and len(items) < max_pages:
        url, depth = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        try:
            response = learning_http_get(url, timeout=45)
            if response.status_code != 200:
                continue
            html = response.text
            title = _extract_title(html)
            text = _strip_html(html)
            if len(text) >= 200:
                items.append(
                    {
                        "source": "docs",
                        "source_id": f"docs:{url}",
                        "title": title or url,
                        "text": text[:8000],
                        "url": url,
                        "kind": "technical",
                    }
                )
            if depth < max_depth:
                host = urlparse(url).netloc
                for link in _extract_links(html, url, host):
                    if link not in visited and len(queue) < max_pages * 3:
                        queue.append((link, depth + 1))
        except Exception:
            continue
    return items[:max_pages]


def fetch(config: Dict[str, Any], max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    limit = max_items or 10
    topics = config.get("docs_topics") or ["python", "javascript"]
    per_topic = max(2, limit // max(1, len(topics)))
    items: List[Dict[str, Any]] = []
    for topic in topics:
        try:
            batch = crawl_docs(topic=topic, max_pages=per_topic, max_depth=1)
            items.extend(batch)
        except Exception:
            continue
        if len(items) >= limit:
            break
    return items[:limit]
