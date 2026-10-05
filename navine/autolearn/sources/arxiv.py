import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from navine.autolearn.http import learning_http_get

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _parse_arxiv_feed(content: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return items
    for entry in root.findall("atom:entry", ATOM_NS):
        entry_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS)
        title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").strip()
        if not title:
            continue
        link_el = entry.find("atom:link", ATOM_NS)
        link = link_el.get("href", entry_id) if link_el is not None else entry_id
        text = f"{title}\n\n{summary}".strip()
        if len(text) < 50:
            continue
        arxiv_id = entry_id.rsplit("/", 1)[-1] if entry_id else title
        items.append(
            {
                "source": "arxiv",
                "source_id": f"arxiv:{arxiv_id}",
                "title": title,
                "text": text[:8000],
                "url": link,
                "category": "technical",
            }
        )
    return items


def fetch_arxiv(max_items: int = 20, query: Optional[str] = None) -> List[Dict[str, Any]]:
    search = query or "cat:cs.* OR cat:stat.ML"
    url = (
        "https://export.arxiv.org/api/query"
        f"?search_query={search}&start=0&max_results={max_items}"
        "&sortBy=submittedDate&sortOrder=descending"
    )
    try:
        response = learning_http_get(url)
        if response.status_code != 200:
            return []
        return _parse_arxiv_feed(response.text)[:max_items]
    except Exception:
        return []


def fetch(config: Dict[str, Any], max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    limit = max_items or 20
    query = config.get("arxiv_query")
    try:
        return fetch_arxiv(max_items=limit, query=query)
    except Exception:
        return []
