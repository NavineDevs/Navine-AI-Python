import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from navine.autolearn.http import learning_http_get

STRIP_TAGS = re.compile(r"<[^>]+>")


def _strip_html(value: str) -> str:
    return STRIP_TAGS.sub("", value).strip()


def _parse_feed_xml(content: str) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return entries
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for item in root.findall(".//item"):
        title = _strip_html(item.findtext("title") or "")
        link = item.findtext("link") or ""
        description = _strip_html(item.findtext("description") or "")
        if title or description:
            entries.append({"title": title, "link": link, "text": description or title})
    if entries:
        return entries
    for entry in root.findall(".//atom:entry", ns):
        title = _strip_html(entry.findtext("atom:title", default="", namespaces=ns) or "")
        link_el = entry.find("atom:link", ns)
        link = link_el.get("href", "") if link_el is not None else ""
        summary = _strip_html(
            entry.findtext("atom:summary", default="", namespaces=ns)
            or entry.findtext("atom:content", default="", namespaces=ns)
            or ""
        )
        if title or summary:
            entries.append({"title": title, "link": link, "text": summary or title})
    return entries


def fetch_feed(url: str, max_items: int = 10) -> List[Dict[str, Any]]:
    response = learning_http_get(url)
    if response.status_code != 200:
        return []
    entries = _parse_feed_xml(response.text)
    items: List[Dict[str, Any]] = []
    for idx, entry in enumerate(entries[:max_items]):
        title = entry.get("title", "")
        text = entry.get("text", "")
        link = entry.get("link", url)
        if not text:
            continue
        items.append(
            {
                "source": "feeds",
                "source_id": f"feed:{url}:{idx}",
                "title": title,
                "text": text[:8000],
                "url": link,
            }
        )
    return items


def fetch(config: Dict[str, Any], max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    urls = config.get("feed_urls") or []
    if not urls:
        return []
    per_feed = max(3, (max_items or 10) // max(1, len(urls)))
    items: List[Dict[str, Any]] = []
    for url in urls:
        try:
            batch = fetch_feed(url, max_items=per_feed)
            items.extend(batch)
        except Exception:
            continue
        if max_items and len(items) >= max_items:
            break
    if max_items:
        return items[:max_items]
    return items
